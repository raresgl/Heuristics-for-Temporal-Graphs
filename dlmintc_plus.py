"""
DLMinTC+ (Lazzarinetti et al., Algorithms 2025) — faithful reimplementation.

This is a best-effort port of the *original* DLMinTC+ deep model, built so we can
run a direct head-to-head against our binary-mask + model-driven-assignment
approach on identical data. Because we train it with the same encoder, the same
k=1 data, and evaluate against the same ILP optimum, the comparison isolates the
two DECODING PARADIGMS (their Pointer-Network + greedy fix-up vs our mask + DP)
while holding everything else constant.

Architecture (paper §4, Table 1)
--------------------------------
  node feature : per-(v,t) degree
  GraphSAGE    : 3 layers, embedding dim 32 (batched supergraph, as in model.py)
  Transformer  : temporal encoder over each node's T-length sequence + sinusoidal PE
  Pointer head : two attention pointers per node -> (start, end) positions.
                 Soft (expected) positions during training for differentiability;
                 argmax positions at inference.

Loss (paper Eqs. 12-16), log-additive with weights (alpha,beta,gamma)=(0.2,0.5,0.3):
  coverage : sum over edges of 1{both endpoints inactive at t}, ReLU/sigmoid-relaxed
  span     : (1/n) sum_u (s_u - s_hat_u)^2 + (e_u - e_hat_u)^2   (MSE vs observed)
  penalty  : sum_u ReLU(s_u - e_u)^2
  Loss = alpha*log(cov+eps) + beta*log(span+eps) + gamma*log(penalty+eps)

Post-processing (paper Alg. 1): greedy iterative adjustment — for each uncovered
edge (u,v,t), extend whichever endpoint interval needs the smallest stretch to
include t. Locally minimal (their Theorem 1).

NOTE: the paper leaves the coverage-loss relaxation and the exact PN decoder
underspecified; we use a smooth sigmoid relaxation and additive-attention
pointers. This is a good-faith reproduction, not a bit-exact one.
"""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

EPS = 1e-6


# ---------------------------------------------------------------------------
# Positional encoding (same as model.py)
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        self.register_buffer('div', div)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        pos = torch.arange(T, dtype=torch.float, device=x.device).unsqueeze(1)
        pe = torch.zeros(T, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(pos * self.div)
        pe[:, 1::2] = torch.cos(pos * self.div)
        return x + pe.unsqueeze(0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DLMinTCPlus(nn.Module):
    def __init__(self, d: int = 32, sage_layers: int = 3, tf_heads: int = 4,
                 tf_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(1, d), nn.ReLU())
        self.sage = nn.ModuleList(SAGEConv(d, d) for _ in range(sage_layers))
        self.pos_enc = SinusoidalPE(d)
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=tf_heads, dim_feedforward=d * 4,
                                         dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=tf_layers)
        # Two attention pointers: learned query vectors scoring each timestamp position
        self.q_start = nn.Linear(d, 1)
        self.q_end = nn.Linear(d, 1)

    def forward(self, node_features: torch.Tensor, edges: torch.Tensor,
                n_nodes: int, T: int, times: torch.Tensor):
        """
        Returns (s, e, a_start, a_end):
          s, e     : (n_nodes,) soft expected start/end TIMES (differentiable)
          a_start/a_end : (n_nodes, T) attention distributions over timestamp positions
        `times` : (T,) actual timestamp values, float.
        """
        device = node_features.device
        x = node_features.reshape(n_nodes * T, 1)
        x = self.input_proj(x)

        if edges.numel() > 0:
            u_g = edges[:, 0] * T + edges[:, 2]
            v_g = edges[:, 1] * T + edges[:, 2]
            edge_index = torch.stack([torch.cat([u_g, v_g]), torch.cat([v_g, u_g])], dim=0)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

        for conv in self.sage:
            x = F.relu(conv(x, edge_index))

        x = x.reshape(n_nodes, T, -1)
        x = self.pos_enc(x)
        x = self.transformer(x)                         # (n_nodes, T, d)

        start_logits = self.q_start(x).squeeze(-1)       # (n_nodes, T)
        end_logits = self.q_end(x).squeeze(-1)
        a_start = F.softmax(start_logits, dim=1)
        a_end = F.softmax(end_logits, dim=1)

        # expected (soft) start/end times
        s = (a_start * times.unsqueeze(0)).sum(dim=1)     # (n_nodes,)
        e = (a_end * times.unsqueeze(0)).sum(dim=1)
        return s, e, a_start, a_end


# ---------------------------------------------------------------------------
# Loss (paper Eqs. 12-16)
# ---------------------------------------------------------------------------

def _observed_intervals(mask: torch.Tensor, times: torch.Tensor):
    """From a binary (n,T) mask derive observed (s_hat, e_hat) per node (min/max active time)."""
    n, T = mask.shape
    big = times.max()
    s_hat = torch.empty(n, device=mask.device)
    e_hat = torch.empty(n, device=mask.device)
    for u in range(n):
        active = (mask[u] > 0.5).nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            s_hat[u] = 0.0
            e_hat[u] = 0.0
        else:
            s_hat[u] = times[active.min()]
            e_hat[u] = times[active.max()]
    return s_hat, e_hat


def dlmintc_loss(s, e, a_start, a_end, edges, mask, times,
                 alpha=0.2, beta=0.5, gamma=0.3, tau=1.0):
    """Log-additive coverage + span + penalty loss."""
    n = s.size(0)
    # --- coverage: smooth relaxation of 1{both inactive at t} ---
    if edges.numel() > 0:
        u, v, ti = edges[:, 0], edges[:, 1], edges[:, 2]
        t = times[ti]
        # c_x(t) in (0,1): high when s_x <= t <= e_x
        cu = torch.sigmoid(tau * (t - s[u])) * torch.sigmoid(tau * (e[u] - t))
        cv = torch.sigmoid(tau * (t - s[v])) * torch.sigmoid(tau * (e[v] - t))
        coverage = ((1 - cu) * (1 - cv)).sum()
    else:
        coverage = torch.tensor(0.0, device=s.device)

    # --- span: MSE vs observed intervals ---
    s_hat, e_hat = _observed_intervals(mask, times)
    span = ((s - s_hat) ** 2 + (e - e_hat) ** 2).mean()

    # --- penalty: enforce s <= e ---
    penalty = F.relu(s - e).pow(2).sum()

    total = (alpha * torch.log(coverage + EPS)
             + beta * torch.log(span + EPS)
             + gamma * torch.log(penalty + EPS))
    return total, {'cov': float(coverage.detach()), 'span': float(span.detach()),
                   'penalty': float(penalty.detach())}


# ---------------------------------------------------------------------------
# Iterative adjustment (paper Algorithm 1) — greedy fix-up
# ---------------------------------------------------------------------------

def iterative_adjustment(s: torch.Tensor, e: torch.Tensor, edges: torch.Tensor,
                         times: List[int]) -> Dict[int, List[Tuple[int, int]]]:
    """
    Snap soft (s,e) to nearest actual timestamps, then greedily extend intervals
    to cover every temporal edge with minimal total sum-span increase.
    Returns {node: [(start, end)]} (single interval per node; k=1).
    """
    n = s.size(0)
    tarr = times
    # snap to nearest actual timestamp
    ts_t = torch.tensor(tarr, dtype=torch.float)

    def nearest(x):
        return int(tarr[int(torch.argmin((ts_t - float(x)).abs()))])

    sv = [nearest(s[u]) for u in range(n)]
    ev = [nearest(e[u]) for u in range(n)]
    for u in range(n):
        if sv[u] > ev[u]:
            sv[u], ev[u] = ev[u], sv[u]           # repair ordering
    active = [True] * n  # k=1: every node has an interval; empty edges handled below

    # greedy adjustment over uncovered edges
    for uu, vv, ti in edges.tolist():
        t = tarr[ti]
        cu = sv[uu] <= t <= ev[uu]
        cv = sv[vv] <= t <= ev[vv]
        if cu or cv:
            continue
        du = 0 if cu else (sv[uu] - t if t < sv[uu] else t - ev[uu])
        dv = 0 if cv else (sv[vv] - t if t < sv[vv] else t - ev[vv])
        if du <= dv:
            if t < sv[uu]:
                sv[uu] = t
            else:
                ev[uu] = t
        else:
            if t < sv[vv]:
                sv[vv] = t
            else:
                ev[vv] = t

    # nodes that were never needed to cover any edge and have zero-width default
    # interval are kept as degenerate (span 0); only emit nodes touched by edges
    touched = set()
    for uu, vv, _ in edges.tolist():
        touched.add(uu); touched.add(vv)
    return {u: [(sv[u], ev[u])] for u in range(n) if u in touched}
