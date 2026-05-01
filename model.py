"""
DLMinTCk: GraphSAGE + Transformer + binary-mask head for k-MinTemporalCover.

Architecture overview
---------------------
1. k-embedding     : learned vector for the interval budget k, concatenated to
                     every per-(node, time) feature vector.
2. GraphSAGE       : all T temporal snapshots are processed in a single batched
                     pass.  Snapshot t occupies the block of virtual nodes
                     [n * t, n * t + n_nodes) in the supergraph, so the
                     snapshot edges become shifted intra-block edges.
3. Transformer     : for each node independently, the T SAGE embeddings form a
                     sequence; sinusoidal positional encoding is added along T
                     before the encoder layers.
4. Output head     : small MLP → sigmoid → p_v^t ∈ (0,1).

Forward signature
-----------------
  p = model(node_features, edges, k, n_nodes, T)

  node_features : FloatTensor (n_nodes, T)  — per-(v,t) degree from data pipeline
  edges         : LongTensor  (E, 3)        — (u_idx, v_idx, t_idx), 0-indexed
  k             : int                       — interval budget
  n_nodes       : int
  T             : int (number of unique timestamps)

  returns p     : FloatTensor (n_nodes, T)  — sigmoid probabilities
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    """Adds fixed sinusoidal encoding to the T (sequence) dimension.

    Computed on-the-fly so it handles any T without a hard max_len cap.
    The frequency divisors are cached as a buffer so they move with the model's device.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        self.register_buffer('div', div)  # (d_model//2,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, d_model)
        T = x.size(1)
        pos = torch.arange(T, dtype=torch.float, device=x.device).unsqueeze(1)  # (T, 1)
        pe = torch.zeros(T, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(pos * self.div)
        pe[:, 1::2] = torch.cos(pos * self.div)
        return x + pe.unsqueeze(0)  # broadcast over batch dim


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DLMinTCk(nn.Module):
    """
    Single network for k-MinTemporalCover, parameterised by k ∈ [1, max_k].

    Hyper-parameters
    ----------------
    k_emb_dim   : dimension of the learned k embedding
    sage_hidden : hidden dimension inside GraphSAGE (all layers share this width)
    sage_layers : number of SAGEConv layers (≥ 1)
    tf_hidden   : Transformer d_model
    tf_heads    : Transformer attention heads (must divide tf_hidden)
    tf_layers   : Transformer encoder depth
    max_k       : maximum k value the embedding table covers
    dropout     : dropout rate in the Transformer layers
    """

    def __init__(
        self,
        k_emb_dim: int = 16,
        sage_hidden: int = 64,
        sage_layers: int = 2,
        tf_hidden: int = 64,
        tf_heads: int = 4,
        tf_layers: int = 2,
        max_k: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()

        d_in = 1 + k_emb_dim  # scalar degree + k vector

        # k conditioning
        self.k_embedding = nn.Embedding(max_k + 1, k_emb_dim)

        # Project raw input to sage_hidden before SAGE layers
        self.input_proj = nn.Sequential(
            nn.Linear(d_in, sage_hidden),
            nn.ReLU(),
        )

        # GraphSAGE layers
        self.sage = nn.ModuleList(
            SAGEConv(sage_hidden, sage_hidden) for _ in range(sage_layers)
        )

        # Optional dimension adjustment between SAGE and Transformer
        self.sage_to_tf = (
            nn.Linear(sage_hidden, tf_hidden)
            if sage_hidden != tf_hidden
            else nn.Identity()
        )

        # Temporal Transformer encoder
        self.pos_enc = SinusoidalPE(tf_hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tf_hidden,
            nhead=tf_heads,
            dim_feedforward=tf_hidden * 4,
            dropout=dropout,
            batch_first=True,   # expects (batch, seq, d)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)

        # Output MLP: per-(v,t) → scalar probability
        self.head = nn.Sequential(
            nn.Linear(tf_hidden, tf_hidden // 2),
            nn.ReLU(),
            nn.Linear(tf_hidden // 2, 1),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        node_features: torch.Tensor,  # (n_nodes, T)
        edges: torch.Tensor,          # (E, 3): u_idx, v_idx, t_idx
        k: int,
        n_nodes: int,
        T: int,
    ) -> torch.Tensor:               # (n_nodes, T)

        device = node_features.device

        # ── k embedding ──────────────────────────────────────────────
        k_vec = self.k_embedding(torch.tensor([k], dtype=torch.long, device=device))
        # Broadcast to every (node, time) virtual node: (n_nodes*T, k_emb_dim)
        k_vec_exp = k_vec.expand(n_nodes * T, -1)

        # ── Per-(v,t) input features ──────────────────────────────────
        # node_features layout after reshape: index = v * T + t  (row-major)
        degree_flat = node_features.reshape(n_nodes * T, 1)          # (N*T, 1)
        x = torch.cat([degree_flat, k_vec_exp], dim=1)               # (N*T, d_in)
        x = self.input_proj(x)                                        # (N*T, sage_hidden)

        # ── Batched GraphSAGE across all snapshots ────────────────────
        # Global index for virtual node (v, t): v * T + t
        if edges.numel() > 0:
            u_g = edges[:, 0] * T + edges[:, 2]   # shape (E,)
            v_g = edges[:, 1] * T + edges[:, 2]
            # Both directions for undirected message passing
            edge_index = torch.stack(
                [torch.cat([u_g, v_g]), torch.cat([v_g, u_g])], dim=0
            )                                                         # (2, 2E)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

        for conv in self.sage:
            x = F.relu(conv(x, edge_index))                          # (N*T, sage_hidden)

        x = self.sage_to_tf(x)                                       # (N*T, tf_hidden)

        # ── Temporal Transformer (per-node sequence over T) ───────────
        x = x.reshape(n_nodes, T, -1)   # (n_nodes, T, tf_hidden)
        x = self.pos_enc(x)             # add sinusoidal PE along T
        x = self.transformer(x)         # (n_nodes, T, tf_hidden)

        # ── Output head ───────────────────────────────────────────────
        p = self.head(x).squeeze(-1)    # (n_nodes, T)
        return torch.sigmoid(p)
