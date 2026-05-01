"""
Layer 3: Loss functions and training loop for DLMinTCk.

Loss terms (all averaged over the instance)
-------------------------------------------
1. Imitation  : BCE against pseudo-ground-truth binary mask.
2. Coverage   : for every edge (u,v,t), penalise -log(p_u^t + p_v^t - p_u^t*p_v^t).
3. Sparsity   : mean(p) — encourages tight masks.
4. Transition : ReLU(Σ_t |p_{v,t+1} - p_{v,t}| - 2k) per node, averaged.
                Encodes the ≤2k transitions constraint from the ILP.

Combined: L = w_imit*L_imit + w_cov*L_cov + w_span*L_span + w_trans*L_trans

Training details
----------------
- One instance per forward pass (variable n_nodes, T).
- Gradient accumulation over `accum_steps` instances before optimizer.step().
- ReduceLROnPlateau scheduler on validation coverage rate.
- Best checkpoint by highest validation coverage rate.

CLI usage
---------
  python train.py --train data/train --val data/val --epochs 50
"""

import argparse
import os
import time
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_pipeline import TemporalCoverDataset
from model import DLMinTCk


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

EPS = 1e-7


def coverage_loss(p: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """
    For each edge (u,v,t): -log(p_u^t + p_v^t - p_u^t * p_v^t + eps).
    This is 0 when either endpoint is certainly active and large when both are 0.
    """
    if edges.numel() == 0:
        return torch.tensor(0.0, device=p.device)
    u, v, t = edges[:, 0], edges[:, 1], edges[:, 2]
    pu, pv = p[u, t], p[v, t]
    # at_least_one = 1 - (1-pu)*(1-pv) = pu + pv - pu*pv
    at_least_one = pu + pv - pu * pv
    return -torch.log(at_least_one.clamp(min=EPS)).mean()


def sparsity_loss(p: torch.Tensor) -> torch.Tensor:
    """Mean activation — encourages the mask to be as tight as possible."""
    return p.mean()


def transition_loss(p: torch.Tensor, k: int) -> torch.Tensor:
    """
    ReLU(Σ_t |p_{v,t+1} - p_{v,t}| - 2k) per node, averaged over nodes.
    Relaxed version of the ILP constraint Σ_t y_v^t ≤ 2k.
    """
    if p.size(1) < 2:
        return torch.tensor(0.0, device=p.device)
    transitions = torch.abs(p[:, 1:] - p[:, :-1]).sum(dim=1)   # (n_nodes,)
    return F.relu(transitions - 2.0 * k).mean()


def imitation_loss(p: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy against the pseudo-ground-truth mask."""
    return F.binary_cross_entropy(p, mask)


def compute_loss(
    p: torch.Tensor,
    mask: torch.Tensor,
    edges: torch.Tensor,
    k: int,
    w_imit: float = 1.0,
    w_cov: float = 1.0,
    w_span: float = 0.1,
    w_trans: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Weighted sum of all four loss terms.
    Returns (total_loss, dict_of_individual_values).
    """
    L_imit  = imitation_loss(p, mask)
    L_cov   = coverage_loss(p, edges)
    L_span  = sparsity_loss(p)
    L_trans = transition_loss(p, k)

    total = w_imit * L_imit + w_cov * L_cov + w_span * L_span + w_trans * L_trans
    return total, {
        'imit':  L_imit.item(),
        'cov':   L_cov.item(),
        'span':  L_span.item(),
        'trans': L_trans.item(),
    }


# ---------------------------------------------------------------------------
# Evaluation metrics (proxy, pre-post-processing)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: DLMinTCk, dataset: TemporalCoverDataset, device: torch.device) -> Dict[str, float]:
    """
    Evaluate on a dataset. Returns proxy metrics (no DP post-processing yet):
      coverage_rate : fraction of edges with max(p_u^t, p_v^t) > 0.5
      bce_acc       : fraction of (v,t) where round(p_v^t) == mask_v^t
      trans_viol    : fraction of nodes with > 2k transitions in binarized mask
      loss          : average total loss
    """
    model.eval()
    totals: Dict[str, float] = {
        'coverage_rate': 0.0, 'bce_acc': 0.0, 'trans_viol': 0.0, 'loss': 0.0
    }
    n = len(dataset)
    if n == 0:
        return totals

    for inst in dataset:
        nf   = inst['node_features'].to(device)
        edg  = inst['edges'].to(device)
        mask = inst['mask'].to(device)
        k    = int(inst['k'])

        p = model(nf, edg, k, inst['n_nodes'], inst['T'])

        loss, _ = compute_loss(p, mask, edg, k)
        totals['loss'] += loss.item()

        pred = (p > 0.5).float()

        # Coverage
        if edg.numel() > 0:
            u, v, t = edg[:, 0], edg[:, 1], edg[:, 2]
            covered = (pred[u, t] + pred[v, t]) >= 1
            totals['coverage_rate'] += covered.float().mean().item()
        else:
            totals['coverage_rate'] += 1.0

        # BCE accuracy
        totals['bce_acc'] += (pred == mask).float().mean().item()

        # Transition violations
        if pred.size(1) >= 2:
            trans = torch.abs(pred[:, 1:] - pred[:, :-1]).sum(dim=1)
            violated = (trans > 2 * k).float().mean().item()
        else:
            violated = 0.0
        totals['trans_viol'] += violated

    return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: DLMinTCk,
    train_ds: TemporalCoverDataset,
    val_ds: TemporalCoverDataset,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    accum_steps: int = 8,
    w_imit: float = 1.0,
    w_cov: float = 1.0,
    w_span: float = 0.1,
    w_trans: float = 0.5,
    checkpoint_dir: str = 'checkpoints',
) -> None:

    os.makedirs(checkpoint_dir, exist_ok=True)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_coverage = -1.0
    indices = list(range(len(train_ds)))

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        term_totals: Dict[str, float] = {'imit': 0.0, 'cov': 0.0, 'span': 0.0, 'trans': 0.0}
        t0 = time.time()

        # Shuffle training order each epoch
        import random
        random.shuffle(indices)

        optimizer.zero_grad()
        for step, idx in enumerate(indices):
            inst = train_ds[idx]
            nf   = inst['node_features'].to(device)
            edg  = inst['edges'].to(device)
            mask = inst['mask'].to(device)
            k    = int(inst['k'])

            p = model(nf, edg, k, inst['n_nodes'], inst['T'])
            loss, terms = compute_loss(p, mask, edg, k, w_imit, w_cov, w_span, w_trans)

            # Scale loss by accum_steps so gradients are averaged
            (loss / accum_steps).backward()

            epoch_loss += loss.item()
            for key in term_totals:
                term_totals[key] += terms[key]

            if (step + 1) % accum_steps == 0 or (step + 1) == len(indices):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        n_train = len(indices)
        avg_loss = epoch_loss / n_train

        # Validation
        val_metrics = evaluate(model, val_ds, device)
        scheduler.step(val_metrics['coverage_rate'])

        elapsed = time.time() - t0
        print(
            f'Epoch {epoch:3d}/{epochs}  '
            f'loss={avg_loss:.4f}  '
            f'val_cov={val_metrics["coverage_rate"]:.4f}  '
            f'val_acc={val_metrics["bce_acc"]:.4f}  '
            f'val_trans_viol={val_metrics["trans_viol"]:.4f}  '
            f'lr={optimizer.param_groups[0]["lr"]:.2e}  '
            f'({elapsed:.1f}s)'
        )

        # Checkpoint best model
        if val_metrics['coverage_rate'] > best_coverage:
            best_coverage = val_metrics['coverage_rate']
            path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_metrics': val_metrics,
                'train_loss': avg_loss,
                'term_totals': {k: v / n_train for k, v in term_totals.items()},
            }, path)
            print(f'  → saved best model (coverage={best_coverage:.4f})')

    print(f'\nTraining complete. Best val coverage rate: {best_coverage:.4f}')
    print(f'Best model saved to: {os.path.join(checkpoint_dir, "best_model.pt")}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DLMinTCk model')
    parser.add_argument('--train', type=str, default='data/train', help='Training data directory')
    parser.add_argument('--val',   type=str, default='data/val',   help='Validation data directory')
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--accum-steps', type=int,   default=8,
                        help='Gradient accumulation steps (effective batch size)')
    parser.add_argument('--k-emb-dim',   type=int,   default=16)
    parser.add_argument('--sage-hidden', type=int,   default=64)
    parser.add_argument('--sage-layers', type=int,   default=2)
    parser.add_argument('--tf-hidden',   type=int,   default=64)
    parser.add_argument('--tf-heads',    type=int,   default=4)
    parser.add_argument('--tf-layers',   type=int,   default=2)
    parser.add_argument('--dropout',     type=float, default=0.1)
    parser.add_argument('--w-imit',      type=float, default=1.0)
    parser.add_argument('--w-cov',       type=float, default=1.0)
    parser.add_argument('--w-span',      type=float, default=0.1)
    parser.add_argument('--w-trans',     type=float, default=0.5)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--device',      type=str,   default='auto',
                        help='auto | cpu | cuda | mps')
    args = parser.parse_args()

    # Device selection
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f'Using device: {device}')

    train_ds = TemporalCoverDataset(args.train)
    val_ds   = TemporalCoverDataset(args.val)
    print(f'Train: {len(train_ds)} instances   Val: {len(val_ds)} instances')

    model = DLMinTCk(
        k_emb_dim=args.k_emb_dim,
        sage_hidden=args.sage_hidden,
        sage_layers=args.sage_layers,
        tf_hidden=args.tf_hidden,
        tf_heads=args.tf_heads,
        tf_layers=args.tf_layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {n_params:,}')

    train(
        model, train_ds, val_ds, device,
        epochs=args.epochs,
        lr=args.lr,
        accum_steps=args.accum_steps,
        w_imit=args.w_imit,
        w_cov=args.w_cov,
        w_span=args.w_span,
        w_trans=args.w_trans,
        checkpoint_dir=args.checkpoint_dir,
    )
