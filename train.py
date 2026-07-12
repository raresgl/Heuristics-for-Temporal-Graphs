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


def _empty_cache():
    """Release unused memory on whichever accelerator is active."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

from data_pipeline import TemporalCoverDataset
from model import DLMinTCk
from postprocess import postprocess, sum_span, coverage_fraction


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


def transition_loss(p: torch.Tensor, k: int, margin: float = 0.0) -> torch.Tensor:
    """
    Squared hinge: ReLU(Σ_t |p_{v,t+1} - p_{v,t}| - 2k + margin)² per node.

    margin > 0 penalises masks that are 'barely' within budget, pushing the
    model toward distributions with slack.  A plain ReLU (margin=0) has zero
    gradient whenever the budget is not violated, giving no incentive to reduce
    transitions further.  With margin=1 the model is penalised as soon as
    transitions exceed 2k-1, producing a tighter learned constraint.
    """
    if p.size(1) < 2:
        return torch.tensor(0.0, device=p.device)
    transitions = torch.abs(p[:, 1:] - p[:, :-1]).sum(dim=1)   # (n_nodes,)
    return F.relu(transitions - 2.0 * k + margin).pow(2).mean()


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
    trans_margin: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Weighted sum of all four loss terms.
    Returns (total_loss, dict_of_individual_values).
    """
    L_imit  = imitation_loss(p, mask)
    L_cov   = coverage_loss(p, edges)
    L_span  = sparsity_loss(p)
    L_trans = transition_loss(p, k, margin=trans_margin)

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
def evaluate(
    model: DLMinTCk,
    dataset: TemporalCoverDataset,
    device: torch.device,
    assign_mode: str = 'model',
    beta: float = 1.0,
) -> Dict[str, float]:
    """
    Evaluate on a dataset. Returns:
      coverage_rate : fraction of edges with max(p_u^t, p_v^t) > 0.5 (proxy, pre-DP)
      bce_acc       : fraction of (v,t) where round(p_v^t) == mask_v^t
      trans_viol    : fraction of nodes with > 2k transitions in binarized mask
      loss          : average total loss
      pp_span       : average total span of the post-processed k-interval cover
                      (THE downstream metric — what we actually optimise for)
      pp_coverage   : average edge coverage after post-processing (≈1.0 always)

    pp_span is computed with the given assign_mode/beta so that checkpoint
    selection matches how the model will be used at evaluation time.
    """
    model.eval()
    totals: Dict[str, float] = {
        'coverage_rate': 0.0, 'bce_acc': 0.0, 'trans_viol': 0.0, 'loss': 0.0,
        'pp_span': 0.0, 'pp_coverage': 0.0,
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

        # Coverage (proxy, pre-DP)
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

        # Post-processed span — the real objective
        if 'unique_times' in inst:
            ut = inst['unique_times']
            intervals = postprocess(p.cpu(), edg.cpu(), k, ut, mode=assign_mode, beta=beta)
            totals['pp_span'] += sum_span(intervals)
            totals['pp_coverage'] += coverage_fraction(intervals, edg.cpu(), ut)
        else:
            totals['pp_coverage'] += 1.0

    return {key: v / n for key, v in totals.items()}


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
    w_trans: float = 2.0,
    w_trans_start: float = 0.1,
    trans_anneal_epochs: int = 10,
    trans_margin: float = 1.0,
    early_stop_patience: int = 25,
    checkpoint_dir: str = 'checkpoints',
    assign_mode: str = 'model',
    beta: float = 1.0,
) -> None:
    """
    w_trans_start → w_trans annealing:  for the first trans_anneal_epochs
    epochs the transition weight ramps linearly from w_trans_start to w_trans.
    This lets the network first learn coverage before the hard k-interval
    constraint is enforced at full strength.

    Checkpointing and the LR scheduler are driven by the post-processed
    validation span (pp_span) — the actual downstream objective — using the
    given assign_mode/beta, rather than by the surrogate training loss.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    optimizer = Adam(model.parameters(), lr=lr)
    # Monitor downstream span (mode='min'). patience=15 prevents premature LR collapse.
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    best_span = float('inf')
    epochs_no_improve = 0
    indices = list(range(len(train_ds)))

    import random

    for epoch in range(1, epochs + 1):
        # --- anneal transition weight ---
        if trans_anneal_epochs > 0 and epoch <= trans_anneal_epochs:
            frac = (epoch - 1) / trans_anneal_epochs
            eff_w_trans = w_trans_start + frac * (w_trans - w_trans_start)
        else:
            eff_w_trans = w_trans

        model.train()
        epoch_loss = 0.0
        term_totals: Dict[str, float] = {'imit': 0.0, 'cov': 0.0, 'span': 0.0, 'trans': 0.0}
        t0 = time.time()

        random.shuffle(indices)

        optimizer.zero_grad()
        for step, idx in enumerate(indices):
            inst = train_ds[idx]
            nf   = inst['node_features'].to(device)
            edg  = inst['edges'].to(device)
            mask = inst['mask'].to(device)
            k    = int(inst['k'])

            p = model(nf, edg, k, inst['n_nodes'], inst['T'])
            loss, terms = compute_loss(
                p, mask, edg, k,
                w_imit, w_cov, w_span, eff_w_trans, trans_margin,
            )

            (loss / accum_steps).backward()

            epoch_loss += loss.item()
            for key in term_totals:
                term_totals[key] += terms[key]

            if (step + 1) % accum_steps == 0 or (step + 1) == len(indices):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                _empty_cache()

        n_train = len(indices)
        avg_loss = epoch_loss / n_train
        avg_terms = {k: v / n_train for k, v in term_totals.items()}

        # Validation
        val_metrics = evaluate(model, val_ds, device, assign_mode, beta)
        scheduler.step(val_metrics['pp_span'])

        elapsed = time.time() - t0
        print(
            f'Epoch {epoch:3d}/{epochs}  '
            f'loss={avg_loss:.4f}  '
            f'[imit={avg_terms["imit"]:.3f} '
            f'cov={avg_terms["cov"]:.3f} '
            f'trans={avg_terms["trans"]:.3f}]  '
            f'val_span={val_metrics["pp_span"]:.1f}  '
            f'val_pp_cov={val_metrics["pp_coverage"]:.4f}  '
            f'val_loss={val_metrics["loss"]:.4f}  '
            f'val_cov={val_metrics["coverage_rate"]:.4f}  '
            f'w_trans={eff_w_trans:.2f}  '
            f'lr={optimizer.param_groups[0]["lr"]:.2e}  '
            f'({elapsed:.1f}s)'
        )

        if val_metrics['pp_span'] < best_span:
            best_span = val_metrics['pp_span']
            epochs_no_improve = 0
            path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_metrics': val_metrics,
                'train_loss': avg_loss,
                'term_totals': avg_terms,
                'assign_mode': assign_mode,
                'beta': beta,
            }, path)
            print(f'  → saved best model (val_span={best_span:.1f})')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f'\nEarly stopping at epoch {epoch} '
                      f'(no improvement for {early_stop_patience} epochs)')
                break

    print(f'\nTraining complete. Best val span: {best_span:.1f}')
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
    parser.add_argument('--w-imit',             type=float, default=1.0)
    parser.add_argument('--w-cov',              type=float, default=1.0)
    parser.add_argument('--w-span',             type=float, default=0.1)
    parser.add_argument('--w-trans',            type=float, default=2.0,
                        help='Final transition loss weight (after annealing)')
    parser.add_argument('--w-trans-start',      type=float, default=0.1,
                        help='Initial transition loss weight (start of annealing)')
    parser.add_argument('--trans-anneal-epochs', type=int,  default=10,
                        help='Epochs over which to ramp w-trans-start → w-trans (0 = no annealing)')
    parser.add_argument('--trans-margin',        type=float, default=1.0,
                        help='Margin in squared-hinge transition loss: penalise when transitions > 2k - margin')
    parser.add_argument('--early-stop-patience', type=int,   default=25,
                        help='Stop training if val span does not improve for this many epochs')
    parser.add_argument('--assign', type=str, default='model',
                        choices=['greedy', 'model', 'hybrid'],
                        help='Edge-assignment mode used to compute the validation span '
                             'that drives checkpointing (match this to evaluate.py --assign)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Model influence weight for --assign hybrid')
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
        w_trans_start=args.w_trans_start,
        trans_anneal_epochs=args.trans_anneal_epochs,
        trans_margin=args.trans_margin,
        early_stop_patience=args.early_stop_patience,
        checkpoint_dir=args.checkpoint_dir,
        assign_mode=args.assign,
        beta=args.beta,
    )
