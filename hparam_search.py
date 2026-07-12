"""
Hyperparameter grid search over loss weights for DLMinTCk.

Searches w_imit × w_span × w_trans over a 3×3×3 grid (27 combinations).
Each combination trains for `--epochs` (default 15) on the full training set
and is scored by validation coverage_rate (primary) and val_trans_viol (secondary).

Results are printed as a ranked table at the end.

CLI usage
---------
  python hparam_search.py --train data/train --val data/val --epochs 15
"""

import argparse
import copy
import itertools
import os
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_pipeline import TemporalCoverDataset
from model import DLMinTCk
from train import compute_loss, evaluate, _empty_cache


# ---------------------------------------------------------------------------
# Lightweight training run (no checkpointing, returns val metrics)
# ---------------------------------------------------------------------------

def quick_train(
    train_ds: TemporalCoverDataset,
    val_ds: TemporalCoverDataset,
    device: torch.device,
    epochs: int,
    w_imit: float,
    w_cov: float,
    w_span: float,
    w_trans: float,
    w_trans_start: float,
    trans_anneal_epochs: int,
    trans_margin: float,
    # fixed arch
    k_emb_dim: int = 16,
    sage_hidden: int = 64,
    sage_layers: int = 2,
    tf_hidden: int = 64,
    tf_heads: int = 4,
    tf_layers: int = 2,
    lr: float = 1e-3,
    accum_steps: int = 8,
) -> Dict[str, float]:
    """Train a fresh model and return the best val metrics seen across all epochs."""
    import random

    model = DLMinTCk(
        k_emb_dim=k_emb_dim,
        sage_hidden=sage_hidden,
        sage_layers=sage_layers,
        tf_hidden=tf_hidden,
        tf_heads=tf_heads,
        tf_layers=tf_layers,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    indices = list(range(len(train_ds)))

    best_metrics: Dict[str, float] = {'coverage_rate': -1.0, 'trans_viol': 1.0, 'loss': 1e9}

    for epoch in range(1, epochs + 1):
        if trans_anneal_epochs > 0 and epoch <= trans_anneal_epochs:
            eff_w_trans = w_trans_start + (epoch - 1) / trans_anneal_epochs * (w_trans - w_trans_start)
        else:
            eff_w_trans = w_trans

        model.train()
        random.shuffle(indices)
        optimizer.zero_grad()

        for step, idx in enumerate(indices):
            inst = train_ds[idx]
            nf   = inst['node_features'].to(device)
            edg  = inst['edges'].to(device)
            mask = inst['mask'].to(device)
            k    = int(inst['k'])

            p = model(nf, edg, k, inst['n_nodes'], inst['T'])
            loss, _ = compute_loss(
                p, mask, edg, k,
                w_imit, w_cov, w_span, eff_w_trans, trans_margin,
            )
            (loss / accum_steps).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(indices):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                _empty_cache()

        val_metrics = evaluate(model, val_ds, device)
        if val_metrics['coverage_rate'] > best_metrics['coverage_rate']:
            best_metrics = dict(val_metrics)

    return best_metrics


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f'Device: {device}')

    train_ds = TemporalCoverDataset(args.train)
    val_ds   = TemporalCoverDataset(args.val)

    if args.max_train and args.max_train < len(train_ds):
        import random as _rng
        subset_indices = _rng.sample(range(len(train_ds)), args.max_train)
        # Wrap as a lightweight subset view
        class _Subset:
            def __init__(self, ds, idx): self._ds, self._idx = ds, idx
            def __len__(self): return len(self._idx)
            def __iter__(self):
                for i in self._idx:
                    yield self._ds[i]
            def __getitem__(self, i): return self._ds[self._idx[i]]
        train_ds = _Subset(train_ds, subset_indices)
        print(f'Train: {len(train_ds)} (subset)  Val: {len(val_ds)}')
    else:
        print(f'Train: {len(train_ds)}  Val: {len(val_ds)}')
    print(f'Epochs per combination: {args.epochs}')

    # --- Grid definition ---
    w_imit_grid  = [0.3, 0.5, 1.0]
    w_span_grid  = [0.1, 0.3, 0.5]
    w_trans_grid = [1.0, 2.0, 5.0]
    w_cov        = 1.0   # fixed — coverage is a hard requirement, keep weight high

    grid = list(itertools.product(w_imit_grid, w_span_grid, w_trans_grid))
    total = len(grid)
    print(f'Grid size: {total} combinations  (w_imit × w_span × w_trans, w_cov fixed at {w_cov})')
    print()

    results: List[Tuple] = []

    for i, (w_imit, w_span, w_trans) in enumerate(grid, 1):
        t0 = time.time()
        print(f'[{i:2d}/{total}] w_imit={w_imit}  w_span={w_span}  w_trans={w_trans} ...', end='', flush=True)

        metrics = quick_train(
            train_ds, val_ds, device,
            epochs=args.epochs,
            w_imit=w_imit,
            w_cov=w_cov,
            w_span=w_span,
            w_trans=w_trans,
            w_trans_start=args.w_trans_start,
            trans_anneal_epochs=args.trans_anneal_epochs,
            trans_margin=args.trans_margin,
            lr=args.lr,
            accum_steps=args.accum_steps,
        )

        elapsed = time.time() - t0
        cov   = metrics['coverage_rate']
        viol  = metrics['trans_viol']
        loss  = metrics['loss']
        print(f'  cov={cov:.4f}  trans_viol={viol:.4f}  loss={loss:.4f}  ({elapsed:.0f}s)')

        results.append((w_imit, w_span, w_trans, cov, viol, loss))

    # Sort by coverage (desc), then by trans_viol (asc), then by loss (asc)
    results.sort(key=lambda r: (-r[3], r[4], r[5]))

    print('\n' + '=' * 80)
    print('RANKED RESULTS  (sorted by coverage ↓, trans_viol ↑, loss ↑)')
    print('=' * 80)
    print(f"{'rank':>4}  {'w_imit':>7}  {'w_span':>7}  {'w_trans':>8}  {'val_cov':>8}  {'trans_viol':>10}  {'val_loss':>9}")
    print('-' * 80)
    for rank, (w_imit, w_span, w_trans, cov, viol, loss) in enumerate(results, 1):
        marker = '  ←' if rank == 1 else ''
        print(f'{rank:>4}  {w_imit:>7.1f}  {w_span:>7.1f}  {w_trans:>8.1f}  {cov:>8.4f}  {viol:>10.4f}  {loss:>9.4f}{marker}')

    best = results[0]
    print(f'\nBest combination: w_imit={best[0]}  w_span={best[1]}  w_trans={best[2]}')
    print(f'Retrain with:')
    print(f'  python train.py --train {args.train} --val {args.val} --epochs 100 \\')
    print(f'    --w-imit {best[0]} --w-span {best[1]} --w-trans {best[2]} \\')
    print(f'    --w-trans-start {args.w_trans_start} --trans-anneal-epochs {args.trans_anneal_epochs} --trans-margin {args.trans_margin}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grid search over DLMinTCk loss weights')
    parser.add_argument('--train',     type=str, default='data/train')
    parser.add_argument('--val',       type=str, default='data/val')
    parser.add_argument('--epochs',    type=int, default=10,
                        help='Training epochs per combination')
    parser.add_argument('--max-train', type=int, default=300,
                        help='Random subset of training instances to use (0 = use all)')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--accum-steps',  type=int,   default=8)
    # Transition settings (fixed across grid — only weights are searched)
    parser.add_argument('--w-trans-start',       type=float, default=0.1)
    parser.add_argument('--trans-anneal-epochs', type=int,   default=10)
    parser.add_argument('--trans-margin',        type=float, default=1.0)
    args = parser.parse_args()
    main(args)
