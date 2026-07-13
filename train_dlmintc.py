"""
Train the DLMinTC+ reimplementation (dlmintc_plus.py) and select the checkpoint
by post-processed validation sum-span — the same fair protocol used for our model
(train.py), so the head-to-head compares the two decoding paradigms on equal terms.

Usage:
  python train_dlmintc.py --train data/train_k1 --val data/val_k1 --epochs 60 \
      --checkpoint-dir checkpoints_dlmintc
"""

import argparse
import os
import time

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_pipeline import TemporalCoverDataset
from dlmintc_plus import DLMinTCPlus, dlmintc_loss, iterative_adjustment
from postprocess import sum_span, coverage_fraction


def _empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


@torch.no_grad()
def evaluate(model, ds, device):
    model.eval()
    tot_span = 0.0
    tot_cov = 0.0
    n = 0
    for inst in ds:
        if 'unique_times' not in inst:
            continue
        nf = inst['node_features'].to(device)
        edg = inst['edges'].to(device)
        ut = inst['unique_times']
        times = torch.tensor(ut, dtype=torch.float, device=device)
        s, e, _, _ = model(nf, edg, inst['n_nodes'], inst['T'], times)
        iv = iterative_adjustment(s.cpu(), e.cpu(), edg.cpu(), ut)
        tot_span += sum_span(iv)
        tot_cov += coverage_fraction(iv, edg.cpu(), ut)
        n += 1
    return tot_span / max(n, 1), tot_cov / max(n, 1)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}')
    train_ds = TemporalCoverDataset(args.train)
    val_ds = TemporalCoverDataset(args.val)
    print(f'Train {len(train_ds)}  Val {len(val_ds)}')

    model = DLMinTCPlus().to(device)
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
    opt = Adam(model.parameters(), lr=args.lr)
    sched = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_span = float('inf')
    no_improve = 0
    idx = list(range(len(train_ds)))
    import random

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(idx)
        opt.zero_grad()
        t0 = time.time()
        ep_loss = 0.0
        for step, i in enumerate(idx):
            inst = train_ds[i]
            nf = inst['node_features'].to(device)
            edg = inst['edges'].to(device)
            mask = inst['mask'].to(device)
            ut = inst['unique_times']
            times = torch.tensor(ut, dtype=torch.float, device=device)
            s, e, a_s, a_e = model(nf, edg, inst['n_nodes'], inst['T'], times)
            loss, _ = dlmintc_loss(s, e, a_s, a_e, edg, mask, times)
            (loss / args.accum).backward()
            ep_loss += float(loss.detach())
            if (step + 1) % args.accum == 0 or (step + 1) == len(idx):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(); _empty_cache()

        val_span, val_cov = evaluate(model, val_ds, device)
        sched.step(val_span)
        print(f'Epoch {epoch:3d}/{args.epochs}  loss={ep_loss/len(idx):.4f}  '
              f'val_span={val_span:.1f}  val_cov={val_cov:.4f}  '
              f'lr={opt.param_groups[0]["lr"]:.1e}  ({time.time()-t0:.0f}s)')

        if val_span < best_span:
            best_span = val_span
            no_improve = 0
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_span': val_span, 'val_cov': val_cov},
                       os.path.join(args.checkpoint_dir, 'best_model.pt'))
            print(f'  -> saved best (val_span={best_span:.1f})')
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f'Early stop at epoch {epoch}')
                break

    print(f'Done. Best val_span={best_span:.1f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='data/train_k1')
    ap.add_argument('--val', default='data/val_k1')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--checkpoint-dir', default='checkpoints_dlmintc')
    main(ap.parse_args())
