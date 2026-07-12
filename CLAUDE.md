# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Generate training data
```bash
# Standard run (200 train, 50 val, ILP ground-truth where feasible)
python data_pipeline.py --n 200 --val-n 50 --ilp-budget 60 --max-t 1000 --verify

# Larger dataset
python data_pipeline.py --n 5000 --val-n 1000 --out data/train --val-out data/val \
  --ilp-budget 60 --ilp-node-limit 50 --ilp-edge-limit 500 --max-t 500 --n-nodes 20,30,50
```

### Generate held-out test split (fresh seed, labels recomputed at eval)
```bash
python data_pipeline.py --n 200 --out data/test --ilp-node-limit 0 --ilp-edge-limit 0 \
  --max-t 500 --n-nodes 20,30,50 --seed 7777
```

### Train
```bash
# --assign model is the key flag: validation span (the selection metric) is
# computed with this assignment mode, matching how the model is used at test time.
python train.py --train data/train --val data/val --epochs 100 --assign model

# With explicit hyperparameters
python train.py --train data/train --val data/val --epochs 100 --assign model \
  --w-imit 1.0 --w-cov 1.0 --w-span 0.1 --w-trans 2.0 \
  --w-trans-start 0.1 --trans-anneal-epochs 10 --trans-margin 1.0
```

### Evaluate
```bash
# Paper-grade: all assign modes + heuristics + proven-optimal ILP in one pass
python eval_paper.py --test data/test --checkpoint checkpoints/best_model.pt \
  --ilp-budget 30 --out experiment_results/test_eval.json

# Single mode, quick (no ILP). --assign: greedy | model | hybrid
python evaluate.py --val data/test --checkpoint checkpoints/best_model.pt --assign model --ilp-budget 0
```

### Hyperparameter search
```bash
python hparam_search.py --train data/train --val data/val --epochs 15 --max-train 300
```

### Dependencies
```bash
python -m venv .venv
.venv/bin/pip install torch torch-geometric networkx numpy
# Gurobi license required for ILP-sourced ground truth (data generation and evaluation)
```

## Architecture

The codebase is a five-layer pipeline extending DLMinTC+ (Lazzarinetti et al.) from k=1 to arbitrary k. The goal is k-Interval Minimum Temporal Vertex Cover: assign each node at most k disjoint activity intervals to cover every temporal edge at minimum total span.

### Layer 1 — `data_pipeline.py`
Generates labeled `.pt` instances saved to `data/train/` and `data/val/`. Each instance is a dict:
- `node_features`: `FloatTensor (n_nodes, T)` — per-(v,t) degree
- `edges`: `LongTensor (E, 3)` — `(u_idx, v_idx, t_idx)` all 0-based
- `mask`: `FloatTensor (n_nodes, T)` — binary ground-truth activity mask
- `k`, `n_nodes`, `T`, `unique_times`, `gt_source`, `seed`

Ground-truth: ILP (Gurobi) for small instances (≤50 nodes, ≤500 edges); best-of-heuristics (k-Inner → k-Budget → Baseline) otherwise.

### Layer 2 — `model.py`
`DLMinTCk`: single network parameterised by k ∈ [1, max_k].

Forward: `p = model(node_features, edges, k, n_nodes, T)` → `FloatTensor (n_nodes, T)` sigmoid probabilities.

Stages:
1. **k-embedding**: `nn.Embedding(max_k+1, k_emb_dim)` vector concatenated to every (v,t) input
2. **Batched GraphSAGE**: all T snapshots in one supergraph; virtual node index `v*T + t`
3. **Temporal Transformer**: reshape to `(n_nodes, T, tf_hidden)`, sinusoidal PE along T, TransformerEncoder
4. **MLP head**: `(tf_hidden → tf_hidden//2 → 1)` + sigmoid

### Layer 3 — `train.py`
Four loss terms combined as a weighted sum:
- **Imitation** (`w_imit=1.0`): BCE against pseudo-ground-truth mask
- **Coverage** (`w_cov=1.0`): `-log(p_u + p_v - p_u*p_v)` per edge
- **Sparsity** (`w_span=0.1`): `mean(p)`
- **Transition** (`w_trans=2.0`): squared hinge on `Σ|p_{v,t+1} - p_{v,t}| - 2k`

`w_trans` anneals from `w_trans_start=0.1` over `trans_anneal_epochs=10` epochs (lets the network learn coverage before the k-interval constraint is fully enforced).

**Checkpointing is driven by the post-processed validation span (`pp_span`), NOT `val_loss`.** `evaluate()` computes `pp_span` by running `postprocess` (with the `--assign` mode) on the val set; the best `pp_span` checkpoint is saved and the LR scheduler / early stopping (patience 25) key off it. This aligns the selection metric with the actual downstream objective.

### Layer 4 — `postprocess.py`
Converts soft mask → valid k-interval cover in two steps:
1. **Edge assignment** (`_assign_edges`, selected by `mode`): decides which endpoint covers each edge.
   - `greedy` — smallest marginal span cost; model used only as tiebreak (closed form `span(S,k) = (max-min) - sum of (k-1) largest gaps`).
   - **`model`** — `argmax(p_u^t, p_v^t)`; the network drives the cover. **This is what makes the model matter and produces the strong results** (≈1.19× optimal vs greedy's ≈3.26×). The greedy mode effectively discards the model.
   - `hybrid` — minimise `marginal_span_cost - beta*log(p)`.
2. **Per-node DP** (`find_minimal_intervals`): O(n²k) DP finds minimum-span set of ≤k intervals covering assigned timestamps (optimal given the assignment).

Coverage is always 1.0 regardless of model quality; only span varies.

Public API:
- `postprocess(p, edges, k, unique_times, mode='greedy', beta=1.0)` → `{node_idx: [(start, end), ...]}`
- `sum_span(active_intervals)` → int
- `coverage_fraction(active_intervals, edges, unique_times)` → float

### Layer 5 — `evaluate.py` / `eval_paper.py`
- `evaluate.py` — single `--assign` mode vs ILP + heuristics, grouped by k.
- `eval_paper.py` — paper-grade: one forward pass feeds all three assign modes; ILP (with proven-optimal tracking) and heuristics run once each; reports per-k span, coverage, and approximation ratios; dumps JSON. Use this for reproducible results.

## Heuristic interfaces

All three heuristics (`k_inner.py`, `k_budget.py`, `baseline.py`) share the same calling convention:
```python
Xstart, Xend = runKInner(timestamps, k)   # or runKBudget / kbaseline
```
where `timestamps` is a list of `(t, u, v)` integer tuples and the return value is a pair of nested dicts: `{node: {interval_idx: timestamp}}`. Convert to `{node: [(s, e), ...]}` via `data_pipeline.xstart_xend_to_active_intervals`.

## ILP interface (`ilp_version.py`)

```python
m, x = ilp_mod.ilp(timestamps, G, k)
m.setParam('TimeLimit', budget); m.setParam('OutputFlag', 0)
m.optimize()
if m.SolCount > 0:
    intervals = ilp_mod.active_intervals(m, x)  # {node: [(s, e), ...]}
```

ILP variables: `x[v,t] ∈ {0,1}` (activity), `y[v,t] ∈ {0,1}` (transitions). Constraint `Σ y[v,t] ≤ 2k` enforces ≤k intervals per node.

## Key data-flow invariants

- Node IDs and timestamp indices in `edges` are always 0-based (consecutive integers after `data_pipeline.reindex`).
- `unique_times` maps `t_idx → actual integer timestamp`; required for post-processing and evaluation.
- The Transformer's self-attention is O(T²), so `max_T` (default 1000 in generation, 500 recommended for GPU) must cap instance size.
- All `.pt` files must include `unique_times` to be usable by `evaluate.py`; older files without it are silently skipped.
