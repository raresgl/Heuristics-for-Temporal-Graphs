# DLMinTCk — Deep Learning for k-Interval Minimum Temporal Vertex Cover

This repository implements **DLMinTCk**, a graph neural network approach to the
**k-Interval Minimum Temporal Vertex Cover** problem, where each node may be
active during at most k disjoint time intervals and the goal is to minimise the
total active time while ensuring that every temporal edge is covered by at least
one active endpoint.

> **Headline result (held-out test set, n=200, k∈[2,10], all ILP instances
> proven optimal):** ML+DP reaches **1.185× the optimal span** on average
> (within 18.5% of the ILP optimum) and **beats the strongest heuristic,
> k-Inner, by 60.7%** (601.0 vs 1528.8 average span), winning at **every** value
> of k. The full methodology, per-k tables, ablations, and discussion are in
> [`EXPERIMENTS.md`](EXPERIMENTS.md).

---

## Problem Definition

A **temporal graph** is a sequence of timestamped interactions (u, v, t) between
nodes. A **k-interval temporal vertex cover** assigns to each node v a set of at
most k disjoint time intervals such that for every edge (u, v, t), at least one
of u or v is active at time t (i.e., t falls inside one of their intervals). The
**minimum** such cover minimises the total span: the sum of all interval lengths
across all nodes.

This extends the classical k=1 setting studied by Lazzarinetti et al. (DLMinTC+)
to arbitrary k ≥ 1.

---

## Repository Structure

```
data_pipeline.py    — synthetic graph generation and ground-truth labelling
model.py            — DLMinTCk neural network (k-embedding + GraphSAGE + Transformer)
train.py            — loss functions and training loop (span-based checkpointing)
postprocess.py      — edge assignment (greedy/model/hybrid) + k-interval covering DP
evaluate.py         — single-mode evaluation vs ILP and heuristic baselines
eval_paper.py       — paper-grade evaluation: all assign modes + baselines + ILP, one pass
ilp_version.py      — Gurobi ILP formulation (exact solver / optimal lower bound)
k_inner.py          — k-Inner heuristic (Rozenshtein et al.)
k_budget.py         — k-Budget heuristic (Rozenshtein et al.)
baseline.py         — greedy baseline
graph.py            — graph and interval generation utilities
EXPERIMENTS.md      — full methodology and held-out results writeup
```

---

## Method overview

DLMinTCk is a **seed-and-refine** pipeline:

```
forward pass → p ∈ (0,1)^{n×T}  →  edge assignment  →  per-node DP  →  intervals
```

A single `k`-conditioned network (**119,921 parameters**) predicts a soft
per-`(node, time)` activity probability `p`; deterministic post-processing turns
`p` into a feasible k-interval cover. Because every edge is assigned to an
endpoint and the DP covers all assigned timestamps, **the output is always a
fully valid cover (coverage = 1.0)** — model quality affects span, never
feasibility.

### Network (`model.py`)

1. **k-embedding** — `nn.Embedding(max_k+1, 16)` conditions the network on k.
2. **Batched GraphSAGE** — all T snapshots in one supergraph pass; 2 layers, width 64.
3. **Temporal Transformer** — per-node sequence over T with sinusoidal positional
   encoding; 2 layers, 4 heads, `d_model = 64`. Attention is O(T²), so `max_T`
   caps instance size at data-generation time.
4. **Output head** — 2-layer MLP + sigmoid → `p_v^t ∈ (0,1)`.

### Loss (`train.py`)

| Term | Formula | Weight |
|------|---------|--------|
| Imitation | BCE(p, mask) | 1.0 |
| Coverage | −log(p\_u + p\_v − p\_u·p\_v + ε) per edge | 1.0 |
| Sparsity | mean(p) | 0.1 |
| Transition | ReLU(Σ\_t \|p\_{v,t+1}−p\_{v,t}\| − 2k + margin)² per node | 2.0 |

The transition weight anneals from 0.1 → 2.0 over the first 10 epochs (learn
coverage first, then enforce the ≤k-interval / ≤2k-transition budget). The
imitation target is the ILP-optimal mask where available, else best-of-heuristics.

### Post-processing (`postprocess.py`)

1. **Edge assignment** (`--assign`) — assign each edge to the endpoint that will
   cover it:
   - `greedy` — smallest marginal span-cost increase; model used only as tiebreak.
   - **`model`** *(default, best)* — `argmax(p_u^t, p_v^t)`; the network's learned
     decisions drive the cover.
   - `hybrid` — minimise `marginal_span_cost − β·log(p)`; span-cost primary,
     model can tip ties.
2. **Per-node DP** — an `O(n²k)` dynamic program covers each node's assigned
   timestamps with the minimum-span set of ≤k intervals (optimal given the
   assignment).

> **Why the assignment mode matters.** The per-node DP is identical across modes,
> so total span is decided entirely by *who covers each edge*. The original
> `greedy` decoder used the model only as a tiebreak — effectively discarding the
> network — which is why early "ML+DP" results merely matched the heuristics.
> Letting the model drive assignment (`model`) is what produces the headline
> result. See [`EXPERIMENTS.md`](EXPERIMENTS.md) §3.

---

## Data Pipeline

Synthetic temporal graphs are generated, timestamps floored and deduplicated,
nodes reindexed to 0-based, and labelled with the Gurobi ILP (small instances) or
the best of three heuristics. Instances with more than `max_T` unique timestamps
are rejected (Transformer O(T²) guard).

| Split | Instances | Labels | Purpose |
|-------|-----------|--------|---------|
| Train | 2000 | 918 ILP + 1082 heuristic | training |
| Val   | 500  | 226 ILP + 274 heuristic | checkpoint selection |
| **Test** | **200** | labels unused (recomputed at eval) | **held-out evaluation** |

The test split uses a distinct seed (7777), disjoint from train/val. Generation
parameters: k ∈ [2,10], n ∈ {20,30,50}, event_length ∈ {5,10}, overlap ∈
{0.0,0.3,0.5}, max_T = 500.

---

## Training

- Adam (lr = 1e-3), gradient accumulation 8, gradient clipping at norm 1.0
- `ReduceLROnPlateau` and early stopping driven by the **post-processed
  validation span (`pp_span`)** — the true downstream objective — computed with
  the `--assign` mode used at test time (not the surrogate loss)
- 100 epochs

**Best checkpoint: epoch 98, validation pp_span = 602.4, coverage 1.0.**
Validation span fell from 2190 (epoch 1) to 602 (epoch 98) — a 3.6× reduction
with a long, flat convergence tail.

---

## Evaluation Results (held-out test set, n = 200)

All methods achieve **perfect coverage (1.0)**; **span** is the differentiator.
All 200 ILP instances were proven optimal within the 30s budget, so ratios are
against the true optimum.

| Method | Avg span | Mean ratio vs OPT | Median ratio |
|--------|----------|-------------------|--------------|
| ILP (optimal) | 506.4 | 1.000× | 1.000× |
| **ML+DP (model)** | **601.0** | **1.185×** | **1.171×** |
| ML+DP (hybrid) | 1110.1 | 2.181× | 1.727× |
| k-Inner | 1528.8 | 3.023× | 2.792× |
| ML+DP (greedy) | 1599.1 | 3.256× | 2.784× |
| Baseline | 2692.5 | 5.368× | 4.953× |
| k-Budget | 5273.8 | 10.543× | 9.155× |

### Per-k (average span)

| k | n | ILP | ML+DP(model) | k-Inner | ML+DP(greedy) | Baseline | k-Budget |
|---|---|-----|--------------|---------|---------------|----------|----------|
| 2 | 59 | 385.0 | **438.5** | 1384.2 | 2011.9 | 2883.1 | 5330.2 |
| 3 | 37 | 451.5 | **539.0** | 1448.6 | 1692.7 | 2597.8 | 5263.0 |
| 4 | 29 | 541.2 | **625.9** | 1619.9 | 1482.8 | 2871.4 | 5705.6 |
| 5 | 27 | 575.7 | **698.1** | 1683.9 | 1411.8 | 2648.9 | 5385.7 |
| 6 | 19 | 521.3 | **655.9** | 1521.5 | 1135.5 | 2293.9 | 4586.7 |
| 7 | 11 | 613.9 | **722.7** | 1462.1 | 1129.8 | 2300.5 | 4565.6 |
| 8 | 7  | 802.4 | **954.3** | 1896.6 | 1363.6 | 2914.4 | 5857.3 |
| 9 | 9  | 769.8 | **956.7** | 1877.0 | 1319.8 | 2714.6 | 5423.6 |
| 10| 2  | 707.0 | **845.5** | 1447.5 | 972.0  | 1881.0 | 3744.5 |

`model` mode beats k-Inner (the best heuristic) at **every** k; the largest
margin is at low k (k=2: 438.5 vs 1384.2).

### Ablation: the gain is in the learned signal

Same network, same DP, same test set — only the assignment signal changes:

| Assignment signal | Avg span |
|-------------------|----------|
| Random `p` (model-mode) | 3711.3 |
| Greedy span-cost (old) | 1599.1 |
| **Learned `p` (model-mode)** | **601.0** |

Random probabilities are *worse* than greedy; learned probabilities are 2.7×
*better* — isolating the gain to the learned assignment, not the post-processing.

---

## Usage

### Generate data

```bash
# Train + val (ILP for small instances, heuristic fallback)
python data_pipeline.py --n 2000 --val-n 500 --ilp-budget 60 --max-t 500 --n-nodes 20,30,50 --verify

# Held-out test split (fresh seed; labels recomputed at eval time)
python data_pipeline.py --n 200 --out data/test --ilp-node-limit 0 --ilp-edge-limit 0 \
  --max-t 500 --n-nodes 20,30,50 --seed 7777
```

### Train

```bash
python train.py --train data/train --val data/val --epochs 100 --assign model
```

### Evaluate

```bash
# Paper-grade: all assign modes + heuristics + proven-optimal ILP, one pass
python eval_paper.py --test data/test --checkpoint checkpoints/best_model.pt \
  --ilp-budget 30 --out experiment_results/test_eval.json

# Single mode, quick (no ILP); compare decoders on the same checkpoint
python evaluate.py --val data/test --checkpoint checkpoints/best_model.pt --assign model  --ilp-budget 0
python evaluate.py --val data/test --checkpoint checkpoints/best_model.pt --assign greedy --ilp-budget 0
```

### Dependencies

```bash
python -m venv .venv
.venv/bin/pip install torch torch-geometric networkx numpy
# Gurobi licence required for ILP-sourced ground truth (data generation and evaluation)
```

---

## Limitations and Future Work

1. **Synthetic data only** — validation on real temporal networks is the key next step.
2. **Graph scale** — n ∈ {20,30,50}, T ≤ 500 (Transformer O(T²)); scaling up is open.
3. **Single-pass post-processing** — iterating assignment ↔ DP to a fixpoint could
   close more of the remaining 18.5% gap to optimal.
4. **`hybrid` β untuned** — pure `model` is best on this data; `β` would matter
   only if a span-aware tiebreak is needed for unreliable predictions.

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the complete discussion.
