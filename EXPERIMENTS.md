# DLMinTCk: Methodology and Experimental Results

This document is the long-form, paper-oriented record of the method and the
held-out evaluation for the k-interval extension of DLMinTC+. It is written to
be adapted directly into the methodology, experiments, and discussion sections
of the paper. All numbers below come from a single held-out test set that the
model never saw during training or checkpoint selection.

---

## 1. Problem statement

A **temporal graph** is a set of timestamped interactions `(u, v, t)`. In the
**k-interval Minimum Temporal Vertex Cover** problem, each vertex `v` is assigned
at most `k` pairwise time-disjoint activity intervals. A temporal edge `(u, v, t)`
is *covered* if at least one of its endpoints is active at time `t` (i.e. `t`
falls inside one of that endpoint's intervals). The objective is to cover every
temporal edge while minimising the **total span** — the sum of all interval
lengths across all vertices.

The `k = 1` case is the setting of DLMinTC+ (Lazzarinetti et al.), where each
vertex receives a single contiguous interval `[s_v, e_v]`. This work extends the
learning-based approach to arbitrary `k ∈ [2, 10]` with a single network
conditioned on `k`.

### ILP reference formulation

The problem is stated exactly as a binary program (`ilp_version.py`):

- `x[v,t] ∈ {0,1}` — vertex `v` active at time `t`
- `y[v,t] ∈ {0,1}` — a transition (interval boundary) of `v` between `t` and `t+1`
- **Coverage:** `x[u,t] + x[v,t] ≥ 1` for every temporal edge `(u, v, t)`
- **Budget:** `Σ_t y[v,t] ≤ 2k` for every vertex (at most `k` intervals ⇔ at most
  `2k` boundaries)
- **Objective:** minimise `Σ_{v,t} x[v,t]`

This ILP provides the optimal lower bound used to compute approximation ratios.

---

## 2. Method

DLMinTCk is a **seed-and-refine** pipeline: a neural network produces a soft
per-`(vertex, time)` activity prediction, and a deterministic post-processing
stage converts that prediction into a feasible k-interval cover.

```
forward pass → p ∈ (0,1)^{n×T}  →  edge assignment  →  per-vertex DP  →  intervals
```

### 2.1 Network (`model.py`)

A single network of **119,921 parameters**, parameterised by `k`:

1. **k-embedding** — `nn.Embedding(max_k+1, 16)` maps the budget `k` to a vector
   concatenated to every `(vertex, time)` input feature, so one network serves
   all `k`.
2. **Batched GraphSAGE** — all `T` temporal snapshots are processed in a single
   "supergraph" pass (snapshot `t` occupies virtual nodes `[v·T + t]`), giving a
   SAGE cost of `O(E · layers)` independent of `T`. 2 layers, width 64.
3. **Temporal Transformer** — each vertex's `T` embeddings form a sequence;
   sinusoidal positional encoding is added along time and 2 encoder layers
   (4 heads, `d_model = 64`) model temporal dependencies per vertex.
4. **Output head** — a 2-layer MLP + sigmoid produces `p_v^t ∈ (0,1)`.

The only input node feature is the per-`(vertex, time)` degree.

### 2.2 Training loss (`train.py`)

A weighted sum of four differentiable terms:

| Term | Definition | Weight | Role |
|------|-----------|--------|------|
| Imitation | `BCE(p, mask)` against the pseudo-ground-truth mask | 1.0 | learn the optimal activity pattern |
| Coverage | `−log(p_u + p_v − p_u p_v)` per edge | 1.0 | every edge should have an active endpoint |
| Sparsity | `mean(p)` | 0.1 | keep masks tight |
| Transition | squared hinge on `Σ_t |p_{v,t+1} − p_{v,t}| − 2k` | 2.0 | soft ≤k-interval (≤2k-transition) constraint |

The transition weight is **annealed** from 0.1 to 2.0 over the first 10 epochs so
the network first learns coverage before the budget constraint is enforced at
full strength. The imitation target is the ILP-optimal mask where the ILP
terminates within budget and the best-of-heuristics mask otherwise.

### 2.3 Post-processing (`postprocess.py`)

Two steps convert the soft mask into a feasible cover:

1. **Edge assignment** — every temporal edge is assigned to exactly one endpoint
   ("who covers it"). This step is the subject of Section 3.
2. **Per-vertex DP** — for each vertex, an `O(n²k)` dynamic program computes the
   minimum-span set of at most `k` intervals that covers all timestamps assigned
   to it. This is **optimal given the assignment**, so the entire quality of the
   final cover is determined by how edges are assigned.

Because every edge is assigned to some endpoint and the DP covers all of a
vertex's assigned timestamps, **the pipeline always produces a fully feasible
cover (coverage = 1.0) regardless of the model's quality** — a poor model only
yields a larger span, never an invalid solution.

---

## 3. The central finding: the decoder must consume the learned structure

### 3.1 Diagnosis

The original pipeline assigned each edge to the endpoint with the smaller
**marginal span-cost increase**, using the model probability only as a tiebreak.
We found this *almost entirely discards the network's output*: the model
influenced the final cover only when two span costs were exactly equal, and the
DP ignored `p` outright. In effect, "ML+DP" was a span-greedy heuristic with a
nearly decorative neural network — which is why its span sat next to the other
heuristics rather than beating them.

This is a train/decode misalignment. The network is trained (imitation loss) to
reproduce the **ILP's global activity decisions** — i.e. which endpoint the
*optimal* solution uses to cover each edge — but the decoder then re-derived the
assignment locally and greedily, throwing that knowledge away.

### 3.2 The fix: assignment modes

We expose the edge-assignment strategy as a first-class choice (`--assign`):

- **`greedy`** — assign by smallest marginal span cost; model used only as
  tiebreak (the original behaviour).
- **`model`** — assign each edge to `argmax(p_u^t, p_v^t)`; the network's learned
  decisions drive the cover, while feasibility/optimal-span-per-vertex is still
  guaranteed by the DP.
- **`hybrid`** — assign by minimising `marginal_span_cost − β·log(p)`; span cost
  stays primary but a confident prediction can tip the decision.

### 3.3 Why `model` wins (mechanism)

The per-vertex DP is identical across modes, so the span gap is entirely an
*assignment-quality* effect. Minimum temporal vertex cover rewards
**concentrating** activity onto a few well-chosen "cover" vertices kept active
over the right windows, so their neighbours stay idle; cost is paid per
active vertex-time. The greedy rule is myopic and order-dependent — it evaluates
marginal cost against a partial assignment, and in the common regime where
adding an isolated timestamp costs 0 it falls back to an essentially arbitrary
tiebreak, **scattering** responsibility across many vertices and recreating
coverage instead of reusing it. The `model` rule instead routes each edge to the
endpoint the optimal solution would use (learned by imitation of ILP masks),
concentrating timestamps so the DP wraps them in a few tight intervals.

### 3.4 Train/eval alignment

To remove the second half of the misalignment, checkpoint selection and the LR
scheduler were switched from the surrogate validation **loss** to the actual
downstream **post-processed validation span** (`pp_span`), computed with the same
assignment mode used at test time. Selecting on the metric we ultimately care
about ensures the saved checkpoint is the one that produces the best covers, not
merely the lowest loss.

---

## 4. Experimental setup

### 4.1 Datasets

All instances are synthetic temporal graphs (power-law configuration-model
graphs with interval-structured activity; `graph.py`). `k` is sampled uniformly
from `[2, 10]`; graph size `n ∈ {20, 30, 50}`; event length `∈ {5, 10}`; overlap
`∈ {0.0, 0.3, 0.5}`; instances with more than 500 unique timestamps are rejected
(the Transformer's `O(T²)` attention).

| Split | Instances | Labels | Purpose |
|-------|-----------|--------|---------|
| train | 2000 | 918 ILP + 1082 heuristic | network training |
| val | 500 | 226 ILP + 274 heuristic | checkpoint selection |
| **test** | **200** | (labels unused) | **held-out final evaluation** |

The test set was generated with a distinct seed (7777), disjoint from train/val.
Test labels are not used: `eval_paper.py` recomputes the ILP optimum and all
heuristic baselines from scratch at evaluation time. All three splits share the
same generating distribution (`n`, `k`, event length, overlap).

### 4.2 Training configuration

Adam (`lr = 1e-3`), gradient accumulation of 8 instances, gradient clipping at
norm 1.0, `ReduceLROnPlateau` on `pp_span`, 100 epochs. Device: Apple MPS.
Checkpointing on best `pp_span` with `--assign model`.

**Outcome:** best checkpoint at **epoch 98**, validation `pp_span = 602.4`,
validation coverage 1.0. The validation span fell from 2190 (epoch 1) to 602
(epoch 98), a 3.6× reduction, with a long flat tail — i.e. clean convergence.

### 4.3 Evaluation protocol

`eval_paper.py` runs one model forward pass per instance, post-processes it in
all three assignment modes, and runs k-Inner, k-Budget, Baseline, and the Gurobi
ILP (30s/instance budget) once each. It records per-instance span and coverage,
aggregates by `k`, and computes approximation ratios against the
**proven-optimal** ILP subset. On this test set **all 200 ILP runs were proven
optimal within 30s**, so ratios are against the true optimum over the entire set.

---

## 5. Results

### 5.1 Headline (held-out test set, n = 200, all k)

| Method | Avg span | Mean ratio vs OPT | Median ratio | Coverage |
|--------|----------|-------------------|--------------|----------|
| ILP (optimal) | 506.4 | 1.000× | 1.000× | 1.0000 |
| **ML+DP (model)** | **601.0** | **1.185×** | **1.171×** | 1.0000 |
| ML+DP (hybrid) | 1110.1 | 2.181× | 1.727× | 1.0000 |
| k-Inner | 1528.8 | 3.023× | 2.792× | 1.0000 |
| ML+DP (greedy) | 1599.1 | 3.256× | 2.784× | 1.0000 |
| Baseline | 2692.5 | 5.368× | 4.953× | 1.0000 |
| k-Budget | 5273.8 | 10.543× | 9.155× | 1.0000 |

**ML+DP (model) is within 18.5% of the optimum on average and reduces span by
60.7% relative to the best heuristic (k-Inner)** and by 62.4% relative to the old
greedy decoder. Every method achieves perfect coverage; span is the sole
differentiator.

### 5.2 Per-k breakdown (average span)

| k | n | ILP | ML+DP(model) | model ratio | k-Inner | ML+DP(greedy) | Baseline | k-Budget |
|---|---|-----|--------------|-------------|---------|---------------|----------|----------|
| 2 | 59 | 385.0 | 438.5 | 1.14× | 1384.2 | 2011.9 | 2883.1 | 5330.2 |
| 3 | 37 | 451.5 | 539.0 | 1.19× | 1448.6 | 1692.7 | 2597.8 | 5263.0 |
| 4 | 29 | 541.2 | 625.9 | 1.16× | 1619.9 | 1482.8 | 2871.4 | 5705.6 |
| 5 | 27 | 575.7 | 698.1 | 1.21× | 1683.9 | 1411.8 | 2648.9 | 5385.7 |
| 6 | 19 | 521.3 | 655.9 | 1.26× | 1521.5 | 1135.5 | 2293.9 | 4586.7 |
| 7 | 11 | 613.9 | 722.7 | 1.18× | 1462.1 | 1129.8 | 2300.5 | 4565.6 |
| 8 | 7 | 802.4 | 954.3 | 1.19× | 1896.6 | 1363.6 | 2914.4 | 5857.3 |
| 9 | 9 | 769.8 | 956.7 | 1.24× | 1877.0 | 1319.8 | 2714.6 | 5423.6 |
| 10 | 2 | 707.0 | 845.5 | 1.20× | 1447.5 | 972.0 | 1881.0 | 3744.5 |

`model` mode beats k-Inner — the strongest heuristic — at **every** value of `k`.
The advantage is largest at **low `k`** (k=2: 438.5 vs k-Inner's 1384.2, a 3.2×
improvement). This **reverses** the earlier observation (from the greedy decoder)
that the learned method was weakest at low `k`: with the model actually driving
assignment, the tightly-constrained low-`k` regime is where learning the global
cover structure helps most.

### 5.3 Ablation: the value is in the learned signal

Same network, same DP, same test set; only the assignment signal changes:

| Assignment signal | Avg span | Interpretation |
|-------------------|----------|----------------|
| Random `p` (model-mode) | 3711.3 | pure scatter; no structure |
| Greedy span-cost | 1599.1 | weak local signal; still scatters |
| **Learned `p` (model-mode)** | **601.0** | global optimal structure ⇒ concentrate |

Random probabilities in `model` mode are **worse than greedy**, while learned
probabilities are **2.7× better than greedy**. This isolates the gain to the
*learned* assignment, not to the post-processing machinery: the decoder only
helps when it is fed the network's learned structure.

### 5.4 Interpretation of the greedy vs model gap

The old `greedy` decoder (3.26× optimal) actually trails k-Inner (3.02×),
reproducing the original report that "ML+DP" did not beat the heuristics. The
network was never the bottleneck — it had already learned the optimal assignment
from the ILP masks; the greedy decoder simply discarded that knowledge. Switching
the decoder to consume the prediction (`model` mode) yields the entire jump to
1.185× optimal, **with no change to the trained weights** beyond the
span-selected retrain.

---

## 6. The k = 1 case: comparison with DLMinTC+ (Lazzarinetti et al.)

The k = 1 setting is exactly the MinTCover+ problem solved by DLMinTC+, whose
objective — the sum-span `S(T) = Σ_u δ(I_u)` — is identical to ours. This lets us
ask whether our binary-mask + model-driven-decoding approach also wins at k = 1,
where DLMinTC+ uses a Pointer-Network head plus an iterative greedy fix-up.

### 6.1 Setup

A **dedicated k = 1 model** (same architecture, trained only on k = 1 data) was
trained on 1500 train + 300 val instances (100% ILP-optimal labels; n ∈ {20,30,50}).
Best checkpoint at epoch 19 (val sum-span 251.0), early-stopped at epoch 44.
Evaluation is on a fresh **held-out 200-instance test set** (seed 9001); all 200
ILP runs were proven optimal within 30 s, so ratios are against the true optimum.

### 6.2 Results (held-out test set, k = 1, n = 200)

| Method | Avg span | Mean ratio vs OPT | Median ratio |
|--------|----------|-------------------|--------------|
| ILP (optimal) | 234.2 | 1.000× | 1.000× |
| **ML+DP (model)** | **261.8** | **1.146×** | **1.031×** |
| INNER (k-Inner) | 736.0 | 2.826× | 2.633× |
| ML+DP (hybrid) | 2103.3 | 7.656× | 7.048× |
| ML+DP (greedy) | 2298.7 | 8.551× | 7.999× |
| Baseline | 2743.7 | 10.227× | 9.473× |
| k-Budget | 4109.1 | 15.564× | 14.200× |

**Our method reaches 1.146× optimal at k = 1 (median 1.031× — within 3% of optimal
on half the instances) and reduces span by 64% relative to INNER** (261.8 vs 736.0),
a heuristic DLMinTC+ also benchmarks against. All methods have perfect coverage.

Note that at k = 1 the old `greedy` decoder collapses to **8.55× optimal** — far
worse than INNER — whereas the `model` decoder is near-optimal. k = 1 is precisely
the regime where discarding the network hurts most (a single interval per vertex
leaves no slack), so this is the strongest confirmation of the decoder thesis of
Section 3: the learned assignment is what matters.

### 6.3 Relationship to DLMinTC+'s reported numbers

DLMinTC+ reports average sum-span reductions of **9.3%** (Dataset1, sparse) and
**7.5%** (Dataset2, dense) over FastMinTC+, at +29%/+38% execution time, with no
optimal baseline (their instances scale to 10,000 vertices / 5000 timestamps,
beyond exact ILP). Because the objective is identical, the comparison is
protocol-level rather than same-instance: we report our method against the **exact
optimum** on ILP-solvable sizes and against INNER, rather than a percentage over a
heuristic. Within-15%-of-optimal (median within 3%) is a strong absolute result on
this regime.

**FastMinTC+ reproduction caveat.** FastMinTC+ is DLMinTC+'s primary baseline, but
neither its code nor its datasets are public ("Data are contained within this
article"). We implemented FastMinTC+ from the paper (Algorithms 1–2) and verified
it produces valid covers, but could not reproduce its published performance without
the authors' reference implementation (our port underperforms even INNER on our
data). We therefore do **not** use its numbers as a baseline; a fabricated-weak
FastMinTC+ would only flatter our method. The honest, reviewer-robust comparison
is against the ILP optimum and INNER, with DLMinTC+'s FastMinTC+-relative figures
cited for context.

### 6.4 Caveats

- Our approximation ratios are on ILP-solvable instance sizes (small/medium);
  DLMinTC+'s headline is on much larger instances. The comparison establishes that
  our approach is near-optimal where the optimum is computable, not that it scales
  to their largest instances (see Section 8).
- The synthetic distribution (power-law configuration model + planted intervals)
  differs from Lazzarinetti's Dataset1/2/3; this is a same-protocol, not
  same-data, comparison.

---

## 7. Discussion / contribution framing

1. **A decoding insight for learned combinatorial optimisation.** The value of a
   learned seed is realised only if the decoding step consumes the learned
   structure. A feasibility-restoring decoder that ignores the seed (here, the
   span-greedy assignment) masks the model's value entirely. Making the decoder
   consume the prediction (`argmax p` assignment) turned a method that trailed
   the best heuristic into one within 18.5% of optimal.

2. **A clean k>1 extension of DLMinTC+.** A single `k`-conditioned network with a
   per-`(vertex, time)` binary-mask head plus a per-vertex `O(n²k)` interval DP
   generalises DLMinTC+ to arbitrary `k`, with the ≤k-interval constraint
   expressed as a differentiable transition penalty and guaranteed exactly by the
   DP.

3. **Train/eval alignment.** Selecting checkpoints on the post-processed span —
   the true objective — rather than the surrogate loss removes a second
   misalignment and is what the LR schedule and early stopping should track.

---

## 8. Limitations and future work

1. **Synthetic data only.** All graphs are power-law configuration-model graphs
   with synthetic interval activity. Validation on real temporal networks is the
   most important next step.
2. **Graph scale.** `n ∈ {20, 30, 50}` and `T ≤ 500`; the Transformer's `O(T²)`
   attention currently caps instance size. Scaling to larger temporal graphs
   (sparse/long-range attention, or windowing) is open.
3. **Single-pass post-processing.** Edges are assigned once and the DP runs once
   per vertex. An iterative scheme (reassign edges given current intervals,
   re-run the DP to a fixpoint) could close more of the remaining 18.5% gap to
   optimal, at extra runtime.
4. **`hybrid` underperforms `model`.** The simple `cost − β·log p` blend (2.18×)
   sits between greedy and model; `β` was not tuned. If a span-aware tie-break is
   ever needed (e.g. for adversarial inputs where the model is unreliable), `β`
   should be tuned on validation. For the present data, pure `model` is best.
5. **Ratio is "ratio of averages" in the per-k table.** The headline 1.185× is
   the mean of per-instance ratios; the per-k column ratios are ratios of average
   spans and are reported only for trend illustration.

---

## 9. Figures

Generated by `make_figures.py` from `experiment_results/test_eval.json`; both PNG
(300 dpi) and PDF (vector, for LaTeX) are written to `figures/`.

| File | Content | Suggested use |
|------|---------|---------------|
| `fig1_approx_ratio_by_k` | Mean approximation ratio vs k per method (log y). `model` is a near-flat line just above 1.0; all others rise well above. | Main results figure |
| `fig2_span_by_k` | Mean span vs k per method (log y), with the ILP optimal floor. | Results / appendix |
| `fig3_ablation_ladder` | random-p (7.33×) → greedy (3.16×) → learned-model (1.19×) → optimal (1.00×). | The decoder-insight figure |
| `fig4_overall_ratio` | Overall mean approximation ratio per method (horizontal bars). | Headline summary |
| `fig5_ratio_boxplot` | Per-instance ratio distributions: `model` is tight just above 1.0, greedy/k-Inner are wide around ~2.8. | Shows low variance, not just low mean |
| `fig6_scatter_vs_optimal` | Per-instance span vs ILP optimum; `model` hugs the y=x diagonal, k-Inner fans out. | Per-instance evidence |

Regenerate with:
```bash
python make_figures.py --json experiment_results/test_eval.json --out figures
```

---

## 10. Reproducibility

```bash
# 1. Data (train/val with ILP-or-heuristic labels; test with a fresh seed)
python data_pipeline.py --n 2000 --val-n 500 --ilp-budget 60 --max-t 500 --n-nodes 20,30,50
python data_pipeline.py --n 200 --out data/test --ilp-node-limit 0 --ilp-edge-limit 0 \
    --max-t 500 --n-nodes 20,30,50 --seed 7777

# 2. Train (model-mode assignment drives span-based checkpoint selection)
python train.py --train data/train --val data/val --epochs 100 --assign model

# 3. Held-out evaluation (all assign modes + heuristics + proven-optimal ILP)
python eval_paper.py --test data/test --checkpoint checkpoints/best_model.pt \
    --ilp-budget 30 --out experiment_results/test_eval.json

# Ablation comparison of decoders on the same checkpoint
python evaluate.py --val data/test --checkpoint checkpoints/best_model.pt --assign greedy --ilp-budget 0
python evaluate.py --val data/test --checkpoint checkpoints/best_model.pt --assign model  --ilp-budget 0
```

Raw per-instance results are saved to `experiment_results/test_eval.json`.

### k = 1 comparison (Section 6)

```bash
# Data: dedicated k=1 splits (--k-min 1 --k-max 1)
python data_pipeline.py --n 1500 --out data/train_k1 --val-n 300 --val-out data/val_k1 \
    --k-min 1 --k-max 1 --ilp-budget 30 --max-t 500 --n-nodes 20,30,50 --seed 100
python data_pipeline.py --n 200 --out data/test_k1 --k-min 1 --k-max 1 \
    --ilp-node-limit 0 --ilp-edge-limit 0 --max-t 500 --n-nodes 20,30,50 --seed 9001

# Train the dedicated k=1 model
python train.py --train data/train_k1 --val data/val_k1 --epochs 80 --assign model \
    --checkpoint-dir checkpoints_k1

# Evaluate vs ILP-optimal + INNER (+ other baselines)
python eval_paper.py --test data/test_k1 --checkpoint checkpoints_k1/best_model.pt \
    --ilp-budget 30 --out experiment_results/test_eval_k1.json
```

Raw k = 1 results: `experiment_results/test_eval_k1.json`. `fastmintc.py` holds the
best-effort FastMinTC+ port (valid but below published performance; see Section 6.3).

### Key code references
- `postprocess.py:_assign_edges` — the three assignment modes (Section 3.2)
- `train.py:evaluate` — `pp_span` computation and span-based selection (Section 3.4)
- `eval_paper.py` — single-pass evaluation harness with optimality tracking
- `ilp_version.py` — the reference ILP (Section 1)
- `fastmintc.py` — best-effort FastMinTC+ reimplementation (Section 6.3)
- `k_inner.py` — INNER; the `-(k-1)` slice was fixed to support k = 1
