# Real-World Datasets and Applications for Minimum Timeline Cover

A survey of candidate real-world datasets for evaluating DLMinTCk on the Minimum
Timeline Cover (MinTCover) problem, with sources, sizes, licensing, fit
assessment, and application framing. This is a planning document written to
become a "Datasets and Applications" section of the paper; no dataset has been
run yet (survey-first).

---

## 1. Motivation: the real-world gap

- The **TIME 2025** paper (our prior work) evaluates on **synthetic data only**.
- **DLMinTC+** (Lazzarinetti et al.) uses synthetic sparse/dense instances plus a
  real-world "DIMACS" set, but reports **no optimal baseline** — its instances
  scale to 10,000 vertices, beyond exact ILP.
- The **originators'** code (Rozenshtein, Tatti, Gionis; DMKD 2020;
  github.com/polinapolina/the-network-untangling-problem) ships the reference
  heuristics (Inner/Budget/Baseline, k-Inner/k-Budget/k-Baseline) but **no
  datasets** — their real case study (Twitter) is not bundled.

**Opportunity.** Real-world validation *with the exact optimum as reference* on
ILP-solvable instances is a claim none of the prior work makes. It is directly
enabled by our pipeline (ILP labels + approximation-ratio reporting) and is a
clean contribution.

---

## 2. The problem-to-application mapping

MinTCover's premise: *each entity is active during a small number (k) of time
intervals; an interaction (u, v, t) is explained if at least one endpoint is
active at t; minimise the total active time (sum-span).*

This is precisely the semantics of **contact and communication logs**:

| Domain | Interaction (u, v, t) | Activity interval | "Minimise total active time" means |
|--------|-----------------------|-------------------|-------------------------------------|
| Face-to-face contact | u and v are physically near at time t | when a person is present/active at a venue | most parsimonious presence windows explaining all contacts |
| Epidemic / contact tracing | a proximity event | an infectious/active window | smallest total exposure time consistent with the contact log |
| Communication | u messages/emails v at t | a user's active/online period | tightest active periods explaining all messages |
| Event participation | two attendees converse at t | an attendee's engagement window | minimal engagement time covering all conversations |

**Why k > 1 is natural on real data.** Synthetic MinTCover often assumes one
interval per node (k = 1). Real contact/communication data almost always has
**multiple activity windows** per entity (a nurse across several shifts, a user
online on several days), which directly motivates our k > 1 extension — arguably
a better fit for reality than the original single-interval formulation.

---

## 3. Fit criteria (what "suits our algorithm")

1. **Format**: timestamped pairwise interactions `(u, v, t)` — a temporal graph.
2. **Semantic validity**: a meaningful notion of node "activity interval."
3. **ILP-tractability (at least partial)**: small enough node/timestamp counts
   that Gurobi returns the optimum on some instances → we can report
   approximation ratios (our signature strength).
4. **Manageable T after binning**: the Transformer is O(T²); raw resolutions
   (e.g. 20 s) must be binned so the number of distinct timestamps T ≲ 500.
5. **Credibility**: datasets already used in the temporal-networks literature.

---

## 4. Candidate datasets

### Tier A — ILP-feasible face-to-face contact networks (SocioPatterns)

RFID proximity logs at 20 s resolution; format `t i j` (a contact between i and j
during the 20 s window ending at t). Source: **www.sociopatterns.org**. License:
freely available for research **with attribution / citation** (per-dataset terms;
generally non-commercial). These are the best fit: small node counts give an
**ILP-optimal reference on real data**, and multi-day presence naturally yields
k > 1.

| Dataset | Nodes | Contacts | Setting | Fit notes |
|---------|-------|----------|---------|-----------|
| Hospital ward (LyonDataset) | ~75 | ~32k | patients + staff, 4 days | **ILP-optimal feasible**; staff shifts → clean k>1; epidemic story |
| Primary School | ~240 | ~125k | pupils + teachers, 2 days | class-structured; medium |
| High School (2011–2013) | ~180 | ~45k | students, several days | ILP-tractable; social presence |
| Workplace (InVS13/15) | ~90–220 | ~78k | office, ~2 weeks | multi-day → strong k>1 motivation |
| SFHH conference | ~403 | ~70k | conference attendees, 2 days | event-participation; tests scaling (heuristic reference) |

### Tier B — Communication / messaging (SNAP)

Larger; mostly **heuristic reference** (beyond exact ILP), useful to demonstrate
scaling. Source: **snap.stanford.edu/data**. Free for research with citation; no
explicit license. Format: `SRC DST UNIXTS`.

| Dataset | Nodes | Temporal edges | Span | Fit notes |
|---------|-------|----------------|------|-----------|
| CollegeMsg | 1,899 | 59,835 | 193 days | UC Irvine private messages; "active period" mining |
| email-Eu-core-temporal | ~986 | ~332k | 803 days | dept email; heavy T → aggressive binning needed |
| (sx-*) Stack Exchange temporal | 10^4–10^6 | large | years | scaling stress-test only; heuristic reference |

### Tier C — Literature-matched / collections

- **Rozenshtein et al. repo** — reference heuristics (already mirrored by our
  `k_inner.py` / `k_budget.py` / `baseline.py`); their **Twitter hashtag
  co-occurrence** case study is the canonical real application but the data is
  not bundled (would require reconstruction).
- **DIMACS** (DLMinTC+'s Dataset3) — for a same-source comparison to DLMinTC+,
  though DIMACS graphs are largely static and require an added temporal model.
- **Collections**: `github.com/4AlexMin/dynamic-networks` (79 temporal datasets),
  Network Repository, KONECT — for breadth / robustness experiments.

---

## 5. Recommended experimental plan

1. **Primary (real-world + optimum)**: SocioPatterns **Hospital** and **High
   School**. Small node counts → ILP-optimal reference. Report our approximation
   ratio vs ILP on real data, plus vs INNER — the claim no prior work makes.
   Natural k > 1 from multi-day/multi-shift presence.
2. **Scaling (heuristic reference)**: SFHH conference and CollegeMsg — show the
   method degrades gracefully where ILP is infeasible, comparing to INNER only.
3. **Optional same-source**: a DIMACS-derived instance to touch DLMinTC+'s
   Dataset3, with the temporal-model caveat stated.

---

## 6. Preprocessing considerations

- **Timestamp binning**: bin raw times (20 s for SocioPatterns; seconds for SNAP)
  to keep T ≲ 500 (the O(T²) attention guard, `--max-t`). Binning granularity
  (minute / 10-minute / hour) is a modelling choice to sweep; coarser bins reduce
  T and change the sum-span units.
- **Node reindexing**: to consecutive 0-based IDs (already handled by
  `data_pipeline.reindex`).
- **Choosing k**: on real data k is a parameter, not ground truth. Report results
  across k ∈ {1, …, K} and let the objective/coverage trade-off guide the choice
  (e.g. shift count for Hospital staff suggests a natural K).
- **Instance slicing**: very large logs can be windowed (per-day / per-session)
  into multiple instances, keeping each ILP-tractable.

---

## 7. Evaluation on real data (metric notes)

- **No planted ground truth**: unlike synthetic data, real logs have no true
  activity intervals, so precision/recall-vs-planted-truth (Rozenshtein-style) is
  **not** available. We evaluate on **sum-span** and **coverage** (always 1.0 by
  construction), and — where ILP terminates — the **approximation ratio vs the
  optimum**. This keeps the headline metric identical to the synthetic study.
- **Baselines**: ILP-optimal (small instances), INNER, k-Budget, Baseline; our
  ML+DP(model). FastMinTC+ only if the port is hardened (currently not trusted).
- **Transfer question**: whether a model trained on synthetic data transfers to
  real contact networks, or needs fine-tuning per domain, is itself a result.

---

## 8. Licensing / attribution summary

| Source | Terms | Action needed |
|--------|-------|---------------|
| SocioPatterns | Free for research, **cite the dataset paper**; generally non-commercial | Add per-dataset citations; check each dataset's page |
| SNAP | Free for research, **cite source paper** (e.g. Panzarasa et al. 2009 for CollegeMsg) | Add citations |
| Rozenshtein repo | No explicit license | Use as algorithm reference only; do not redistribute |
| DIMACS | Varies by collection | Verify before use |

None of the Tier A/B datasets are committed to this repo; they are downloaded at
run time and kept under the gitignored `data/` directory.

---

## 9. Open questions for the author

1. Target application framing for the paper — **epidemic/contact-tracing**
   (Hospital/School) reads strongest given the ILP-optimal angle; confirm.
2. Is a **same-source DIMACS** comparison to DLMinTC+ worth the temporal-modelling
   caveat, or is beating INNER + near-optimality on SocioPatterns sufficient?
3. Preferred **binning granularity** / whether to sweep it as a sensitivity study.
