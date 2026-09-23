# Archive of the pre-OPERA research direction (frozen 2026-09-23)

Kept for historical reproducibility only. Nothing in here is imported by the active OPERA code (`src/`).
Git branches `phase1-sampling-invariant`, `phase2-adaptive-query`, `phase3-structural-prior` hold the full
history of these experiments.

| folder | contents |
|---|---|
| `old_consistency/` | job scripts of the sampling-view + representation-consistency experiments |
| `old_adaptive/` | job script of the observation-statistics query experiments (the mechanism itself survives as the optional `ObservationAwareQuery` in `models/embeddings/quite.py`) |
| `old_causal/` | causal attention bias (`causal.py`), synthetic structural dataset, LLM physiological prior, graph estimation, and their job / analysis scripts |
| `old_scripts/` | the previous training entrypoints (`train_forecasting.py`, `train_classification.py`, `train_quite_plus.py`), `evaluation.py` with the consistency objective, `sampling_views.py`, checkpoint re-evaluation and summary scripts |
| `old_results/` | all per-run result JSONs (`p0_*`, `p1_*`, `p2_*`, `p3_*`, `p3low_*`, `p3phys_*`, `p3b_*`, `*.stuck.json`) |
| `old_reports/` | `PHASE01_REPORT.md`, `PHASE2_REPORT.md`, `PHASE3_REPORT.md`, summary tables |
| `old_paper/draft_v0/` | the first manuscript draft (ACM sigconf), which discussed the consistency and causal negative results |

These numbers are historical and were obtained under a reduced training budget (epoch cap 30, 2-3 seeds).
They must not be copied into OPERA reports; OPERA re-runs everything under its own protocol.
