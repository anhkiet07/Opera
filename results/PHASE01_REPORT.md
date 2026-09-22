# Phase 0 + Phase 1 report: Reproducing QuITE and testing Sampling-Invariant QuITE

Date: 2026-09-22. All runs: W&B project `quite` (https://wandb.ai/causal_rl/quite), raw numbers in `results/*.json`,
aggregate tables in `results/summary.md` (regenerate with `python summarize_results.py`).

## Setup

| item | value |
|---|---|
| Forecasting | PhysioNet 2012, history 24 h -> horizon 24 h, iTransformer backbone, hid 64, 4 heads, 3 layers, patch 6 |
| Classification | P12 (Raindrop split 1 only), PatchTST backbone, hid 64, 4 heads, 3 layers, patch 6 |
| Training | Adam, lr 1e-3, batch 32, epoch cap 30, early-stop patience 8 on validation MSE / CE |
| Hardware | RTX 3070 Laptop (8 GB), Windows, torch 2.14 + CUDA 12.6 |
| Seeds | plain QuITE: 3 seeds; Phase 1 main comparison: 2 seeds; ablations and classification: 1 seed / 1 fold |

Deviations from the paper protocol (paper: patience 50, up to 1000 epochs, batch 64, 5 seeds / 5 folds) were made to
fit the GPU budget. They mostly affect the two-view runs, which hit the 30-epoch cap with validation still improving
slowly, so their clean-data numbers are slightly pessimistic.

## Phase 0: reproduction and cost

| config | test MSE | test MAE | params | s / epoch | peak GPU | inference |
|---|---|---|---|---|---|---|
| QuITE + iTransformer (3 seeds) | 0.0054 +- 0.0001 | 0.0382 +- 0.0004 | 129,153 | 24.6 | 3.7 GB | 0.63 ms / series |
| vanilla iTransformer (no QuITE, 1 seed) | 0.0178 | 0.0767 | 123,393 | 9.4 | n/a (eval-only peak 0.5 GB) | 0.45 ms / series |
| QuITE + PatchTST, P12 (fold 1) | AUROC 83.7 / AUPRC 47.3 | acc 83.0 | 108,802 | 35.4 | 2.7 GB | 1.37 ms / sample |

- QuITE + iTransformer lands in the paper's range for PhysioNet 24 h and is stable across seeds (spread 1e-4).
- The vanilla backbone is 3.2x worse in MSE (0.0178 vs 0.0055) and degrades further under re-observed inputs
  (0.0347 at 70 % random dropout, 0.0416 with 30 % of variables removed), confirming that the query-based irregular
  embedding is what makes the standard MTS backbone usable on IMTS. Its in-script robustness evaluation crashed on an
  input-layout bug (since fixed), so it was finalised from the saved checkpoint and its training peak memory was not recorded.
- The data pipeline (t-PatchGNN-style collate, min-max normalisation, masked MSE / MAE / MAPE) and the Raindrop
  classification pipeline were exercised end to end; two Windows-only bugs in the PhysioNet downloader were fixed.

## Phase 1: Sampling-Invariant QuITE

Method (`sampling_views.py`): every training batch is re-observed through sampling *views* that only edit the
observation mask: `random:p` (drop each observation w.p. p), `bursty:p` (drop contiguous time windows covering a
fraction p of the observed range), `variable:p` (drop whole variables), `mixed` (uniform over
random 0.3 / 0.6, bursty 0.3 / 0.5, variable 0.3). With two views A, B per batch the loss is
`mean(L_task(A), L_task(B)) + lambda * D(z_A, z_B)` with `D` = cosine distance between the encoder representations
(or MSE between predictions for the `pred` variant). Targets are never touched.

Evaluation: the best checkpoint is tested on the original test inputs (`none`) and on the same inputs re-observed
through each view with a fixed seed. "Relative degradation" is the mean MSE over the six degraded views relative to
`none`.

### Main comparison (forecasting; plain QuITE 3 seeds, others 2 seeds)

| config | clean MSE | random 0.3 | random 0.5 | random 0.7 | bursty 0.3 | bursty 0.5 | variable 0.3 | mean degraded | rel. degradation |
|---|---|---|---|---|---|---|---|---|---|
| plain QuITE (3 seeds) | 0.0054 | 0.0088 | 0.0119 | 0.0164 | 0.0070 | 0.0079 | 0.0166 | 0.0114 | +111.1 +- 8.4 % |
| + mixed views, no consistency (aug-only) | **0.0054** | **0.0058** | **0.0063** | **0.0070** | **0.0056** | **0.0057** | **0.0068** | **0.0062** | +14.8 +- 0.7 % |
| + mixed views + consistency, lambda 0.1 | 0.0056 | 0.0060 | 0.0064 | 0.0071 | 0.0057 | 0.0058 | 0.0070 | 0.0063 | **+13.8 +- 0.1 %** |

Seed standard deviations are <= 0.0001 MSE on every cell, so differences of 0.0002 or more are real.

### Ablations (forecasting, seed 1)

| config | clean MSE | random 0.7 | bursty 0.5 | variable 0.3 | mean degraded | rel. degradation |
|---|---|---|---|---|---|---|
| aug-only (reference, seed 1) | 0.0054 | 0.0071 | 0.0057 | 0.0067 | 0.0062 | +14.1 % |
| consistency lambda 0.01 | 0.0056 | 0.0073 | 0.0059 | 0.0070 | 0.0064 | +14.7 % |
| consistency lambda 0.1 | 0.0056 | 0.0072 | 0.0058 | 0.0070 | 0.0063 | +13.7 % |
| consistency lambda 1.0 | 0.0059 | 0.0074 | 0.0061 | 0.0072 | 0.0066 | +12.5 % |
| consistency on predictions (lambda 0.1) | 0.0057 | 0.0070 | 0.0058 | 0.0068 | 0.0063 | +10.6 % |
| single view type: random 0.5 only (lambda 0.1) | 0.0059 | 0.0073 | 0.0061 | 0.0087 | 0.0068 | +15.1 % |
| single view type: bursty 0.5 only (lambda 0.1) | 0.0057 | 0.0080 | 0.0059 | 0.0101 | 0.0071 | +25.5 % |

Collapse diagnostics (per-epoch `repr/pos_cos` = cosine between the two views of the same sample, `repr/neg_cos` =
cosine between views of *different* samples, `repr/std` = feature std over the batch; last 3 epochs, seed 1):

| config | pos_cos | neg_cos | repr std |
|---|---|---|---|
| aug-only | 0.896 | 0.807 | 0.413 |
| consistency on predictions, lambda 0.1 | 0.909 | 0.836 | 0.377 |
| representation consistency, lambda 0.01 | 0.996 | 0.992 | 0.113 |
| representation consistency, lambda 0.1 | 1.000 | 0.999 | 0.044 |
| representation consistency, lambda 1.0 | 1.000 | 1.000 | 0.018 |

The representation-level cosine loss is satisfied by making *all* samples' representations point the same way
(neg_cos -> 1, std -> 0), i.e. partial collapse, rather than by making the two views of a sample agree while
keeping samples apart. The task loss keeps enough variance for the decoder, which is why test error barely moves,
but the invariance term is not doing what it was meant to do. Prediction-level consistency does not have this problem.

### Classification (P12, fold 1)

| config | clean AUROC / AUPRC | random 0.7 | bursty 0.5 | variable 0.3 |
|---|---|---|---|---|
| plain QuITE + PatchTST | 83.7 / 47.3 | 81.6 / 40.4 | 82.7 / 45.2 | 79.6 / 36.8 |
| + mixed views, aug-only | **85.5 / 49.7** | 84.0 / 43.1 | **85.3 / 47.1** | 81.5 / **39.8** |
| + mixed views + consistency 0.1 | 85.5 / 48.4 | **84.2 / 43.8** | 85.3 / 45.6 | **81.6** / 39.0 |

## Findings

1. **Plain QuITE is not sampling-invariant.** Re-observing the test inputs through a different irregular pattern
   roughly triples forecasting MSE (70 % random dropout, or 30 % of variables removed) and costs 2 to 7 AUPRC points
   on P12. This confirms the premise of Idea 1.
2. **Observation-view augmentation removes most of the gap at no cost on clean data.** With mixed views the relative
   degradation falls from +111 % to +15 % (forecasting) and clean performance is unchanged or better (P12 AUROC
   83.7 -> 85.5). The effect is identical across seeds.
3. **The explicit consistency loss adds nothing measurable, and the representation-level version partially
   collapses.** Across lambda in {0.01, 0.1, 1.0}, representation vs prediction targets, and both tasks, consistency
   runs are within noise of aug-only on degraded views and slightly worse on clean data at large lambda (0.0059 at
   lambda 1.0). The diagnostics show why: the cosine penalty on the flattened encoder output is minimised by aligning
   every sample's representation to a common direction (between-sample cosine 0.99 to 1.00, feature std 0.02 to 0.11
   vs 0.41 without the term). The mechanism behind the robustness gain is exposure to varied sampling, not the
   invariance penalty. A future invariance objective must keep samples apart (e.g. contrastive / VICReg-style
   variance terms, or consistency on centred, per-variable representations).
4. **A mixture of view types is required for transfer to an unseen regime.** Training on random dropout only leaves
   variable removal at 0.0087; training on bursty windows only leaves random 70 % at 0.0080 and variable removal at
   0.0101. Each single regime protects itself and partially its neighbours, but not the unseen one.
5. **Cost.** Two-view training costs about 1.6x time per epoch and 1.7x peak memory (3.7 -> 6.2 GB); inference cost
   and parameter count are unchanged because the architecture is untouched.

## Sanity checks from the plan

| question | answer |
|---|---|
| Gain from the proposed mechanism? | The gain is from view augmentation; the consistency term is a negative result. |
| Could it be extra parameters? | No: 129,153 parameters in every QuITE forecasting config (views and consistency add none). |
| Survives random / shuffled controls? | Aug-only is the control for the consistency term and matches it; the shuffled-negative cosine exposes partial collapse of the representation-level term. |
| Improves the intended regime? | Yes, every degraded view improves by 20 to 60 % MSE. |
| Transfers to an unseen pattern? | Only when trained on a mixture of regimes (finding 4). |
| Preserves dense / regular performance? | Yes on the original pattern (clean MSE unchanged, P12 AUROC up). |
| Consistent across seeds? | Yes for the forecasting comparison (3 / 2 seeds, std <= 1e-4 MSE). Ablations and classification are single seed / fold. |

## Suggested next steps

- Re-run the three main configs with the paper protocol (patience 50, batch 64 if memory allows) and 3 to 5 seeds to
  remove the epoch-cap caveat; add USHCN or Activity as a second forecasting dataset and a second backbone (PatchTST).
- Since augmentation alone works, the strongest framing is "observation-process augmentation for irregular time
  series" with the invariance loss reported as a negative control, rather than a consistency-loss method.
- Phase 2 (adaptive queries) can reuse the same view machinery for its mixed-regime train / unseen-regime test protocol.
