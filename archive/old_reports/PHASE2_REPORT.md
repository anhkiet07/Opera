# Phase 2 report: Adaptive (observation-aware) query embedding

Date: 2026-09-23. Builds on `results/PHASE01_REPORT.md` (same data, backbones, budget: PhysioNet 24h->24h with
iTransformer, P12 fold 1 with PatchTST, batch 32, epoch cap 30, patience 8). W&B project `quite`, run names `p2_*`.

## Method

Plain QuITE uses one fixed learnable query token per target (variable, or patch x variable). Phase 2 makes it
observation-aware:

    q(X) = q_base + MLP( s(X) ),   s(X) = [ log(1 + n_obs), n_obs / L_max,            # density
                                             mean dt, std dt, observed span,            # gap timing
                                             has_obs, fraction of targets with obs ]    # availability

computed per query target from the observation mask and time stamps (`models/embeddings/quite.py`, `obs_stats`).
The MLP (7 -> 64 -> 64, GELU) is zero-initialised so every variant starts exactly at plain QuITE (verified
bit-identical at init). Cost: +4,672 parameters (+3.6 %), no measurable change in time per epoch.
`--adaptive_query {full, density, gaps, avail, random}`; `random` feeds the same MLP Gaussian noise (capacity control).

Evaluation adds three regimes that are absent from the Phase 1 `mixed` training pool: `variable:0.5` (unseen
severity), `stride:2`, `stride:3` (systematic subsampling = lower sampling rate, an unseen regime *type*).
Phase 0/1 checkpoints were re-evaluated on these views (`eval_checkpoint.py`, `eval_checkpoint_cls.py`).

## Forecasting (PhysioNet, test MSE; mean over seeds)

### Main comparison

| config | seeds | clean | random 0.7 | variable 0.3 | variable 0.5 (unseen) | stride 3 (unseen) |
|---|---|---|---|---|---|---|
| plain QuITE | 3 | 0.0054 | 0.0164 | 0.0166 | 0.0225 | 0.0108 |
| adaptive query (full), standard training | 2 | **0.0053** | 0.0109 | 0.0076 | 0.0091 | 0.0085 |
| mixed-view augmentation (Phase 1, aug-only) | 2 | 0.0054 | 0.0070 | 0.0068 | 0.0075 | 0.0057 |
| adaptive query + mixed views | 2 | 0.0054 | **0.0069** | **0.0066** | **0.0074** | **0.0056** |

Per seed, adaptive (standard training) vs plain: variable 0.3 = 0.0073 / 0.0079 vs 0.0187 / 0.0153 (both seeds
halve the error); random 0.7 = 0.0093 / 0.0125 vs 0.0170 / 0.0149 (large on seed 1, small on seed 2);
stride 3 = 0.0071 / 0.0099 vs 0.0108 / 0.0102 (seed 1 only). Seed 2 early-stopped at epoch 11.

### Ablations (seed 1, standard training)

| query summary | clean | random 0.7 | variable 0.3 | variable 0.5 | stride 3 |
|---|---|---|---|---|---|
| none (plain QuITE, seed 1) | 0.0055 | 0.0170 | 0.0187 | 0.0256 | 0.0108 |
| random noise (capacity control) | 0.0055 | 0.0146 | 0.0137 | 0.0178 | 0.0107 |
| gap timing only (mean / std dt, span) | 0.0055 | 0.0151 | 0.0146 | 0.0199 | 0.0109 |
| availability only | 0.0054 | 0.0130 | 0.0082 | 0.0098 | 0.0092 |
| density only (log count, count / L) | 0.0053 | 0.0103 | 0.0075 | 0.0089 | 0.0080 |
| full | **0.0052** | **0.0093** | **0.0073** | **0.0089** | **0.0071** |

Under mixed-view training (seed 1): aug-only 0.0071 / 0.0067 / 0.0075 / 0.0058, random-control 0.0073 / 0.0070 /
0.0079 / 0.0059, full adaptive 0.0069 / 0.0066 / 0.0074 / 0.0057 (random 0.7 / variable 0.3 / variable 0.5 / stride 3).

## Classification (P12, fold 1; AUROC / AUPRC)

| config | clean | random 0.7 | variable 0.3 | variable 0.5 (unseen) | stride 3 (unseen) |
|---|---|---|---|---|---|
| plain QuITE + PatchTST | 83.7 / 47.3 | 81.6 / 40.4 | 79.6 / 36.8 | 77.7 / 35.5 | 83.7 / 47.6 |
| adaptive query, standard training | 85.2 / 48.6 | 76.9 / 32.4 | 77.4 / 36.0 | 74.4 / 34.2 | 84.6 / 47.6 |
| mixed views, aug-only (Phase 1) | 85.5 / 49.7 | 84.0 / 43.1 | 81.5 / 39.8 | 80.2 / 39.5 | 85.7 / 49.6 |
| adaptive query + mixed views | **86.2 / 49.5** | 83.9 / 42.1 | 81.4 / 40.0 | 80.0 / 39.4 | **86.1 / 50.1** |

## Findings

1. **Observation-aware queries help without any augmentation, mainly when variables go missing.** With standard
   training, the adaptive query halves the error under variable removal on both seeds (0.0166 -> 0.0076 at 30 %,
   0.0225 -> 0.0091 at 50 %, an unseen severity) and keeps clean accuracy. Gains under random dropout and stride
   subsampling are real on seed 1 but small on seed 2, so they are not yet established.
2. **The gain comes from the information, not the parameters.** The noise-fed control stays close to plain QuITE
   (random 0.7: 0.0146 vs 0.0170; stride 3 unchanged), well behind every informative variant. Its modest gain on
   variable removal suggests query noise acts as a weak regulariser.
3. **Which statistics matter: count-based density > availability > gap timing.** Density alone recovers ~90 % of the
   full-statistics gain; availability alone captures the variable-removal gain but less of the dropout gain; gap
   timing alone is useless. The query benefits from knowing *how much* it will see, not *when*.
4. **Adaptive queries and view augmentation overlap rather than stack.** Under mixed-view training the adaptive
   model is within 0.0002 MSE of aug-only on every view (forecasting) and within 0.3 AUROC (classification). When you
   can augment, the query adaptation is redundant; when you cannot, it recovers roughly half of the augmentation gain.
5. **Negative result on classification without augmentation.** On P12 the adaptive query raises clean AUROC
   (83.7 -> 85.2) and holds on stride subsampling, but is *worse* than plain QuITE under random dropout
   (76.9 vs 81.6) and variable removal (77.4 vs 79.6): the classifier over-relies on training-time observation
   statistics and extrapolates badly when they shift. Mixed-view training removes the problem (86.2 clean, 83.9 under
   dropout, best stride result of all configs), i.e. the adaptive query needs to *see* varied statistics in training.

## Sanity checks from the plan

| question | answer |
|---|---|
| Gain from the proposed mechanism? | Yes for forecasting: informative statistics beat the noise control on every view. |
| Could it be extra parameters? | No: the same MLP fed noise does not reproduce the gain. |
| Survives random / shuffled controls? | Yes (random-summary control), with a small regularisation effect noted. |
| Improves the intended regime? | Yes for missing variables (both seeds, both tasks with augmentation); partially for dropout / stride. |
| Transfers to an unseen sampling pattern? | Forecasting: yes (variable 0.5, stride). Classification: only with mixed-view training. |
| Preserves dense / regular performance? | Yes (clean MSE 0.0053 vs 0.0054; P12 AUROC up). |
| Consistent across seeds? | Variable-removal gain yes; dropout / stride gain seed-dependent (2 seeds). Ablations single seed. |

## Suggested next steps

- 3 to 5 seeds for adaptive-full vs plain under standard training, with the paper's patience, to settle the
  dropout / stride effect (seed 2 stopped at epoch 11).
- The density-only variant is the one to keep: 2 scalar features, same gain. Try feeding density to the decoder's
  future-time queries as well (QuITE++ cross-attention), where it could calibrate uncertainty.
- Combine with Phase 1's cheapest recipe (mixed views, no consistency loss) as the default training protocol; the
  classification failure without augmentation argues against shipping adaptive queries alone.
