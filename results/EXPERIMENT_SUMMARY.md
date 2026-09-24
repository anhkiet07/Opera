# OPERA experiment summary (22 runs, seed 0)

Date: 2026-09-24. Branch `opera`. Raw results `results/raw/<run>/results.json`, per-dataset tables
`results/aggregated/tables_<dataset>.md`, figures `figures/`, W&B project `opera`.

**Scope and caveat.** Every number is **one seed (0)**. The plan requires 5 seeds for a primary claim, so
nothing here is a statistical result. What this pass establishes is the direction and magnitude of each effect
across 5 datasets, 6 task settings, 2 tasks and 2 backbones, and which mechanisms deserve the seeds.

## Coverage

| setting | task | variables | samples | backbone | metric |
|---|---|---|---|---|---|
| PhysioNet 24h -> 24h | forecasting | 41 | 12,000 | iTransformer | MSE |
| Activity 3000ms -> 1000ms | forecasting | 12 | 5,400 | iTransformer | MSE |
| USHCN 24mo -> 1mo | forecasting | 5 | 26,736 | iTransformer | MSE |
| USHCN 24mo -> 12mo | forecasting | 5 | 26,736 | iTransformer | MSE |
| P12 mortality | classification (2) | 36 | 11,988 | PatchTST | AUROC / AUPRC |
| P19 sepsis | classification (2) | 34 | 38,803 | PatchTST | AUROC / AUPRC |
| PAM activity | classification (8) | 17 | 5,333 | PatchTST | accuracy / macro F1 |

Observation processes come from one API (`src/observation/`). Training draws from a mild pool (random 0.1-0.4,
stride 2, burst 0.2-0.3, variable 0.2, local mask 0.3). Evaluation re-observes the **history only** of the test
inputs through 16 conditions with a fixed seed, targets untouched; 9 of the 16 never occur in training.

## Q1 - Does observation-process shift matter? Yes, and its size spans three orders of magnitude.

| setting | clean | worst condition | mean degradation |
|---|---|---|---|
| Activity | 0.00263 | 0.08547 (stride 4) | **+1793 %** |
| PhysioNet | 0.00549 | 0.02134 (variable 0.5) | **+107 %** |
| PAM | 88.01 % acc | 53.93 % (variable 0.5) | **+64 %** (CE) |
| P12 | 83.53 AUROC | 77.98 (random+variable) | +4.4 % (CE) |
| USHCN 12-mo | 0.5202 | 0.5534 (variable 0.5) | +3.7 % |
| P19 | 82.13 AUROC | 74.63 (variable 0.3) | +2.9 % |
| USHCN 1-mo | 0.5379 | 0.5693 (variable 0.5) | +1.9 % |

The size tracks how much predictive information lives in the irregular history. At the low end, USHCN with a
1-month horizon *improves* when half its observations are removed at matched density (0.5355 vs 0.5379 clean):
that task is carried by marginal/seasonal structure and barely uses its history. Lengthening its horizon to 12
months doubles the sensitivity (+1.9 % -> +3.7 %), confirming the mechanism of the null rather than excusing it.

**It is not explained by observation count.** Processes removing ~50 % of observations differ by up to 2.9x on
PhysioNet (density-matched 0.0074, stride 0.0079, burst 0.0080, variable dropout 0.0213) and 3.5x in CE on PAM.

**Which shift hurts depends on the sensing modality.** On the clinical datasets, losing whole variables is the
damaging shift and time-axis shifts are nearly free (P19: variable 0.3 costs 7.5 AUROC, stride 3 costs 1.1;
PAM: variable 0.5 costs 34 accuracy points, stride 3 costs 1.5). On Activity the opposite holds: stride 4 is the
single worst condition at 32x clean error. Body-sensor dynamics at millisecond resolution genuinely depend on
sampling rate; hourly lab values do not.

## Q2-Q5 - Which mechanism contributes?

### PhysioNet (test MSE), the full component sweep

| config | clean | mean shifted | worst | degradation | s/epoch |
|---|---|---|---|---|---|
| E0 baseline | **0.00549** | 0.01137 | 0.02134 | +107.0 % | 24 |
| E1 augmentation | 0.00553 | 0.00632 | 0.00772 | +14.3 % | 41 |
| E2 + distillation | 0.00573 | 0.00644 | 0.00776 | +12.3 % | 45 |
| E3 + mining (pool = training pool) | 0.00558 | 0.00641 | 0.00765 | +14.8 % | 111 |
| **E3b + mining (harsher pool)** | 0.00562 | **0.00626** | **0.00721** | +11.4 % | 82 |
| E4 full OPERA | 0.00559 | 0.00632 | 0.00741 | +13.1 % | (contaminated) |
| E4b full OPERA, harsher pool | 0.00590 | 0.00653 | 0.00748 | **+10.7 %** | 95 |

### Other settings (E0 vs E4)

| setting | E0 clean | E4 clean | E0 mean shifted | E4 mean shifted | E0 deg. | E4 deg. |
|---|---|---|---|---|---|---|
| Activity (MSE) | **0.00263** | 0.00286 | 0.04983 | **0.00358** | +1793 % | **+25 %** |
| USHCN 1-mo (MSE) | 0.5379 | 0.5211 | 0.5481 | 0.5287 | +1.9 % | +1.5 % |
| USHCN 12-mo (MSE) | 0.5202 | **0.5169** | 0.5392 | **0.5303** | +3.7 % | **+2.6 %** |
| P12 (AUROC) | **83.53** | 83.30 | - | - | see note | see note |
| P19 (AUROC) | 82.13 | **85.59** | - | - | +2.9 % | **+1.2 %** |
| PAM (accuracy) | **88.01** | 87.27 | - | - | +64 % | **+34 %** |

Findings:

1. **Augmentation is the robustness mechanism**, and it is decisive wherever the phenomenon is large. PhysioNet
   +107 % -> +14 % at no in-distribution cost, improving all 16 conditions; Activity +1793 % -> +25 % with a
   14x lower mean shifted error. It transfers to processes and even process *types* absent from training
   (PhysioNet stride 3: 0.0092 -> 0.0057; unseen degradation +129 % -> +18 %).
2. **Hard mining works, but only with a pool harsher than the training pool.** With the default pool (the same
   8 processes the model trains on) mining is indistinguishable from augmentation (+14.8 % vs +14.3 %) at 2.7x
   the cost, because "hardest of 8" is close to a random draw. Given severities and compositions absent from
   training it becomes the best single configuration on absolute shifted error and worst case (0.00626 /
   0.00721 vs 0.00632 / 0.00772). **This corrects the first pass, which reported mining as useless.**
3. **Distillation is an accuracy mechanism, not a robustness one.** On PhysioNet it is within noise of E1. On
   noisy USHCN it is the largest single effect (clean 0.5379 -> 0.5135, -4.5 %, best on 13 of 16 conditions).
   On P19 the combination lifts clean AUROC 82.13 -> 85.59 and AUPRC 16.89 -> 24.47 (+45 %).
4. **The mechanisms are complementary only where no single one saturates.** USHCN 1-mo and 12-mo: E4 is the
   best overall. PhysioNet: E4 = E1 within noise, and E4b trades clean error (0.00590, the worst of any config)
   for the lowest degradation ratio, so E3b is the configuration to deploy there.
5. **Benefit scales with baseline damage.** Across classification: PAM (+64 % degradation) gains 11.1 accuracy
   points at 50 % variable loss; P19 (+2.9 %) gains 5.6 AUROC at 30 % variable loss and improves clean too;
   P12 (+4.4 %) gains 2.6 AUROC at 50 % variable loss and loses 0.2 clean.
6. **Deployment cost is unchanged** by every mechanism: identical parameter count, inference 0.63 ms per series
   on PhysioNet for all configurations. Teacher and candidate pool are training-time only, as designed.

## Status against the plan's success criteria

| criterion | status |
|---|---|
| 1. improves robustness under shift | **met**: 6 of 7 settings; the 7th (USHCN 1-mo) has no shift to fix |
| 2. improves on unseen processes | **met** (PhysioNet +129 % -> +18 %; Activity +2057 % -> +37 %) |
| 3. works across >= 2 datasets | **met**: 5 datasets, both tasks |
| 4. >= 2 architectures | **met**: iTransformer (forecasting) + PatchTST (classification) |
| 5. each component measurably contributes | **partly**: augmentation everywhere; mining with an adequate pool; distillation only where the task is noisy |
| 6. deployment needs no privileged observation | **met** by construction, verified in the cost tables |
| 7. stable over 5 seeds | **not met**: 1 seed |

## Corrections to the first pass

- **Mining is not useless.** The earlier null came from the candidate pool defaulting to the training pool.
- **Activity is usable and is the strongest result**, not unusable. The pipeline already windows it into the
  published 5,400 samples via `Activity_time_chunk`; the earlier claim came from inspecting only the raw
  loader's 25 person-sequences.
- **USHCN's insensitivity is partly horizon-driven**, not purely a dataset property.
- **OPERA helps on all three classification datasets**, which the first pass had not tested.

## Known issues

- **P12 degradation is unreliable**: its cross-entropy *improves* under mild shifts while AUROC/AUPRC fall, so
  the shared CE column is misleading there. Read P12 from AUROC/AUPRC only. The shared "lower is better"
  aggregation should use 100 - AUROC for binary tasks in the summary metrics, not CE.
- **E4 PhysioNet timing is contaminated** (486 s/epoch): a concurrent GPU job made one epoch take 9,380 s
  instead of ~96 s. Metrics are unaffected; only the cost figure is wrong.
- **P12 runs at batch 16** (both E0 and E4, so the comparison is like-for-like): its attention peaks at 4.7 GB
  with one view, and two retained view graphs do not fit in 8 GB. `_P12_baseline_bs32_s0` is kept as the
  batch-32 phenomenon data point.
- USHCN is not min-max normalised by the repo pipeline, so its MSE is in raw units.

## Next experiments, in priority order

1. **5 seeds for E0, E1, E3b on PhysioNet and E0, E4 on Activity** - the two settings where the effect is large
   enough that seeds will settle the claim.
2. **Mining ablations** (plan section 28): pool size K in {4, 8, 16, 32}, alpha sweep, and per-sample rather
   than per-batch mining, now that the mechanism is known to work.
3. **Distillation diagnosis**: a teacher trained only on rich observations rather than reusing E0, the
   `distillation.weight` sweep, and the teacher-density ablation (1x / 2x / 4x).
4. **Fix the summary metric for binary classification** so degradation uses AUROC, then re-aggregate.
