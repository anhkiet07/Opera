# OPERA experiment summary (first pass, seed 0)

Date: 2026-09-24. Branch `opera`. Raw results in `results/raw/<run>/results.json`, aggregated tables in
`results/aggregated/tables_{physionet,ushcn}.md`, figures in `figures/`, W&B project `opera`.

**Scope and caveat.** Every number below is **one seed (0)**. The plan requires 5 seeds for any primary claim;
nothing here is a statistical result yet. What this pass establishes is the direction and size of each effect
and which mechanisms deserve the seeds.

## Protocol

| | PhysioNet | USHCN |
|---|---|---|
| records / samples | 12,000 stays | 1,114 stations -> 26,736 windows |
| variables | 41 clinical | 5 climate |
| history -> horizon | 24 h -> 24 h | 24 months -> 1 month |
| preprocessing | unmodified repo pipeline (t-PatchGNN protocol) | unmodified repo pipeline |
| encoder | query-based irregular embedding + iTransformer, 129k / 127k params | same |
| training | Adam 1e-3, batch 32 / 128, epoch cap 40, patience 10, min 15 |  |

Observation processes come from one API (`src/observation/`). Training draws from a mild pool
(random 0.1-0.4, stride 2, burst 0.2-0.3, variable 0.2, local mask 0.3); evaluation re-observes the **history
only** of the test inputs, targets untouched, with a fixed seed so every model sees the same shifted input.
Nine of the sixteen test conditions never occur in training (marked "unseen").

## Q1 - Does observation-process shift matter?

**On PhysioNet, yes and the effect is large. On USHCN, no.**

| test condition | PhysioNet E0 | USHCN E0 |
|---|---|---|
| clean | 0.0055 | 0.5379 |
| random dropout 70 % | 0.0152 (+177 %) | 0.5607 (+4 %) |
| variables 30 % removed | 0.0158 (+188 %) | 0.5475 (+2 %) |
| variables 50 % removed (unseen) | 0.0213 (+288 %) | 0.5693 (+6 %) |
| stride 3 (unseen) | 0.0092 (+68 %) | 0.5401 (+0.4 %) |
| **mean over 15 shifted conditions** | **+107 %** | **+1.9 %** |

The USHCN result is not a failure of the measurement: with half the observations removed at matched density the
model gets *better* (0.5355 vs 0.5379 clean). That task is dominated by marginal/seasonal structure and barely
uses its irregular history, so there is no observation-process sensitivity to repair. This bounds the claim
usefully: the phenomenon appears where the model genuinely depends on the irregular history.

**Density is not the whole story** (Table 3, PhysioNet E0): five processes that all remove ~50 % of the
observations give 0.0074 (density-matched), 0.0079 (stride 2), 0.0080 (burst) and 0.0213 (variable dropout) -
a 2.9x spread at equal density. *What* is removed matters more than *how much*.

## Q2-Q5 - Which mechanism contributes?

PhysioNet (mean over shifted conditions / relative degradation / clean):

| | clean | mean shifted | degradation | unseen degradation | s per epoch |
|---|---|---|---|---|---|
| E0 baseline | 0.0055 | 0.0114 | +107.0 % | +129.5 % | 24 |
| **E1 augmentation** | **0.0055** | **0.0063** | **+14.3 %** | **+17.9 %** | 41 |
| E2 + distillation | 0.0057 | 0.0064 | +12.3 % | +15.5 % | 45 |
| E3 + hard mining | 0.0056 | 0.0064 | +14.8 % | +18.0 % | 111 |
| E4 full OPERA | 0.0056 | 0.0063 | +13.1 % | +16.3 % | (contaminated) |

USHCN:

| | clean | mean shifted | degradation | s per epoch |
|---|---|---|---|---|
| E0 baseline | 0.5379 | 0.5481 | +1.9 % | 23 |
| E1 augmentation | 0.5329 | 0.5438 | +2.0 % | 30 |
| **E2 + distillation** | **0.5135** | 0.5266 | +2.6 % | 32 |
| E3 + hard mining | 0.5266 | 0.5475 | +4.0 % | 47 |
| **E4 full OPERA** | 0.5211 | **0.5287** | **+1.5 %** | 49 |

Reading:

1. **Augmentation is the robustness mechanism** and it is decisive where the phenomenon exists: PhysioNet
   degradation +107 % -> +14 %, at no in-distribution cost (0.00553 vs 0.00549), improving *every one* of the 16
   conditions. It transfers to processes absent from training (unseen degradation +129 % -> +18 %), including a
   regime *type* never trained on (stride subsampling: 0.0092 -> 0.0057).
2. **Distillation is an accuracy mechanism, not a robustness one.** On PhysioNet it is within noise of E1
   (+12.3 % vs +14.3 %, identical mean shifted MSE, slightly worse clean). On USHCN it is the single largest
   effect measured anywhere in this pass: clean 0.5379 -> 0.5135 (-4.5 %), best on every condition. The pattern
   is consistent with classic distillation: it helps when the student underfits a noisy target, not when the
   student already fits it.
3. **Hard mining has not earned its cost.** PhysioNet: +14.8 % vs +14.3 % for E1 at 2.7x the training time.
   USHCN: the *worst* degradation of the five configurations (+4.0 %), which is the expected failure mode -
   training preferentially on the hardest observations shifts the model away from the clean regime.
4. **The mechanisms are complementary only on USHCN.** E4 there is the best overall configuration (lowest
   degradation +1.5 %, lowest worst case 0.5458, near-best clean). On PhysioNet E4 = E1 within noise.
5. **Deployment cost is unchanged** by every mechanism: identical parameter count, and inference 0.63 ms per
   series for every PhysioNet configuration. Teacher and candidate pool are training-time only, as designed.

## Status against the plan's success criteria

| criterion | status |
|---|---|
| 1. improves robustness under shift | met on PhysioNet (augmentation); not applicable on USHCN (no shift to fix) |
| 2. improves on unseen processes | met on PhysioNet (+129 % -> +18 %) |
| 3. works across two datasets | partially: the *framework* runs on both, but the mechanism that helps differs by dataset |
| 4. two architectures | not started |
| 5. each component measurably contributes | **not met**: mining contributes nothing; distillation only on USHCN |
| 6. deployment needs no privileged observation | met by construction and verified in cost table |
| 7. stable over 5 seeds | **not met**: 1 seed |

## Next experiments, in priority order

1. **Seeds 1-4 for E0/E1 on PhysioNet** - the one claim strong enough to be worth 5 seeds now.
2. **Diagnose distillation** (plan section 39): measure the logged teacher advantage directly; test a teacher
   trained *only* on rich observations vs the E0 teacher; sweep `distillation.weight`; test the sparse-student
   regime where the teacher's advantage is largest.
3. **Diagnose mining**: the candidate pool may be too weak (its hardest member is barely harder than a random
   draw). Test a harder pool, larger K, and the alpha sweep of section 15.
4. **Decide on Activity**: unusable as implemented (see below).

## Known issues

- **Activity dataset is unusable without new preprocessing.** Each record spans ~192 s but the protocol window
  is 4 s, so only ~116 of 5,829 observations per record fall in the window and the dataset collapses to 25
  samples. The published 5,400 samples require a windowing step this repo's loader does not implement. Not
  added, because silently changing preprocessing is against the plan; needs an explicit decision.
- **E4 PhysioNet cost number is contaminated** (486 s per epoch): a second GPU job ran concurrently and one
  epoch took 9,380 s instead of ~96 s. Its *metrics* are unaffected (same data, same protocol); only the timing
  is wrong. Re-time it when the card is free.
- USHCN is not min-max normalised by the repo pipeline, so its MSE is in raw units and not comparable to
  PhysioNet's.
