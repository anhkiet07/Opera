# Observation-process shift benchmark protocol

The benchmark answers one question:

> Does a model trained under one observation process stay reliable when deployment differs along
> **observation density**, **observation timing**, or **variable availability** — and does robustness transfer to a
> combination of those never seen in training?

Fixed grid. Nothing outside it belongs in the main results.

```
3 backbones  x  7 settings  x  5 conditions  x  5 seeds  =  525 rows per method arm
```

One CSV holds every row. Degradation is derived from it, never stored by a training run.

---

## 1. Backbones (3)

All three consume irregular input through the same query-based embedding, so the only thing that changes is how
the encoder mixes information. Selected with `--set model.backbone=<name>`.

| name | `backbone` | mixes over | family |
|---|---|---|---|
| iTransformer | `itransformer` | variables | variate |
| PatchTST | `patchtst` | time patches | patch |
| TMix | `tmix` | time patches, MLP | patch |

**Naming caution.** `tmix` in this repo is a residual temporal MLP block, not the published TimeMixer with
multi-scale decomposition. Report it as a *TimeMixer-style temporal mixer*, never as TimeMixer.

Available but out of scope: `patchmixer`, `timexer`. Unavailable: `s_mamba`, because `mamba_ssm` is not installed.

Why these three: they span variate attention, patch attention and patch MLP mixing. Crossing all three with both
tasks also removes the current confound in which forecasting used one architecture and classification the other.

---

## 2. Conditions (1 clean + 4 shifts)

Each shift moves **one** dimension of the observation process past what training covers, leaving the others at the
dataset's own setting. The composed condition moves two at once and is the generalisation test.

| column | dimension | training pool has | evaluation spec | question |
|---|---|---|---|---|
| `clean` | none | — | `identity` | in-distribution reference |
| `density` | how much | dropout ≤ 0.4 | `random_dropout:0.7` | substantially sparser record |
| `timing` | when | stride 2 | `temporal_stride:3` | different temporal resolution |
| `variables` | which | variable dropout 0.2 | `variable_dropout:0.5` | sensors become unavailable |
| `composed` | two at once | neither | `temporal_stride:3+variable_group_dropout:0.3` | unseen combination |

Variable-group removal appears in **no** form during training, so the composed condition is unseen in family, not
just in severity.

All four specs already exist in `TEST_CONDITIONS`. No new observation-process code is required. Pass them with:

```
--conditions identity,random_dropout:0.7,temporal_stride:3,variable_dropout:0.5,temporal_stride:3+variable_group_dropout:0.3
```

**Training pool is unchanged** and stays deliberately mild: random dropout `{0.1, 0.25, 0.4}`, stride 2, burst
`{0.2, 0.3}`, variable dropout 0.2, local mask 0.3.

**The complete 15-condition sweep is retained** for two secondary analyses only: the density-matched comparison and
the seen-versus-unseen split. It does not appear in the main table.

---

## 3. Settings (7)

| `setting` | task | primary metric | vars | batch |
|---|---|---|---|---|
| `physionet` | forecast 24h→24h | MSE | 41 | 32 |
| `activity` | forecast 3000→1000 ms | MSE | 12 | 32 |
| `ushcn_1mo` | forecast 24→1 mo | MSE | 5 | 128 |
| `ushcn_12mo` | forecast 24→12 mo | MSE | 5 | 128 |
| `p12` | mortality, 2 class | AUROC | 36 | 16 |
| `p19` | sepsis, 2 class | AUROC | 34 | 32 |
| `pam` | activity, 8 class | accuracy | 17 | 32 |

Preprocessing and splits are the unmodified published pipelines: t-PatchGNN for forecasting, Raindrop for
classification including its minority-class upsampling.

**Batch size is a property of the setting, not the arm.** Every backbone, method and seed for a setting uses the
same batch size, so no comparison is confounded by it. If any backbone cannot fit a setting's batch size, that
setting's batch size drops for *all* arms and the sweep restarts for that setting.

---

## 4. Metrics

Every row carries every metric that applies, so no pivot needs a second file.

| column | forecasting | classification |
|---|---|---|
| `mse`, `mae` | yes | — |
| `ce`, `acc` | — | yes |
| `auroc`, `auprc` | — | binary only |
| `f1` | — | multi-class only |
| `primary_metric` | `mse` | `auroc` or `acc` |
| `primary_value` | the MSE | the AUROC or accuracy |
| `primary_error` | the MSE | `1 - primary_value/100` |

`primary_error` exists so that relative degradation is a one-line pivot and always lower-is-better:

```
degradation = (primary_error[condition] - primary_error[clean]) / primary_error[clean]
```

**Classification degradation is computed on `1 - AUROC` or `1 - accuracy`, never on cross-entropy.** Cross-entropy
moves in the opposite direction to AUROC on the mortality task, which previously made that setting look anomalous.
This requires a fix in `evaluate_conditions`, which currently derives its summary from the cross-entropy carried in
the `mse` key.

---

## 5. The CSV

Single file, long format, one row per (setting, backbone, method, seed, condition):
`results/benchmark.csv`.

```
setting, task, n_vars, backbone, method, seed, batch_size,
condition, condition_spec, dimension, seen_in_training,
primary_metric, primary_value, primary_error,
mse, mae, ce, acc, auroc, auprc, f1,
n_series, infer_ms_per_series, params, train_time_per_epoch_s, epochs_run, peak_gpu_mem_mb,
run_dir, git_commit, config_sha
```

Rules:

- Regenerated in full from `results/raw/*/results.json` on every aggregation, so it cannot drift from its source.
- No derived statistics stored. No mean, no standard deviation, no degradation. Those are computed from this file.
- `method` is `vanilla` or `opera`, so the mechanism arms drop into the same file with no schema change.
- Blank, not zero, where a metric does not apply.
- Row count is checkable: `3 x 7 x 5 x 5 = 525` per method arm.

A second derived file, `results/benchmark_summary.csv`, holds mean and standard deviation over seeds. It is
generated, never edited.

---

## 6. Fixed protocol details

- **Re-observation seed is 12345**, independent of the run seed, so every backbone, method and seed is scored on
  byte-identical shifted inputs. Comparisons are paired at the level of individual test inputs.
- **Only the history is re-observed.** Forecast targets, prediction timestamps and labels are never touched.
- **Model selection is on the clean validation condition only.** No shifted data influences which checkpoint is
  kept, so gains cannot come from selection.
- **Seeds 0, 1, 2, 3, 4**, paired across backbones and methods.
- Hidden width 64, 4 heads, 3 layers, Adam at 1e-3, epoch cap 40, patience 10, minimum 15 epochs.
- **One job at a time.** Two processes on the 8 GB card thrash and have silently cost hours.
- Every run stores config, git commit, seed, checkpoint, logs and metrics next to its results.

---

## 7. Run count and staging

**The method needs no extra runs.** Its frozen rich-view teacher *is* the vanilla baseline: the baseline arm trains
on the dataset's own full observation, which is exactly the rich view, and the launcher points a distillation run at
the same-seed baseline checkpoint. Stage A's checkpoints therefore serve Stage C directly.
`configs/teacher_rich.yaml` duplicates `baseline.yaml` and is not needed; delete it.

| stage | arm | runs | estimate | leaves you with |
|---|---|---|---|---|
| A | vanilla, seed 0 | 21 | 3–4 h | the full main table, one seed |
| C | opera, seed 0 | 21 | 7–10 h | method works on 3 backbones |
| B | vanilla, seeds 1–4 | 84 | 10–16 h | the main table with error bars |
| D | opera, seeds 1–4 | 84 | 28–45 h | the complete benchmark |

Totals: **210 runs**, roughly 50–75 h of card time. The method costs 2–2.5x per run, because it trains on two views
and runs a teacher forward pass, but it does not cost more runs. Each stage ends at a complete, reportable result,
so the sweep can stop after any stage. Stage C depends on Stage A; Stage D depends on Stage B.

Recommended order is A, C, B, D: getting the method onto all three backbones at one seed is more informative than
error bars on the baselines alone.

---

## 8. Known risks

- **The teacher is resolved without the backbone.** `scripts/run_experiments.py` builds the teacher path from
  dataset and seed only (`{dataset}_baseline_s{seed}`), so all three backbones would request the same checkpoint and
  two would receive a teacher of the wrong architecture. `load_state_dict` is strict, so this raises rather than
  loading silently, but it blocks two thirds of Stage C. **Run names and teacher resolution must both carry the
  backbone** before Stage C: `bench_{setting}_{backbone}_{method}_s{seed}`.
- **A missing teacher path crashes at load rather than at launch.** The launcher warns and continues, passing a
  non-existent path, which fails later in `torch.load`. Loud, but it wastes the queue position. Make the launcher
  abort on a missing teacher instead.
- **`build_teacher` falls back to a copy of the student's initialisation** when `teacher_ckpt` is empty, which would
  make the distillation term pure noise. The launcher never passes an empty path, so this is latent, but it should
  raise instead of falling back.

- **iTransformer on the mortality task** previously exhausted memory, surfacing as a misleading CUDA driver error.
  That is why classification moved to PatchTST. Expect to lower that setting's batch size for all arms.
- **Patch-family models on the 5-variable climate data** need a sane patch count; patch size and stride have not
  been checked there.
- **Collapse to a constant predictor** has happened on sparse settings. The 15-epoch minimum guards it, but runs
  need checking rather than trusting.
- **The `local_mask:0.5` condition is absent from every stored result**, which is why the density-matched table has
  an empty column. It should be added to the secondary sweep.

---

## 9. Commands

One cell:

```bash
python scripts/train.py --config configs/baseline.yaml --seed 0 \
  --name bench_physionet_itransformer_vanilla_s0 \
  --set model.backbone=itransformer data.name=physionet \
  --conditions identity,random_dropout:0.7,temporal_stride:3,variable_dropout:0.5,temporal_stride:3+variable_group_dropout:0.3
```

The whole grid, sequentially, with the CSV rebuilt at the end:

```bash
python scripts/run_benchmark.py --stage A        # not yet written
python scripts/aggregate.py --csv results/benchmark.csv
```

---

## 10. What has to be built

1. `MAIN_CONDITIONS` in `src/observation/generator.py`, the five specs above.
0. Backbone-aware run names and teacher resolution, `bench_{setting}_{backbone}_{method}_s{seed}`, and abort
   rather than warn when a teacher checkpoint is missing. Required before Stage C, see section 8.
2. Degradation in `evaluate_conditions` computed on the primary metric, not cross-entropy.
3. `scripts/run_benchmark.py`: the staged grid launcher, sequential, resumes by skipping finished run directories.
4. `scripts/aggregate.py`: emit `results/benchmark.csv` and `results/benchmark_summary.csv` in the schema above.
5. A smoke stage: one epoch per backbone per task to catch memory and patch-count failures before 21 real runs.
