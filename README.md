# OPERA — Observation-Process Robust Representation Learning for Irregular Time Series

Irregularly sampled time series are the product of two processes: the underlying temporal dynamics `X`, and the
**observation process** `S` that decides when and which variables are recorded.

```
X  --S-->  O  --f_theta-->  Y
```

Standard evaluation draws train and test data from the same `P(S)`. OPERA studies what happens when `P(S)`
changes while `P(X)` stays comparable — **observation-process shift** — and which training mechanisms make
representations robust to it.

## Mechanisms

| mechanism | idea | where it lives |
|---|---|---|
| **Observation-process augmentation** | train on several observation views `S_k(X)` of each trajectory | `src/observation/`, `configs/augmentation.yaml` |
| **Privileged-observation distillation** | a frozen teacher sees a richer observation of the *same history*; the student imitates its predictions | `src/distillation/`, `configs/distillation.yaml` |
| **Hard observation-process mining** | search a constrained candidate pool for the observation processes the model handles worst | `src/mining/`, `configs/hard_mining.yaml` |

No representation-consistency loss is used: the objective is `L_task + lambda_d * L_distill`, and the goal is
that different observation processes yield representations that stay *useful*, not identical.
At inference only the student and a sparse observation are needed.

## Layout

```
src/observation/    observation processes, metadata, descriptor r = g(S), validity constraints, generator
src/models/         BaseIrregularEncoder + the query-based irregular encoder
src/distillation/   frozen and EMA teachers, prediction/representation distillation losses
src/mining/         candidate-pool hard miner with constraints and an alpha curriculum
src/training/       trainer and the OPERA objective
src/evaluation/     masked metrics and the observation-shift evaluation suite
src/datasets/       dataset registry (physionet, ushcn, activity)
configs/            E0-E4 for each dataset, inheriting from base.yaml
scripts/            train.py, run_experiments.py, aggregate.py, generate_figures.py, prepare_data.py
tests/              46 tests: process semantics, metadata, constraints, anti-leakage, mining
archive/            the previous research direction (consistency / adaptive / causal), frozen
```

## Reproducing

```bash
python -m pytest tests -q                                   # 46 tests
python scripts/prepare_data.py --dataset ushcn              # build a dataset cache (CPU only)
python scripts/train.py --config configs/opera.yaml --seed 0
python scripts/run_experiments.py --stage all --seeds 0 1 2 3 4
python scripts/aggregate.py --dataset physionet             # -> results/aggregated/tables_physionet.md
python scripts/generate_figures.py --dataset physionet      # -> figures/
```

Each run saves its config, git commit, seed, checkpoint, log and metrics under `results/raw/<run>/`.
**Never run two GPU jobs at once on a small card**: contention here inflated one epoch from 96 s to 9,380 s.

## Experiments

| id | config | mechanisms |
|---|---|---|
| E0 | `baseline.yaml` | none (single observation regime) |
| E1 | `augmentation.yaml` | augmentation |
| E2 | `distillation.yaml` | augmentation + privileged distillation |
| E3 | `hard_mining.yaml` | augmentation + hard mining |
| E4 | `opera.yaml` | all three |

Prefix a config with a dataset (`ushcn_opera.yaml`) to run it elsewhere. E2/E4 use the same-seed E0 checkpoint
of the same dataset as their frozen rich-view teacher.

## Findings so far (seed 0 — see `results/EXPERIMENT_SUMMARY.md`)

- Observation-process shift is large on PhysioNet (**+107 %** mean degradation) and absent on USHCN (+1.9 %).
- It is not explained by density: at equal 50 % density, degradation ranges from +34 % to +288 % depending on
  *which* observations are removed.
- Augmentation removes most of the degradation on PhysioNet (**+107 % -> +14 %**) at no in-distribution cost,
  and transfers to observation processes never seen in training.
- Distillation helps accuracy where the task is noisy (USHCN clean **-4.5 %**), not robustness.
- Hard mining has not justified its cost on either dataset.

All numbers are single-seed and provisional; the plan's protocol requires 5 seeds before any claim is final.
