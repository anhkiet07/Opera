# Handoff: scale the OPERA benchmark to 5 seeds on the cluster

You are picking this up on a SLURM cluster with dedicated GPUs. Seed 0 is **done** on an 8 GB laptop card and
is committed on branch `opera`. Your job is seeds 1--4. Read `BENCHMARK_PROTOCOL.md` for the full grid
definition; this file is only what you need to get running and the traps that already cost us time.

## 1. What is done and what you are running

Seed 0 is complete: 7 settings x 3 backbones x {vanilla, opera} = 42 cells, no failures. `opera++` is running
locally on 6 settings (the mortality task excluded, see §5).

| stage | arm | seeds | cells | needs |
|---|---|---|---|---|
| B | vanilla | 1,2,3,4 | 84 | nothing |
| D | opera | 1,2,3,4 | 84 | **stage B at the same seed** |
| F | opera++ | 1,2,3,4 | 84 | stage B at the same seed |

**Run B before D.** The distillation teacher is the *vanilla checkpoint of the same cell* -- same setting, same
backbone, same seed. A missing teacher aborts that cell with a clear message. Do not try to share one teacher
across backbones; the state dict will not load.

Priority: **B then D**. Skip F unless asked -- mining gained ~0.00006 MSE at seed 0 and may not be worth the
compute.

## 2. Build the environment

```bash
module load cuda                      # or whatever the site uses
conda create -n opera python=3.12 -y
conda activate opera
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Reference: python 3.12.10, torch 2.14.0+cu126. `mamba-ssm` in requirements.txt is commented out and stays that
way; the S-Mamba backbone is not part of the benchmark.

The launcher looks for `.venv/Scripts/python.exe` then `.venv/bin/python`, then falls back to `sys.executable`.
If you use conda rather than a `.venv`, it takes the conda python via the fallback -- fine, but make sure the env
is **activated** when you launch, or it will use the system python and fail on `import yaml`.

## 3. Get the data (it is NOT in git)

Only the loader code is tracked. Fetch and preprocess each dataset first:

```bash
python scripts/prepare_data.py --dataset physionet
python scripts/prepare_data.py --dataset activity
python scripts/prepare_data.py --dataset ushcn
```

P12 / P19 / PAM use the Raindrop pipeline and expect their files under `data/P12`, `data/P19`, `data/PAM`.

**Do not copy the processed `.pt` files from the Windows machine.** They were serialised with CUDA tensors and
fail to load unless a GPU is visible with the same ordinal. Regenerate them.

## 4. Launch

Sequential (what we ran locally):

```bash
python scripts/run_benchmark.py --stage B --no_wandb
python scripts/run_benchmark.py --stage D --no_wandb
```

Finished cells are skipped, so a stage is always safe to re-run after an interruption.

**You should parallelise instead.** Every cell is independent given its teacher. The one-at-a-time loop and the
`wait_for_free_gpu()` gate in `scripts/run_benchmark.py` exist only because two jobs on one 8 GB card thrash --
an epoch once went from 96 s to 9380 s with no error. On dedicated GPUs, drop both and submit per-cell jobs.
Use `--settings` and `--backbones` to slice the grid, e.g.

```bash
python scripts/run_benchmark.py --stage B --settings physionet --backbones itransformer --no_wandb
```

Still: **never two jobs on one GPU.** One job per GPU.

Node guidance from `sinfo`: `l40s-f-[01-04]` (48 GB) for anything memory-hungry, `v100l-f-*` (32 GB) next,
`v100-f-*` (16 GB) for everything else. `rtxp6000l-f-01` is down and `v100-f-07` is draining.

## 5. Traps that already cost us time

**Do not change any batch size.** Seed 0 was run at the sizes in `BENCHMARK_PROTOCOL.md` §3, and seeds must be
comparable. In particular **P12 stays at 8** even though your GPUs have room. It is 8 because iTransformer's
variate attention exhausts 8 GB there -- but raising it now would make seeds 1--4 incomparable with seed 0 and
require rerunning everything for that setting. If you want P12 at 32, that is a separate decision and it means
redoing seed 0 too.

**P12 + iTransformer is pathologically slow at batch 8.** It took 213 min locally against 64 min for the same
setting under PatchTST, because 8 GB was 97 % full and the driver spilled to host memory. On a 48 GB L40S this
should disappear. Put that cell on an `l40s` node and it is ordinary.

**The encoder reads values, not the mask.** If you write any code that intervenes on observations, you must
replace the masked *values* as well as the mask. Passing only a new mask silently changes nothing -- it produced
five identical panels in a figure before we caught it. `gen.condition()` returns both; pass both.

**Two fast failures abort the stage.** If two cells fail in under 30 s the launcher stops, on the assumption the
environment is broken rather than the configs. That guard exists because a machine restart once burned six queue
positions in seconds. If you hit it, fix the environment; do not loop.

**Sepsis + TMix is at chance in both arms** (49.10 / 53.18 AUROC at seed 0). Treat it as a failed training run,
not a result. If seeds 1--4 reproduce it, say so -- it is an architecture failure worth reporting, not a bug to
hide.

## 6. Results plumbing

```bash
python scripts/benchmark_csv.py
```

Writes three files, all regenerated from `results/raw/` so they cannot drift:

* `results/benchmark.csv` -- one row per (setting, backbone, method, seed, condition), every metric
* `results/benchmark_summary.csv` -- mean/std over seeds, degradation derived
* `results/benchmark_curated.csv` -- the main table: clean + the four shifts, one row per seed

Method labels (`vanilla`, `opera`, `opera++`, `abl_*`) are derived from what each config switches on, not from
filenames, so they cannot be mislabelled. `run_group` is `bench` for these runs; anything `explore` is from
earlier exploratory work and is not comparable cell-by-cell.

Degradation for classification is computed on `1 - AUROC` (or `1 - accuracy`), never cross-entropy: the two move
in opposite directions on the mortality task.

## 7. Please do not

* Commit checkpoints if you can avoid it. `results/raw/**/model.pt` is already ~291 MB and 185 MB of git
  history. Keep `results.json`; that is what the CSVs are built from. But note stage D **needs** stage B's
  `model.pt` on disk, so do not delete them mid-sweep.
* Edit `scripts/run_benchmark.py` while a sweep is running. Doing that once produced a `NameError` on the next
  chained call.
* Reformat or restructure the paper in `paper/`. It is mid-revision on this side.

## 8. When you are done

Report per-cell: clean, mean over the four shifts, and `delta_4shifts_pct`, with mean ± std over the five seeds.
Then say which of the seed-0 conclusions survive. The ones to check are:

1. opera improves mean shifted error (20 of 21 cells at seed 0)
2. opera usually costs a little clean accuracy (it won clean in only 9 of 21)
3. the gain scales with how damaged the baseline was
4. the two TMix cells above are artefacts, not results
