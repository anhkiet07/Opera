# Handoff: scale the OPERA benchmark to 5 seeds on the cluster

You are picking this up on a SLURM cluster with dedicated GPUs. Seed 0 is **done** on an 8 GB laptop card and
is committed on branch `opera`. Your job is seeds 1--4. Read `BENCHMARK_PROTOCOL.md` for the full grid
definition; this file is only what you need to get running and the traps that already cost us time.

## 1. What is done and what you are running

Seed 0 is complete: 7 settings x 3 backbones x {vanilla, opera} = 42 cells, no failures. `opera++` is running
locally on 6 settings (the mortality task excluded, see §6).

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

## 4. Work in scratch, not in the repo

Make one dedicated directory on scratch and send **everything** there: run outputs, checkpoints, SLURM logs,
CSVs. Nothing this sweep produces should land inside the git checkout.

```bash
export OPERA_RUN=$SCRATCH/opera_bench          # or /scratch/$USER/opera_bench if $SCRATCH is unset
mkdir -p "$OPERA_RUN"/{raw,logs,csv}
echo "$OPERA_RUN"                              # note this path, every command below needs it
```

Every path is a flag, so no code changes are needed:

| what | flag | value |
|---|---|---|
| run dirs + checkpoints | `--out` | `$OPERA_RUN/raw` |
| CSV inputs | `--raw` | `$OPERA_RUN/raw` |
| CSV outputs | `--out --summary --curated` | `$OPERA_RUN/csv/...` |
| SLURM stdout/stderr | `#SBATCH --output` | `$OPERA_RUN/logs/%x-%j.out` |
| wandb, if you enable it | `WANDB_DIR` | `$OPERA_RUN` |

**Pass the same `--out` to every stage.** The teacher path is derived from it (`{out}/bench_{setting}_{backbone}_vanilla_s{seed}/model.pt`),
so stage D only finds stage B's checkpoints if both used one root. Mixing roots looks like a missing teacher.

**Scratch is usually purged.** Before the purge window, copy back the small artefacts only:
`$OPERA_RUN/raw/*/results.json`, `$OPERA_RUN/raw/*/config.yaml` and `$OPERA_RUN/csv/*`. Leave `model.pt`
behind -- checkpoints are ~25 MB each and the repo is already carrying too many (§8). But do not delete them
while any stage D or F run is still pending, because those read them.

## 5. Launch: submit SLURM jobs, one cell per task

**Do not run the sweep on the login node.** Write `sbatch` scripts. Every cell is independent given its
teacher, so the grid is a job array: `--settings`, `--backbones` and `--seeds` each take a subset, and one array
task should be exactly one `(setting, backbone, seed)` cell.

Save as `$OPERA_RUN/stage.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=opera
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
set -euo pipefail

SETTINGS=(physionet activity ushcn_1mo ushcn_12mo P12 P19 PAM)   # 7
BACKBONES=(itransformer patchtst tmix)                            # 3
SEEDS=(1 2 3 4)                                                   # 4   -> 84 tasks

i=$SLURM_ARRAY_TASK_ID
s=$(( i % 7 )); i=$(( i / 7 ))
b=$(( i % 3 )); i=$(( i / 3 ))
d=$(( i % 4 ))

cd "$REPO"
source activate opera            # or: conda activate opera
python scripts/run_benchmark.py --stage "$STAGE" \
  --settings "${SETTINGS[$s]}" --backbones "${BACKBONES[$b]}" --seeds "${SEEDS[$d]}" \
  --out "$OPERA_RUN/raw" --no_wandb
```

Submit stage B (vanilla), then stage D once B has finished:

```bash
export REPO=$PWD
sbatch --array=0-83 --export=ALL,STAGE=B,REPO="$REPO",OPERA_RUN="$OPERA_RUN" \
       --output="$OPERA_RUN/logs/B-%A_%a.out" "$OPERA_RUN/stage.sbatch"

# after B is complete (see below), same array for D:
sbatch --array=0-83 --export=ALL,STAGE=D,REPO="$REPO",OPERA_RUN="$OPERA_RUN" \
       --output="$OPERA_RUN/logs/D-%A_%a.out" "$OPERA_RUN/stage.sbatch"
```

`#SBATCH` directives do **not** expand shell variables, which is why `--output` and `--export` are passed on
the command line rather than written into the script.

**Stage D must not start until B is done for that cell.** Either wait for B to drain, or chain the arrays:

```bash
JB=$(sbatch --parsable --array=0-83 --export=ALL,STAGE=B,... "$OPERA_RUN/stage.sbatch")
sbatch --dependency=afterok:$JB --array=0-83 --export=ALL,STAGE=D,... "$OPERA_RUN/stage.sbatch"
```

Per-task resources: `--gres=gpu:1` always, never two tasks on one GPU. 32 GB host memory and 4 CPUs is ample;
the loaders are single-process. `--time=08:00:00` is generous for every cell except P12 + iTransformer (§6).

Node guidance from your `sinfo`: `l40s-f-[01-04]` (48 GB) for the P12 cells, `v100l-f-*` (32 GB) next,
`v100-f-*` (16 GB) for everything else. `rtxp6000l-f-01` is down and `v100-f-07` is draining. To pin the heavy
cells, submit them as a separate small array with `--constraint=l40s` (or `--nodelist=l40s-f-[01-04]`, whichever
the site supports) and exclude P12 from the main array by dropping it from `SETTINGS` and using 6 settings, 72
tasks.

Finished cells are skipped, so re-submitting an array after failures costs nothing and is the normal way to
mop up. Check what is still missing with:

```bash
ls -d "$OPERA_RUN"/raw/bench_*/ | wc -l                                  # dirs created
ls "$OPERA_RUN"/raw/bench_*/results.json 2>/dev/null | wc -l             # cells actually finished
```

A directory without `results.json` is an incomplete run and will be redone.

The one-at-a-time loop and the `wait_for_free_gpu()` gate inside `scripts/run_benchmark.py` are harmless when
each task owns its GPU: with one cell per task the loop runs once and the gate sees a free card. They exist only
because two jobs on one 8 GB card thrash -- an epoch once went from 96 s to 9380 s with no error.

## 6. Traps that already cost us time

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

## 7. Results plumbing

```bash
python scripts/benchmark_csv.py \
  --raw     "$OPERA_RUN/raw" \
  --out     "$OPERA_RUN/csv/benchmark.csv" \
  --summary "$OPERA_RUN/csv/benchmark_summary.csv" \
  --curated "$OPERA_RUN/csv/benchmark_curated.csv"
```

Writes three files, all regenerated from the run directories so they cannot drift:

* `benchmark.csv` -- one row per (setting, backbone, method, seed, condition), every metric
* `benchmark_summary.csv` -- mean/std over seeds, degradation derived
* `benchmark_curated.csv` -- the main table: clean + the four shifts, one row per seed

Method labels (`vanilla`, `opera`, `opera++`, `abl_*`) are derived from what each config switches on, not from
filenames, so they cannot be mislabelled. `run_group` is `bench` for these runs; anything `explore` is from
earlier exploratory work and is not comparable cell-by-cell.

Degradation for classification is computed on `1 - AUROC` (or `1 - accuracy`), never cross-entropy: the two move
in opposite directions on the mortality task.

## 8. Please do not

* Commit checkpoints if you can avoid it. `results/raw/**/model.pt` is already ~291 MB and 185 MB of git
  history. Keep `results.json`; that is what the CSVs are built from. But note stage D **needs** stage B's
  `model.pt` on disk, so do not delete them mid-sweep.
* Edit `scripts/run_benchmark.py` while a sweep is running. Doing that once produced a `NameError` on the next
  chained call.
* Reformat or restructure the paper in `paper/`. It is mid-revision on this side.

## 9. When you are done

Report per-cell: clean, mean over the four shifts, and `delta_4shifts_pct`, with mean ± std over the five seeds.
Then say which of the seed-0 conclusions survive. The ones to check are:

1. opera improves mean shifted error (20 of 21 cells at seed 0)
2. opera usually costs a little clean accuracy (it won clean in only 9 of 21)
3. the gain scales with how damaged the baseline was
4. the two TMix cells above are artefacts, not results
