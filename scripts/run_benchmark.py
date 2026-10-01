"""Staged benchmark launcher (BENCHMARK_PROTOCOL.md section 7).

    python scripts/run_benchmark.py --stage smoke     # 1 epoch per backbone per task, catches OOM early
    python scripts/run_benchmark.py --stage A         # vanilla, seed 0,   21 runs
    python scripts/run_benchmark.py --stage C         # opera = aug + distillation, seed 0, 21 runs (needs A)
    python scripts/run_benchmark.py --stage B         # vanilla, seeds 1-4, 84 runs
    python scripts/run_benchmark.py --stage D         # opera,   seeds 1-4, 84 runs (needs B)
    python scripts/run_benchmark.py --stage E         # opera++ = opera + harsh-pool mining, seed 0 (needs A)

Run names carry the backbone, `bench_{setting}_{backbone}_{method}_s{seed}`, because the frozen rich-view
teacher must share the student's architecture: the teacher is the same-cell vanilla checkpoint, and a
cross-architecture load would fail on the state dict. A missing teacher aborts the run rather than warning and
queueing a job that will die in `torch.load`.

Runs execute strictly one at a time: two processes on the 8 GB card thrash and have silently cost hours.
Finished runs are skipped, so a stage can be resumed after an interruption.
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _python():
    """The project venv interpreter. `sys.executable` is the system Python when this script is launched from a
    shell that has not activated the venv, and the deps (yaml, torch) live only in the venv."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in (".venv/Scripts/python.exe", ".venv/bin/python"):
        cand = os.path.join(root, rel)
        if os.path.exists(cand):
            return cand
    return sys.executable


PYTHON = _python()

BACKBONES = ["itransformer", "patchtst", "tmix"]

# setting -> {arm: config}. Each config already carries its dataset's history/patch/batch.
#
#   vanilla  no mechanism
#   opera    augmentation + privileged distillation                        <- headline arm
#   opera++  augmentation + privileged distillation + mining, harsh pool   <- headline arm, later stage
#
# Mining over the training mixture is an ablation, not an arm: the hardest of K draws from the pool the model
# already trains on is barely harder than one draw.
SETTINGS = {
    "physionet":  {"vanilla": "configs/baseline.yaml",
                   "opera":   "configs/distillation.yaml",
                 "operapp": "configs/opera_strong.yaml",
                   "operapp2":        "configs/physionet_operapp2.yaml",
                   "operapp2_scalar": "configs/physionet_operapp2_scalar.yaml"},
    "activity":   {"vanilla": "configs/activity_baseline.yaml",
                   "opera":   "configs/activity_distillation.yaml",
                "operapp": "configs/activity_operapp.yaml",
                   "operapp2":        "configs/activity_operapp2.yaml",
                   "operapp2_scalar": "configs/activity_operapp2_scalar.yaml"},
    "ushcn_1mo":  {"vanilla": "configs/ushcn_baseline.yaml",
                   "opera":   "configs/ushcn_distillation.yaml",
                 "operapp": "configs/ushcn_operapp.yaml",
                   "operapp2":        "configs/ushcn_1mo_operapp2.yaml",
                   "operapp2_scalar": "configs/ushcn_1mo_operapp2_scalar.yaml"},
    "ushcn_12mo": {"vanilla": "configs/ushcn_long_baseline.yaml",
                   "opera":   "configs/ushcn_long_distillation.yaml",
                  "operapp": "configs/ushcn_long_operapp.yaml",
                   "operapp2":        "configs/ushcn_12mo_operapp2.yaml",
                   "operapp2_scalar": "configs/ushcn_12mo_operapp2_scalar.yaml"},
    "P12":        {"vanilla": "configs/P12_baseline.yaml",
                   "opera":   "configs/P12_distillation.yaml",
           "operapp": "configs/P12_operapp.yaml"},
    "P19":        {"vanilla": "configs/P19_baseline.yaml",
                   "opera":   "configs/P19_distillation.yaml",
           "operapp": "configs/P19_operapp.yaml"},
    "PAM":        {"vanilla": "configs/PAM_baseline.yaml",
                   "opera":   "configs/PAM_distillation.yaml",
           "operapp": "configs/PAM_operapp.yaml"},

    # --- forecast-horizon axis -------------------------------------------------------------------
    # One entry per (dataset, horizon), exactly as ushcn_1mo / ushcn_12mo already do, because run
    # names do not encode the horizon: overriding data.history with --set on an existing setting
    # would reuse that setting's run name and either skip the cell as finished or overwrite a good
    # result. The horizon is derived per dataset in data/parse.py -- physionet: 48 - history,
    # activity: 4000 - history, ushcn: data.pred_window -- and npatch follows from history in
    # src/datasets/registry.py. patch_size and stride are held at each dataset's existing values so
    # the horizon is the only axis that moves. Classification settings have no horizon.
    # No operapp arm here: this axis runs vanilla and opera only.
    "physionet_h12":  {"vanilla": "configs/physionet_h12_baseline.yaml",      # 12 h -> 36 h
                       "opera":   "configs/physionet_h12_distillation.yaml"},
    "physionet_h36":  {"vanilla": "configs/physionet_h36_baseline.yaml",      # 36 h -> 12 h
                       "opera":   "configs/physionet_h36_distillation.yaml"},
    "activity_h2000": {"vanilla": "configs/activity_h2000_baseline.yaml",     # 2000 ms -> 2000 ms
                       "opera":   "configs/activity_h2000_distillation.yaml"},
    "activity_h1000": {"vanilla": "configs/activity_h1000_baseline.yaml",     # 1000 ms -> 3000 ms
                       "opera":   "configs/activity_h1000_distillation.yaml"},
    "ushcn_6mo":      {"vanilla": "configs/ushcn_6mo_baseline.yaml",          # 24 mo -> 6 mo
                       "opera":   "configs/ushcn_6mo_distillation.yaml"},

    # --- native distribution shift (no synthetic corruption in the split itself) -------------------
    # Sepsis 2019 across hospital systems: train + validate on A, test on the unseen B. Same records and
    # preprocessing as P19, so the contrast against P19 isolates the shift. The 5 observation conditions
    # are still applied on top, which layers observation-process shift over real covariate shift.
    "P19_hospB":      {"vanilla": "configs/P19_hospB_baseline.yaml",
                       "opera":   "configs/P19_hospB_distillation.yaml"},
    # eICU demo, mortality with a held-out hospital region. Underpowered by construction (57 deaths in
    # the test region); reportable as a large-effect check, not as an arm-vs-arm comparison.
    "eicu_demo":      {"vanilla": "configs/eicu_demo_baseline.yaml",
                       "opera":   "configs/eicu_demo_distillation.yaml"},
}

# Cheap settings first, so a configuration error surfaces in minutes rather than after an hour.
# A setting missing from this list is silently dropped by plan(), so every SETTINGS key must appear.
ORDER = ["eicu_demo", "PAM", "ushcn_12mo", "ushcn_6mo", "P19", "P19_hospB", "activity",
         "activity_h2000", "activity_h1000", "P12", "ushcn_1mo", "physionet", "physionet_h12",
         "physionet_h36"]
assert set(ORDER) == set(SETTINGS), f"ORDER/SETTINGS mismatch: {set(ORDER) ^ set(SETTINGS)}"

CONDITIONS = ",".join([
    "identity",
    "random_dropout:0.7",
    "temporal_stride:3",
    "variable_dropout:0.5",
    "temporal_stride:3+variable_group_dropout:0.3",
])

STAGES = {
    "A": ("vanilla", [0]),
    "B": ("vanilla", [1, 2, 3, 4]),
    "C": ("opera", [0]),
    "D": ("opera", [1, 2, 3, 4]),
    "E": ("operapp", [0]),          # OPERA++ = OPERA + mining from the harsher pool
    "F": ("operapp", [1, 2, 3, 4]),
    # OPERA++ = OPERA + conditioning on the observation descriptor r = g(S). G is the full descriptor,
    # H the scalar-density ablation. Both run all 5 seeds in one stage; the teacher is the same-cell
    # vanilla checkpoint, exactly as for opera.
    # Seeds 1-4, not 0-4: the seed-0 vanilla checkpoints were trained on the other machine and are not on
    # scratch, and stage D/F already use 1-4, so the three arms stay paired on identical teachers.
    "G": ("operapp2", [1, 2, 3, 4]),
    "H": ("operapp2_scalar", [1, 2, 3, 4]),
}


def run_name(setting, backbone, method, seed):
    return f"bench_{setting}_{backbone}_{method}_s{seed}"


def gpu_free_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        return int(out.strip().split("\n")[0])
    except Exception:
        return 0


def wait_for_free_gpu(threshold_mb=1500, timeout_s=300):
    """The previous job's teardown can still hold GB when the next one starts."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if gpu_free_mb() <= threshold_mb:
            return True
        time.sleep(10)
    print(f"[warn] GPU still holds {gpu_free_mb()} MB after {timeout_s}s; continuing anyway", flush=True)
    return False


def plan(stage, settings, backbones):
    """The (setting, backbone, method, seed) cells this stage runs, in a sensible order."""
    if stage == "smoke":
        cells = []
        for backbone in backbones:
            for setting in ORDER:
                if setting in settings:
                    cells.append((setting, backbone, "vanilla", 0))
        return cells
    method, seeds = STAGES[stage]
    return [(s, b, method, seed) for seed in seeds for b in backbones
            for s in ORDER if s in settings]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["smoke", "A", "B", "C", "D", "E", "F", "G", "H"])
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--settings", default=None, help="comma-separated subset")
    ap.add_argument("--backbones", default=None, help="comma-separated subset")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated subset of the stage's seeds, e.g. 2. Lets a cluster put one "
                         "(setting, backbone, seed) cell in each array task.")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smoke only: cap records, hides OOM")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--no_csv", action="store_true",
                    help="skip the end-of-stage CSV rebuild. Required when one array task runs one cell: the "
                         "rebuild ignores --out, writes into the repo, and 84 tasks would race on it. "
                         "Aggregate once at the end with scripts/benchmark_csv.py --raw <out>.")
    a = ap.parse_args()

    settings = a.settings.split(",") if a.settings else list(SETTINGS)
    backbones = a.backbones.split(",") if a.backbones else BACKBONES
    for s in settings:
        if s not in SETTINGS:
            sys.exit(f"unknown setting {s!r}; known: {', '.join(SETTINGS)}")

    cells = plan(a.stage, settings, backbones)
    if a.seeds:
        want = {int(x) for x in a.seeds.split(",")}
        stage_seeds = set(STAGES[a.stage][1]) if a.stage in STAGES else {0}
        unknown = want - stage_seeds
        if unknown:
            sys.exit(f"stage {a.stage} has seeds {sorted(stage_seeds)}; got {sorted(unknown)}")
        cells = [c for c in cells if c[3] in want]
    print(f"stage {a.stage}: {len(cells)} cells", flush=True)

    done, failed, instant = [], [], 0
    for i, (setting, backbone, method, seed) in enumerate(cells, 1):
        name = run_name(setting, backbone, method, seed)
        if a.stage == "smoke":
            name = f"_smoke_{setting}_{backbone}"
        out_dir = os.path.join(a.out, name)
        if os.path.exists(os.path.join(out_dir, "results.json")):
            print(f"[{i}/{len(cells)}] skip {name} (done)", flush=True)
            done.append(name)
            continue

        cfg = SETTINGS[setting][method]
        overrides = [f"model.backbone={backbone}"]
        if a.stage == "smoke":
            # full data by default: the point of the smoke stage is to catch out-of-memory and patch-count
            # failures, and a record cap hides exactly those. --quick trades that away for speed.
            overrides += ["training.epochs=1", "training.min_epochs=1"]
            if a.quick:
                overrides.append("data.n_records=600")

        if method != "vanilla":
            # the teacher must share the student's architecture: same setting, same backbone, same seed
            teacher = run_name(setting, backbone, "vanilla", seed)
            ckpt = os.path.join(a.out, teacher, "model.pt")
            if not os.path.exists(ckpt):
                print(f"[{i}/{len(cells)}] ABORT {name}: teacher {ckpt} missing; run its vanilla cell first",
                      flush=True)
                failed.append((name, "missing teacher"))
                continue
            overrides.append(f"distillation.teacher_ckpt={ckpt}")

        cmd = [PYTHON, "scripts/train.py", "--config", cfg, "--seed", str(seed),
               "--name", name, "--out", a.out, "--conditions", CONDITIONS, "--set", *overrides]
        if a.no_wandb or a.stage == "smoke":
            cmd.append("--no_wandb")

        print(f"\n[{i}/{len(cells)}] {name}\n  {' '.join(cmd)}", flush=True)
        if a.dry_run:
            continue
        wait_for_free_gpu()
        t0 = time.time()
        rc = subprocess.call(cmd)
        dt = time.time() - t0
        if rc == 0:
            print(f"[{i}/{len(cells)}] ok {name} ({dt/60:.1f} min)", flush=True)
            done.append(name)
        else:
            print(f"[{i}/{len(cells)}] FAILED {name} rc={rc} ({dt/60:.1f} min)", flush=True)
            failed.append((name, f"rc={rc}"))
            # A cell that dies before it could plausibly have started is environmental, not a bad config:
            # an OS shutdown, a dead GPU, a broken venv. Windows reports these as 0xC000026B
            # (DLL_INIT_FAILED_LOGOFF) / 0x40010004 (DBG_TERMINATE_PROCESS). Without this guard the launcher
            # marches through the whole queue at 0.0 min per cell and marks every remaining cell FAILED, which
            # is what happened during a machine restart: six cells burned in seconds.
            if dt < 30:
                instant += 1
                if instant >= 2:
                    print(f"[abort] {instant} cells failed in under 30s; the environment is going down or "
                          f"broken, not the configs. Stopping so the queue is not burned. Re-run this "
                          f"stage once the machine is up; finished cells are skipped.", flush=True)
                    break
            else:
                instant = 0

    print(f"\nstage {a.stage}: {len(done)} ok, {len(failed)} failed")
    for n, why in failed:
        print(f"  FAILED {n}: {why}")
    if not a.dry_run and a.stage != "smoke" and not a.no_csv:
        subprocess.call([PYTHON, "scripts/benchmark_csv.py"])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
