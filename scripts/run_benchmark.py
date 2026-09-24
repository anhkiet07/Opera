"""Staged benchmark launcher (BENCHMARK_PROTOCOL.md section 7).

    python scripts/run_benchmark.py --stage smoke     # 1 epoch per backbone per task, catches OOM early
    python scripts/run_benchmark.py --stage A         # vanilla, seed 0,   21 runs
    python scripts/run_benchmark.py --stage C         # opera,   seed 0,   21 runs (needs A)
    python scripts/run_benchmark.py --stage B         # vanilla, seeds 1-4, 84 runs
    python scripts/run_benchmark.py --stage D         # opera,   seeds 1-4, 84 runs (needs B)

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

# setting -> (vanilla config, opera config). Each config already carries its dataset's history/patch/batch.
SETTINGS = {
    "physionet":  ("configs/baseline.yaml",            "configs/opera.yaml"),
    "activity":   ("configs/activity_baseline.yaml",   "configs/activity_opera.yaml"),
    "ushcn_1mo":  ("configs/ushcn_baseline.yaml",      "configs/ushcn_opera.yaml"),
    "ushcn_12mo": ("configs/ushcn_long_baseline.yaml", "configs/ushcn_long_opera.yaml"),
    "P12":        ("configs/P12_baseline.yaml",         "configs/P12_opera.yaml"),
    "P19":        ("configs/P19_baseline.yaml",         "configs/P19_opera.yaml"),
    "PAM":        ("configs/PAM_baseline.yaml",         "configs/PAM_opera.yaml"),
}

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
        order = ["PAM", "ushcn_12mo", "P19", "activity", "P12", "ushcn_1mo", "physionet"]
        for backbone in backbones:
            for setting in order:
                if setting in settings:
                    cells.append((setting, backbone, "vanilla", 0))
        return cells
    method, seeds = STAGES[stage]
    # cheap settings first, so a configuration error surfaces in minutes rather than after an hour
    order = ["PAM", "ushcn_12mo", "P19", "activity", "P12", "ushcn_1mo", "physionet"]
    return [(s, b, method, seed) for seed in seeds for b in backbones
            for s in order if s in settings]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["smoke", "A", "B", "C", "D"])
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--settings", default=None, help="comma-separated subset")
    ap.add_argument("--backbones", default=None, help="comma-separated subset")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smoke only: cap records, hides OOM")
    ap.add_argument("--no_wandb", action="store_true")
    a = ap.parse_args()

    settings = a.settings.split(",") if a.settings else list(SETTINGS)
    backbones = a.backbones.split(",") if a.backbones else BACKBONES
    for s in settings:
        if s not in SETTINGS:
            sys.exit(f"unknown setting {s!r}; known: {', '.join(SETTINGS)}")

    cells = plan(a.stage, settings, backbones)
    print(f"stage {a.stage}: {len(cells)} cells", flush=True)

    done, failed = [], []
    for i, (setting, backbone, method, seed) in enumerate(cells, 1):
        name = run_name(setting, backbone, method, seed)
        if a.stage == "smoke":
            name = f"_smoke_{setting}_{backbone}"
        out_dir = os.path.join(a.out, name)
        if os.path.exists(os.path.join(out_dir, "results.json")):
            print(f"[{i}/{len(cells)}] skip {name} (done)", flush=True)
            done.append(name)
            continue

        cfg = SETTINGS[setting][0 if method == "vanilla" else 1]
        overrides = [f"model.backbone={backbone}"]
        if a.stage == "smoke":
            # full data by default: the point of the smoke stage is to catch out-of-memory and patch-count
            # failures, and a record cap hides exactly those. --quick trades that away for speed.
            overrides += ["training.epochs=1", "training.min_epochs=1"]
            if a.quick:
                overrides.append("data.n_records=600")

        if method == "opera":
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

    print(f"\nstage {a.stage}: {len(done)} ok, {len(failed)} failed")
    for n, why in failed:
        print(f"  FAILED {n}: {why}")
    if not a.dry_run and a.stage != "smoke":
        subprocess.call([PYTHON, "scripts/benchmark_csv.py"])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
