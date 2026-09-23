"""Experiment launcher (OPERA section 35): runs configs x seeds sequentially and aggregates.

    python scripts/run_experiments.py --stage baseline      # E0 + E1, seeds 0-4  (Step 2 deliverable)
    python scripts/run_experiments.py --stage opera         # E2, E3, E4, seeds 0-4
    python scripts/run_experiments.py --configs configs/opera.yaml --seeds 0 1 --set training.epochs=5

A run whose results.json already exists is skipped, so the queue is resumable.
E2/E4 use the same-seed E0 checkpoint as their frozen rich-view teacher (E0 trains on the full observation).
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

STAGES = {
    "baseline": ["configs/baseline.yaml", "configs/augmentation.yaml"],
    "opera": ["configs/distillation.yaml", "configs/hard_mining.yaml", "configs/opera.yaml"],
    "all": ["configs/baseline.yaml", "configs/augmentation.yaml", "configs/distillation.yaml",
            "configs/hard_mining.yaml", "configs/opera.yaml"],
}
NEEDS_TEACHER = ("distillation", "opera")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=sorted(STAGES), default=None)
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--conditions", default=None)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    configs = a.configs or STAGES[a.stage or "baseline"]
    for cfg in configs:
        stem = os.path.splitext(os.path.basename(cfg))[0]
        for seed in a.seeds:
            name = f"{stem}_s{seed}"
            if os.path.exists(os.path.join(a.out, name, "results.json")):
                print(f"[skip] {name}", flush=True)
                continue
            overrides = list(a.set)
            if stem.endswith(NEEDS_TEACHER):
                # The frozen rich-view teacher is the same-seed E0 run OF THE SAME DATASET: config stems are
                # '<dataset>_distillation' / '<dataset>_opera' (or bare, for the default dataset), so the
                # teacher is '<dataset>_baseline_s<seed>' / 'baseline_s<seed>'.
                for suffix in NEEDS_TEACHER:
                    if stem.endswith(suffix):
                        prefix = stem[: -len(suffix)]
                        break
                teacher = f"{prefix}baseline_s{seed}"
                ckpt = os.path.join(a.out, teacher, "model.pt")
                if not os.path.exists(ckpt):
                    print(f"[warn] {name}: teacher checkpoint {ckpt} is missing; run its baseline first",
                          flush=True)
                overrides.append(f"distillation.teacher_ckpt={a.out}/{teacher}/model.pt")
            cmd = [sys.executable, "scripts/train.py", "--config", cfg, "--seed", str(seed),
                   "--name", name, "--out", a.out]
            if overrides:
                cmd += ["--set"] + overrides
            if a.conditions:
                cmd += ["--conditions", a.conditions]
            if a.no_wandb:
                cmd.append("--no_wandb")
            print(f"===== {name} =====", flush=True)
            print(" ".join(cmd), flush=True)
            if a.dry:
                continue
            r = subprocess.run(cmd, cwd=ROOT)
            print(f"[{'done' if r.returncode == 0 else 'FAILED'}] {name} (exit {r.returncode})", flush=True)
    print("LAUNCHER DONE", flush=True)


if __name__ == "__main__":
    main()
