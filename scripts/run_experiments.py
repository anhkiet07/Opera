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
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from train import load_yaml  # noqa: E402  (config resolution incl. _base_ inheritance)

STAGES = {
    "baseline": ["configs/baseline.yaml", "configs/augmentation.yaml"],
    "opera": ["configs/distillation.yaml", "configs/hard_mining.yaml", "configs/opera.yaml"],
    "all": ["configs/baseline.yaml", "configs/augmentation.yaml", "configs/distillation.yaml",
            "configs/hard_mining.yaml", "configs/opera.yaml"],
}
NEEDS_TEACHER = ("distillation", "opera")


def wait_for_free_gpu(threshold_mb=1500, timeout_s=180):
    """Block until the GPU is (nearly) idle.

    A finished run can still hold GPU memory for a few seconds while its process and W&B sync tear down. On a
    small card that overlap is enough to make the next run fail with an out-of-memory error even though its own
    footprint fits, so the launcher waits rather than racing.
    """
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
            last = int(out.strip().splitlines()[0])
        except Exception:
            return
        if last <= threshold_mb:
            return
        time.sleep(5)
    print(f"[warn] GPU still holds {last} MB after {timeout_s}s; starting anyway", flush=True)


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
            # Any config that enables distillation needs a frozen rich-view teacher: the same-seed E0 run of the
            # SAME dataset. Resolve it from the config's contents, not its filename, so variants like
            # 'opera_strong' or 'ushcn_opera' are handled the same way.
            resolved = load_yaml(cfg)
            if (resolved.get("distillation") or {}).get("enabled"):
                dataset = (resolved.get("data") or {}).get("name", "physionet")
                teacher = f"baseline_s{seed}" if dataset == "physionet" else f"{dataset}_baseline_s{seed}"
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
            wait_for_free_gpu()
            r = subprocess.run(cmd, cwd=ROOT)
            print(f"[{'done' if r.returncode == 0 else 'FAILED'}] {name} (exit {r.returncode})", flush=True)
    print("LAUNCHER DONE", flush=True)


if __name__ == "__main__":
    main()
