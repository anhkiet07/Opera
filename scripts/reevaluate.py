"""Re-score finished runs from their saved checkpoint, without retraining.

    python scripts/reevaluate.py --run_dir results/raw/bench_P12_patchtst_vanilla_s1
    python scripts/reevaluate.py --glob '/scratch/.../raw/bench_P12_*'  --dry_run

Why this exists: `classification_metrics` originally emitted macro F1 / precision / recall only on the
multi-class branch, so the binary tasks (P12 mortality, P19 sepsis) have no F1, precision or recall in
their stored results. Those are functions of the saved predictions, not of training, so the runs do not
need to be repeated -- only re-scored.

Evaluation is deterministic: the re-observation seed is fixed at 12345 (BENCHMARK_PROTOCOL section 6), so
every condition is replayed on byte-identical inputs. This script therefore verifies that the metrics that
already existed reproduce exactly before it overwrites anything, and refuses to write if they do not.

`cost`, `best` and `run` are carried over from the existing results.json: a re-evaluation has no training
history, so recomputing them would replace real timings with nulls.
"""
import argparse
import glob as globmod
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONDITIONS = ["identity", "random_dropout:0.7", "temporal_stride:3", "variable_dropout:0.5",
              "temporal_stride:3+variable_group_dropout:0.3"]
# metrics that already existed for classification runs; must reproduce on replay
CHECK = ["mse", "ce", "acc", "auroc", "auprc", "f1", "mae"]
# Relative tolerance. Non-deterministic floating-point reduction order across GPUs moves metrics by ~1e-5
# relative, which is not a faithfulness problem. A genuinely different test input moves them by ~1e-2:
# runs produced on a different platform replay their shifted conditions against different re-observation
# masks, and those must still be rejected. 1e-4 separates the two cleanly.
TOL = 1e-4


def reevaluate(run_dir, device, dry_run=False, tol=TOL):
    import torch
    import yaml
    import utils
    from src.models.base import build_model
    from src.training.trainer import Trainer

    res_path = os.path.join(run_dir, "results.json")
    ckpt_path = os.path.join(run_dir, "model.pt")
    if not os.path.exists(res_path):
        return "no results.json"
    if not os.path.exists(ckpt_path):
        return "no model.pt"

    old = json.load(open(res_path))
    cfg = old.get("config") or yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))
    seed = int(cfg.get("seed", old.get("run", {}).get("seed", 0)))

    # Evaluation runs the student forward pass only, so the teacher is irrelevant here. Leaving
    # distillation enabled makes build_model try to load `distillation.teacher_ckpt`, which fails for any
    # run whose config recorded a path from another machine (the seed-0 runs carry Windows paths such as
    # 'results/raw\\bench_..._vanilla_s0\\model.pt'). The student weights come from model.pt regardless.
    cfg = json.loads(json.dumps(cfg))          # don't mutate the block we write back
    cfg.setdefault("distillation", {})["enabled"] = False
    cfg["distillation"]["teacher_ckpt"] = None
    cfg.setdefault("hard_mining", {})["enabled"] = False

    utils.setup_seed(seed)
    dsname = cfg.get("data", {}).get("name", "physionet")
    if dsname in ("P12", "P19", "PAM"):
        from src.datasets import classification as mod
    else:
        from src.datasets import registry as mod
    data_obj, info = mod.load(cfg, device, seed=seed)
    model, _ = build_model(cfg, info, device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])

    trainer = Trainer(cfg, model, data_obj, info, mod.next_batch, device, run_dir,
                      logger=lambda *a: None, wandb_run=None, build_fn=build_model)
    new = trainer.test(conditions=CONDITIONS)

    # every metric that already existed must reproduce, or the replay is not faithful
    for cond, oldm in old.get("per_condition", {}).items():
        newm = new["per_condition"].get(cond)
        if newm is None:
            return f"condition {cond} missing after replay"
        for k in CHECK:
            if k in oldm and isinstance(oldm[k], (int, float)) and k in newm:
                denom = max(abs(oldm[k]), 1e-12)
                rel = abs(oldm[k] - newm[k]) / denom
                if rel > tol:
                    return (f"MISMATCH {cond}.{k}: {oldm[k]:.8f} -> {newm[k]:.8f} "
                            f"(rel {rel:.2e})")

    added = sorted(set(new["per_condition"]["identity"]) - set(old["per_condition"]["identity"]))
    if dry_run:
        return f"ok (dry run) added={added or 'nothing'}"

    merged = dict(old)
    merged["per_condition"] = new["per_condition"]
    merged["summary"] = new["summary"]
    # keep the originals: a re-evaluation has no epoch timings or early-stopping record
    for k in ("cost", "best", "run", "config", "info"):
        if k in old:
            merged[k] = old[k]
    tmp = res_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=2)
    os.replace(tmp, res_path)
    return f"ok added={added or 'nothing'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--glob", default=None, help="shell glob of run dirs")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    if not a.run_dir and not a.glob:
        sys.exit("need --run_dir or --glob")

    import torch
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dirs = [a.run_dir] if a.run_dir else sorted(d for d in globmod.glob(a.glob) if os.path.isdir(d))
    print(f"re-evaluating {len(dirs)} run(s) on {device}", flush=True)
    bad = 0
    for i, d in enumerate(dirs, 1):
        try:
            msg = reevaluate(d, device, dry_run=a.dry_run)
        except Exception as e:
            msg = f"ERROR {type(e).__name__}: {e}"
        if not msg.startswith("ok"):
            bad += 1
        print(f"[{i}/{len(dirs)}] {os.path.basename(d)}: {msg}", flush=True)
    print(f"\n{len(dirs) - bad} ok, {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
