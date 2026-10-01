"""Single-command reproducible OPERA run (OPERA sections 33-34).

    python scripts/train.py --config configs/opera.yaml --seed 0
    python scripts/train.py --config configs/baseline.yaml --seed 0 --set training.epochs=2 data.n_records=800

Everything the run needs is saved next to the results: config, git commit, seed, metrics, checkpoint.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_yaml(path):
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base:
        parent = load_yaml(os.path.join(os.path.dirname(path), base))
        cfg = deep_merge(parent, cfg)
    return cfg


def deep_merge(a, b):
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = deep_merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return out


def set_path(cfg, dotted, value):
    import yaml
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(value)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None, help="run name; defaults to <config stem>_s<seed>")
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--set", nargs="*", default=[], help="dotted.key=value overrides")
    ap.add_argument("--conditions", default=None, help="comma-separated test conditions (default: TEST_CONDITIONS)")
    ap.add_argument("--no_wandb", action="store_true")
    a = ap.parse_args()

    cfg = load_yaml(a.config)
    for kv in a.set:
        k, v = kv.split("=", 1)
        set_path(cfg, k, v)
    cfg["seed"] = a.seed
    name = a.name or f"{os.path.splitext(os.path.basename(a.config))[0]}_s{a.seed}"
    out_dir = os.path.join(a.out, name)
    os.makedirs(out_dir, exist_ok=True)

    import torch
    import utils
    from src.datasets import registry as ds_registry
    from src.models.base import build_model
    from src.training.trainer import Trainer

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    utils.setup_seed(a.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dsname = cfg.get("data", {}).get("name", "physionet")
    # Route on the classification registry itself rather than a hardcoded tuple, so a new classification
    # setting only has to be added to src/datasets/classification.CFG. A hardcoded list silently sent any
    # new name to the forecasting registry, which failed with "unknown dataset".
    from src.datasets import classification as ds_cls
    mod = ds_cls if dsname in ds_cls.CFG else ds_registry
    data_obj, info = mod.load(cfg, device, seed=a.seed)
    model, model_args = build_model(cfg, info, device)

    log_path = os.path.join(out_dir, "train.log")
    logf = open(log_path, "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(str(msg) + "\n")
        logf.flush()

    run = None
    if not a.no_wandb and cfg.get("wandb", {}).get("enabled", True):
        try:
            import wandb
            run = wandb.init(project=cfg.get("wandb", {}).get("project", "opera"), name=name,
                             config={"seed": a.seed, **_flat(cfg)},
                             tags=[dsname, cfg.get("experiment", "custom")])
        except Exception as e:
            log(f"wandb disabled: {e}")

    log(f"run {name} | device {device} | params {model.n_params()} | commit {git_commit()}")
    log(f"config: {json.dumps(cfg, sort_keys=True)}")

    trainer = Trainer(cfg, model, data_obj, info, mod.next_batch, device, out_dir, logger=log,
                      wandb_run=run, build_fn=build_model)
    best = trainer.fit()
    conds = a.conditions.split(",") if a.conditions else None
    res = trainer.test(conditions=conds)
    res["run"] = {"name": name, "seed": a.seed, "config_path": a.config, "git_commit": git_commit(),
                  "experiment": cfg.get("experiment", "custom"), "dataset": dsname}
    path = trainer.save(res)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        import yaml
        yaml.safe_dump(cfg, f, sort_keys=True)
    log(f"best {best} -> {path}")
    log("summary " + json.dumps(res["summary"], indent=2))
    if run is not None:
        run.finish()


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        elif isinstance(v, (int, float, str, bool, type(None))):
            out[key] = v
        else:
            out[key] = str(v)
    return out


if __name__ == "__main__":
    main()
