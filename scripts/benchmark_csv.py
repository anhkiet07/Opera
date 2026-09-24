"""Emit the single benchmark CSV from results/raw/ (BENCHMARK_PROTOCOL.md section 5).

One row per (setting, backbone, method, seed, condition). Every applicable metric sits in its own column, plus
`primary_error`, which is lower-is-better for both tasks so that relative degradation is a one-line pivot:

    degradation = (primary_error[condition] - primary_error[clean]) / primary_error[clean]

No derived statistic is stored here. `benchmark_summary.csv` holds mean/std over seeds and is generated from
this file, never edited.

    python scripts/benchmark_csv.py
    python scripts/benchmark_csv.py --main-only      # restrict to the 5 protocol conditions
"""
import argparse
import csv
import glob
import json
import os
import sys
from statistics import mean, pstdev

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------- protocol constants
MAIN_CONDITIONS = {
    "identity": ("clean", "none", True),
    "random_dropout:0.7": ("density", "density", False),
    "temporal_stride:3": ("timing", "timing", False),
    "variable_dropout:0.5": ("variables", "variables", False),
    "temporal_stride:3+variable_group_dropout:0.3": ("composed", "composed", False),
}

# families present in the training pool (used for seen_in_training on the wider sweep)
TRAIN_FAMILIES = {"identity", "random_dropout", "temporal_stride", "burst", "variable_dropout", "local_mask"}
TRAIN_SEVERITY = {  # the exact severities the training pool draws
    "random_dropout": {0.1, 0.25, 0.4}, "temporal_stride": {2.0}, "burst": {0.2, 0.3},
    "variable_dropout": {0.2}, "local_mask": {0.3},
}

PRIMARY = {  # setting -> primary metric key in the stored per-condition dict
    "physionet": "mse", "activity": "mse", "ushcn_1mo": "mse", "ushcn_12mo": "mse",
    "P12": "auroc", "P19": "auroc", "PAM": "acc",
}
TASK = {"physionet": "forecast", "activity": "forecast", "ushcn_1mo": "forecast", "ushcn_12mo": "forecast",
        "P12": "classify", "P19": "classify", "PAM": "classify"}

FIELDS = [
    "setting", "task", "n_vars", "backbone", "method", "variant", "experiment", "seed", "batch_size",
    "condition", "condition_spec", "dimension", "seen_in_training",
    "primary_metric", "primary_value", "primary_error",
    "mse", "mae", "ce", "acc", "auroc", "auprc", "f1",
    "n_series", "infer_ms_per_series", "params", "train_time_per_epoch_s", "epochs_run", "peak_gpu_mem_mb",
    "run_dir", "git_commit",
]


def setting_id(cfg, info):
    """Dataset name, with the USHCN horizon folded in so the two settings are distinct rows."""
    name = (cfg.get("data") or {}).get("name", "physionet")
    if name == "ushcn":
        return f"ushcn_{int(info.get('pred_window', 1))}mo"
    return name


def method_id(cfg):
    """vanilla | augmentation | distillation | hard_mining | opera, from what is switched on."""
    obs = cfg.get("observation") or {}
    aug = list(obs.get("train_processes") or ["identity"]) != ["identity"] or int(obs.get("num_views", 1)) > 1
    distill = bool((cfg.get("distillation") or {}).get("enabled"))
    mining = bool((cfg.get("hard_mining") or {}).get("enabled"))
    if not (aug or distill or mining):
        return "vanilla"
    if distill and mining:
        return "opera"
    if distill:
        return "distillation"
    if mining:
        return "hard_mining"
    return "augmentation"


def variant_id(cfg):
    """Distinguishes the mining candidate pool. Without an explicit pool the miner scores the training
    mixture, which is a different mechanism in practice (BENCHMARK_PROTOCOL.md section 2)."""
    hm = cfg.get("hard_mining") or {}
    if not hm.get("enabled"):
        return ""
    return "harsh_pool" if hm.get("pool") else "train_pool"


def classify_condition(spec):
    """(column, dimension, seen_in_training) for any condition, protocol or wider sweep."""
    if spec in MAIN_CONDITIONS:
        return MAIN_CONDITIONS[spec]
    if "+" in spec:
        return (spec, "composed", False)
    fam, _, sev = spec.partition(":")
    seen = False
    if fam in TRAIN_SEVERITY and sev:
        try:
            seen = float(sev) in TRAIN_SEVERITY[fam]
        except ValueError:
            seen = False
    dim = {"random_dropout": "density", "density": "density", "temporal_stride": "timing",
           "burst": "timing", "long_gap": "timing", "local_mask": "timing",
           "variable_dropout": "variables", "variable_group_dropout": "variables"}.get(fam, "other")
    return (spec, dim, seen)


def rows_for_run(run_dir, main_only):
    path = os.path.join(run_dir, "results.json")
    if not os.path.exists(path):
        return []
    r = json.load(open(path, encoding="utf-8"))
    cfg, info, per = r.get("config") or {}, r.get("info") or {}, r.get("per_condition") or {}
    if not per:
        return []
    cost, run = r.get("cost") or {}, r.get("run") or {}
    sid = setting_id(cfg, info)
    if sid not in PRIMARY:
        return []
    pkey = PRIMARY[sid]
    base = {
        "setting": sid, "task": TASK[sid], "n_vars": info.get("input_dim"),
        "backbone": (cfg.get("model") or {}).get("backbone"),
        "method": method_id(cfg), "variant": variant_id(cfg), "experiment": cfg.get("experiment"),
        "seed": cfg.get("seed", run.get("seed")), "batch_size": info.get("batch_size"),
        "params": cost.get("n_params"), "train_time_per_epoch_s": cost.get("train_time_per_epoch_s"),
        "epochs_run": cost.get("epochs_run"), "peak_gpu_mem_mb": cost.get("peak_gpu_mem_mb"),
        "run_dir": os.path.basename(run_dir), "git_commit": run.get("git_commit"),
    }
    out = []
    for spec, m in per.items():
        if main_only and spec not in MAIN_CONDITIONS:
            continue
        col, dim, seen = classify_condition(spec)
        pv = m.get(pkey)
        if pv is None:
            continue
        perr = pv if pkey == "mse" else 1.0 - pv / 100.0
        row = dict(base)
        row.update({
            "condition": col, "condition_spec": spec, "dimension": dim, "seen_in_training": seen,
            "primary_metric": pkey, "primary_value": pv, "primary_error": perr,
            "n_series": m.get("n_series"), "infer_ms_per_series": m.get("infer_ms_per_series"),
        })
        # metric columns: blank, not zero, where the metric does not apply to the task
        if TASK[sid] == "forecast":
            row["mse"], row["mae"] = m.get("mse"), m.get("mae")
        else:
            row["ce"], row["acc"] = m.get("ce"), m.get("acc")
            row["auroc"], row["auprc"] = m.get("auroc"), m.get("auprc")
            row["f1"] = m.get("f1")
        out.append(row)
    return out


def write_summary(rows, path):
    """mean/std over seeds per (setting, backbone, method, condition), plus derived degradation."""
    keyed = {}
    for r in rows:
        keyed.setdefault((r["setting"], r["backbone"], r["method"], r["variant"], r["condition"]),
                         []).append(r)
    clean = {}
    for (s, b, m, v, c), rs in keyed.items():
        if c == "clean":
            clean[(s, b, m, v)] = mean(x["primary_error"] for x in rs)
    fields = ["setting", "task", "backbone", "method", "variant", "condition", "dimension", "n_seeds",
              "primary_metric", "primary_value_mean", "primary_value_std",
              "primary_error_mean", "degradation_pct"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (s, b, m, v, c), rs in sorted(keyed.items()):
            vals = [x["primary_value"] for x in rs]
            errs = [x["primary_error"] for x in rs]
            e0 = clean.get((s, b, m, v))
            w.writerow({
                "setting": s, "task": rs[0]["task"], "backbone": b, "method": m, "variant": v,
                "condition": c,
                "dimension": rs[0]["dimension"], "n_seeds": len(rs),
                "primary_metric": rs[0]["primary_metric"],
                "primary_value_mean": round(mean(vals), 6),
                "primary_value_std": round(pstdev(vals), 6) if len(vals) > 1 else "",
                "primary_error_mean": round(mean(errs), 6),
                "degradation_pct": ("" if not e0 or c == "clean"
                                    else round(100.0 * (mean(errs) - e0) / e0, 2)),
            })
    return len(keyed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--out", default="results/benchmark.csv")
    ap.add_argument("--summary", default="results/benchmark_summary.csv")
    ap.add_argument("--main-only", action="store_true", help="keep only the 5 protocol conditions")
    a = ap.parse_args()

    dirs = sorted(d for d in glob.glob(os.path.join(a.raw, "*")) if os.path.isdir(d)
                  and not os.path.basename(d).startswith("_"))
    rows, skipped = [], []
    for d in dirs:
        rs = rows_for_run(d, a.main_only)
        (rows.extend(rs) if rs else skipped.append(os.path.basename(d)))

    rows.sort(key=lambda r: (r["setting"], r["backbone"], r["method"], r["seed"], r["condition"]))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})

    n_groups = write_summary(rows, a.summary)
    cells = {(r["setting"], r["backbone"], r["method"], r["variant"], r["seed"]) for r in rows}
    print(f"{a.out}: {len(rows)} rows from {len(cells)} run cells")
    print(f"{a.summary}: {n_groups} groups")
    if skipped:
        print(f"skipped (no usable per-condition metrics): {', '.join(skipped)}")
    print("\ncoverage by setting x backbone x method:")
    cov = {}
    for s, b, m, v, sd in sorted(cells):
        cov.setdefault((s, b, m, v), []).append(sd)
    for (s, b, m, v), seeds in sorted(cov.items()):
        print(f"  {s:12s} {b:13s} {m:14s} {v:11s} seeds {sorted(seeds)}")


if __name__ == "__main__":
    main()
