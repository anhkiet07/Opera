"""Aggregate results/*.json into Markdown tables (Phase 0 cost + Phase 1 robustness).

Usage: python summarize_results.py [--glob "results/p*_*.json"] [--out results/summary.md]
Runs whose tag differs only by the `_s<seed>` suffix are grouped and reported as mean ± std.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


def load(pattern):
    runs = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def group_key(tag):
    return re.sub(r"_s\d+$", "", tag)


def fmt(vals, scale=1.0, nd=4):
    vals = [v * scale for v in vals if v is not None]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.{nd}f}"
    return f"{np.mean(vals):.{nd}f} ± {np.std(vals):.{nd}f}"


def forecasting_tables(runs):
    groups = defaultdict(list)
    for r in runs:
        groups[group_key(r["run_tag"])].append(r)
    lines = []
    # ---- Phase 0 / cost ----
    lines.append("## Forecasting: test performance and cost (PhysioNet, 24h -> 24h)\n")
    lines.append("| config | seeds | test MSE | test MAE | best epoch | params | s / epoch | peak GPU MB | infer ms / series |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for g, rs in groups.items():
        rs = [r for r in rs if "test" in r]
        if not rs:
            continue
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            g, len(rs),
            fmt([r["test"]["mse"] for r in rs]), fmt([r["test"]["mae"] for r in rs]),
            fmt([r["best_epoch"] for r in rs], nd=1), rs[0]["n_params"],
            fmt([r["train_time_per_epoch_s"] for r in rs], nd=1),
            fmt([r.get("peak_gpu_mem_mb") for r in rs], nd=0),
            fmt([r["test"]["infer_ms_per_series"] for r in rs], nd=3)))
    # ---- Phase 1 robustness ----
    views = None
    for rs in groups.values():
        for r in rs:
            if "robustness" in r:
                views = list(r["robustness"].keys()); break
        if views: break
    if views:
        for metric in ("mse", "mae"):
            lines.append(f"\n## Cross-sampling robustness: test {metric.upper()} under each observation view of the test inputs\n")
            lines.append("| config | seeds | " + " | ".join(views) + " | mean over degraded views | rel. degradation vs none |")
            lines.append("|---|---|" + "---|" * (len(views) + 2))
            for g, rs in groups.items():
                rs = [r for r in rs if "robustness" in r]
                if not rs:
                    continue
                cells = [fmt([r["robustness"][v][metric] for r in rs]) for v in views]
                degraded = [np.mean([r["robustness"][v][metric] for v in views if v != "none"]) for r in rs]
                rel = [(d / r["robustness"]["none"][metric] - 1) * 100 for d, r in zip(degraded, rs)]
                lines.append(f"| {g} | {len(rs)} | " + " | ".join(cells) + f" | {fmt(degraded)} | {fmt(rel, nd=1)}% |")
    return "\n".join(lines)


def classification_tables(runs):
    lines = ["## Classification: test performance (mean over folds) and cost\n"]
    lines.append("| config | folds | acc | AUROC | AUPRC | params | s / epoch | peak GPU MB | infer ms / sample |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        f = r["final"]
        lines.append("| {} | {} | {:.2f} ± {:.2f} | {} | {} | {} | {} | {} | {} |".format(
            r["run_tag"], r["args"].get("n_folds"), f["test/acc_mean"], f["test/acc_std"],
            f"{f['test/auroc_mean']:.2f} ± {f['test/auroc_std']:.2f}" if "test/auroc_mean" in f else "-",
            f"{f['test/auprc_mean']:.2f} ± {f['test/auprc_std']:.2f}" if "test/auprc_mean" in f else "-",
            f.get("n_params"), fmt([f.get("train_time_per_epoch_s")], nd=1),
            fmt([f.get("peak_gpu_mem_mb")], nd=0), fmt([f.get("infer_ms_per_sample")], nd=3)))
    views = None
    for r in runs:
        if r.get("robustness_mean"):
            views = list(r["robustness_mean"].keys()); break
    if views:
        metric = "auroc" if "auroc" in runs[0]["robustness_mean"][views[0]] else "acc"
        lines.append(f"\n## Cross-sampling robustness: test {metric.upper()} (mean over folds) under each view\n")
        lines.append("| config | " + " | ".join(views) + " |")
        lines.append("|---|" + "---|" * len(views))
        for r in runs:
            if not r.get("robustness_mean"):
                continue
            lines.append(f"| {r['run_tag']} | " + " | ".join(
                f"{r['robustness_mean'][v][metric]:.2f}" for v in views) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/p*.json")
    ap.add_argument("--out", default="results/summary.md")
    a = ap.parse_args()
    runs = load(a.glob)
    fc = [r for r in runs if "robustness" in r or "test" in r]
    cl = [r for r in runs if "final" in r]
    parts = []
    if fc:
        parts.append(forecasting_tables(fc))
    if cl:
        parts.append(classification_tables(cl))
    md = "\n\n".join(parts) + "\n"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
