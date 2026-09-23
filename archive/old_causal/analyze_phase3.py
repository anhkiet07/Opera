"""Phase 3 tables: overall test MSE and target-variable (Y = variable 2) MSE per view, mean +- std over seeds.

Usage: python analyze_phase3.py [--out results/phase3_tables.md]
"""
import argparse
import glob
import json
import re
from collections import defaultdict

import numpy as np

from data.synthetic import TARGET

VIEWS = ["none", "random:0.5", "random:0.7", "variable:0.3", "dropvars:5-6-7-8-9", "dropvars:1-4", "dropvars:0-3", "dropvars:9"]
ORDER = ["p3_vanilla", "p3_quite", "p3_soft_true_lam2", "p3_soft_random_lam2", "p3_soft_shuffled_lam2",
         "p3_soft_identity_lam2", "p3_soft_noisy0.3_lam2", "p3_soft_true_lam0.5", "p3_soft_true_lam5",
         "p3_hard_true", "p3_learned", "p3_vanilla_soft_true_lam2"]


def fmt(vals, nd=4):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    return f"{np.mean(vals):.{nd}f}" if len(vals) == 1 else f"{np.mean(vals):.{nd}f} ± {np.std(vals):.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase3_tables.md")
    a = ap.parse_args()
    groups = defaultdict(list)
    for f in glob.glob("results/p3_*.json") + glob.glob("results/p3low_*.json"):
        if f.endswith(".stuck.json"):
            continue
        r = json.load(open(f))
        mses = [v["mse"] for v in r.get("robustness", {}).values()]
        if len(mses) > 1 and max(mses) - min(mses) < 1e-4:   # constant-predictor (plateau) run
            continue
        groups[re.sub(r"_s\d+$", "", r["run_tag"])].append(r)
    lines = []
    for title, key in [("Overall test MSE (all variables)", lambda rv: rv["mse"]),
                       (f"Target-variable (Y = var {TARGET}) test MSE", lambda rv: rv["mse_per_var"][TARGET])]:
        lines.append(f"## {title}\n")
        lines.append("| config | seeds | best ep | " + " | ".join(VIEWS) + " |")
        lines.append("|---|---|---|" + "---|" * len(VIEWS))
        for g in ORDER + sorted(k for k in groups if k not in ORDER):
            rs = groups.get(g)
            if not rs:
                continue
            cells = [fmt([key(r["robustness"][v]) for r in rs if v in r["robustness"]]) for v in VIEWS]
            lines.append(f"| {g} | {len(rs)} | {fmt([r['best_epoch'] for r in rs], nd=0)} | " + " | ".join(cells) + " |")
        lines.append("")
    md = "\n".join(lines)
    open(a.out, "w", encoding="utf-8").write(md)
    print(md)


if __name__ == "__main__":
    main()
