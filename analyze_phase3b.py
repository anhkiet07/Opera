"""Phase 3b tables (PhysioNet structural priors): overall test MSE per view, plus MSE restricted to the variables
that have >= 1 parent in the graph a run used ("structured variables"), where a prior can act.

Usage: python analyze_phase3b.py [--out results/phase3b_tables.md]
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from data import DATA_ROOT
from data.physionet import PhysioNet
from data.physionet_prior import adjacency as llm_adjacency

VIEWS = ["none", "random:0.5", "random:0.7", "bursty:0.5", "variable:0.3", "variable:0.5", "stride:3"]
ORDER_LOW = ["p3b_low_plain", "p3b_low_hard_llm", "p3b_low_hard_est", "p3b_low_hard_hybrid",
             "p3b_low_hard_shuffled", "p3b_low_hard_identity", "p3b_low_hard_random",
             "p3b_low_soft_est_lam2", "p3b_low_soft_hybrid_lam2"]
ORDER_FULL = ["p0_quite_itr", "p3phys_hard_llm", "p3b_full_hard_est", "p3b_full_hard_hybrid",
              "p3b_full_hard_shuffled_hybrid", "p3b_full_soft_est_lam2", "p3phys_soft_llm_lam2", "p3phys_learned"]


def graph_for(n_cap, source):
    A_llm = llm_adjacency()
    if source == "llm":
        return A_llm
    A_est = np.load(os.path.join(DATA_ROOT, "physionet", f"estimated_graph_n{n_cap}.npy"))
    return A_est if source == "estimated" else np.maximum(A_llm, A_est)


def fmt(vals, nd=4):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return "-"
    return f"{np.mean(vals):.{nd}f}" if len(vals) == 1 else f"{np.mean(vals):.{nd}f} ± {np.std(vals):.{nd}f}"


def table(groups, order, subset_name, subset):
    lines = [f"| config | seeds | " + " | ".join(VIEWS) + " |", "|---|---|" + "---|" * len(VIEWS)]
    for g in order:
        rs = groups.get(g)
        if not rs:
            continue
        cells = []
        for v in VIEWS:
            vals = []
            for r in rs:
                if v not in r["robustness"] or (subset is not None and "mse_per_var" not in r["robustness"][v]):
                    continue
                if subset is None:
                    vals.append(r["robustness"][v]["mse"])
                else:
                    pv = np.array(r["robustness"][v]["mse_per_var"])
                    vals.append(float(np.nanmean(pv[subset])))
            cells.append(fmt(vals))
        lines.append(f"| {g} | {len(rs)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase3b_tables.md")
    a = ap.parse_args()
    groups = defaultdict(list)
    for f in glob.glob("results/p3b_*.json") + glob.glob("results/p3phys_*.json") + glob.glob("results/p0_quite_itr_s*.json"):
        if f.endswith(".stuck.json"):
            continue
        r = json.load(open(f))
        mses = [v["mse"] for v in r.get("robustness", {}).values()]
        if len(mses) > 1 and max(mses) - min(mses) < 1e-4:
            continue
        groups[re.sub(r"_s\d+$", "", r["run_tag"])].append(r)

    names = PhysioNet.params
    out = []
    for title, order, n_cap in [("Low data (900 training stays)", ORDER_LOW, 1500), ("Full data (7,200 training stays)", ORDER_FULL, 12000)]:
        out.append(f"## {title}: overall test MSE\n")
        out.append(table(groups, order, "all", None))
        for src in ["llm", "estimated", "hybrid"]:
            A = graph_for(n_cap, src)
            has_parent = np.where(A.sum(0) > 0)[0]
            out.append(f"\n### {title}: MSE on the {len(has_parent)} variables with >= 1 parent in the {src} graph\n")
            out.append(f"variables: {', '.join(names[i] for i in has_parent)}\n")
            out.append(table(groups, order, src, has_parent))
        out.append("")
    md = "\n".join(out)
    open(a.out, "w", encoding="utf-8").write(md)
    print(md)


if __name__ == "__main__":
    main()
