"""Robustness curves from the severity sweep (scripts/severity_sweep.py): MSE vs observation severity, OPERA vs
vanilla, with the severities OPERA trains on shaded.

    python scripts/plot_severity.py [--sweep results/sweep] [--out figures/severity]

Writes, under --out:
  severity_<dataset>.{pdf,png}   rows = backbones, columns = shift family; mean +- std over seeds
  severity_gap.{pdf,png}         OPERA's MSE reduction vs severity, one line per dataset (mean over backbones)
  severity_summary.csv           every point plotted, with the per-seed means and stds
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.severity_sweep import condition  # noqa: E402

# start_x is each ladder's starting point: the clean (identity) process, except for the composed shift, which
# starts from stride 3 and adds variable-group dropout. seen_max = None: no severity of it is in TRAIN_POOL.
FAMILIES = {
    "random_dropout": dict(title="Random dropout", xlabel="drop probability", start_x=0.0, seen_max=0.4),
    "temporal_stride": dict(title="Temporal stride", xlabel="stride k", start_x=1, seen_max=2),
    "variable_dropout": dict(title="Variable dropout", xlabel="variable drop probability", start_x=0.0,
                             seen_max=0.2),
    "composed": dict(title="Composed: stride 3 + variable groups", xlabel="variable-group drop probability",
                     start_x=0.0, seen_max=None),
}
BACKBONES = ["grud", "mtand", "warpformer", "tpatchgnn", "grafiti", "cru"]
METHODS = {"vanilla": dict(label="Vanilla", color="#eb6834"), "opera": dict(label="OPERA", color="#2a78d6")}
DATASET_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEEN_SHADE = "#9aa0a6"
NAME = re.compile(r"bench_(.+)_(" + "|".join(BACKBONES) + r")_(vanilla|opera)_s(\d)$")


def load(sweep_dir):
    """({(dataset, backbone, method): {seed: {condition: mse}}}, {family: severities})"""
    runs, ladders = defaultdict(dict), {}
    for f in glob.glob(os.path.join(sweep_dir, "*.json")):
        rec = json.load(open(f))
        m = NAME.match(rec["run"])
        if m:
            ds, bb, meth, seed = m.groups()
            runs[(ds, bb, meth)][int(seed)] = {c: v["mse"] for c, v in rec["per_condition"].items()}
            ladders.update(rec["ladder"])
    return runs, ladders


def ladder(fam, ladders):
    """Severities of a family, starting point first."""
    return [FAMILIES[fam]["start_x"]] + sorted(ladders[fam])


def cond(fam, x):
    return condition(fam, None if x == FAMILIES[fam]["start_x"] else x)


def series(seeds, fam, xs):
    """(len(seeds), len(xs)) MSE matrix."""
    return np.array([[seeds[s][cond(fam, x)] for x in xs] for s in sorted(seeds)])


def shade_seen(ax, fam, xs):
    lo, hi = FAMILIES[fam]["start_x"], FAMILIES[fam]["seen_max"]
    if hi is None:
        return
    pad = (xs[1] - xs[0]) * 0.5 if len(xs) > 1 else 0.05
    ax.axvspan(lo - pad, hi + pad / 2, color=SEEN_SHADE, alpha=0.12, lw=0)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    ax.tick_params(labelsize=7)


def plot_dataset(ds, runs, fams, out, rows_csv):
    bbs = [b for b in BACKBONES if (ds, b, "vanilla") in runs and (ds, b, "opera") in runs]
    fig, axes = plt.subplots(len(bbs), len(fams), figsize=(2.8 * len(fams), 2.25 * len(bbs)), squeeze=False)
    for i, bb in enumerate(bbs):
        for j, (fam, xs) in enumerate(fams.items()):
            ax = axes[i][j]
            shade_seen(ax, fam, xs)
            mats = {}
            for meth, sty in METHODS.items():
                Y = series(runs[(ds, bb, meth)], fam, xs)
                mats[meth] = Y
                mu, sd = Y.mean(0), Y.std(0)
                ax.fill_between(xs, mu - sd, mu + sd, color=sty["color"], alpha=0.15, lw=0)
                ax.plot(xs, mu, color=sty["color"], lw=2, marker="o", ms=4, label=sty["label"],
                        markeredgecolor="white", markeredgewidth=0.8)
            gap_last = (1 - mats["opera"].mean(0)[-1] / mats["vanilla"].mean(0)[-1]) * 100
            ax.annotate(f"−{gap_last:.0f}%" if gap_last >= 0 else f"+{-gap_last:.0f}%",
                        xy=(xs[-1], mats["opera"].mean(0)[-1]), xytext=(4, -10), textcoords="offset points",
                        fontsize=7, color="#3c4043")
            for k, x in enumerate(xs):
                v, o = mats["vanilla"][:, k], mats["opera"][:, k]
                rows_csv.append(dict(dataset=ds, backbone=bb, family=fam, severity=x,
                                     seen_in_training=FAMILIES[fam]["seen_max"] is not None and x <= FAMILIES[fam]["seen_max"],
                                     mse_vanilla=v.mean(), mse_vanilla_std=v.std(), mse_opera=o.mean(),
                                     mse_opera_std=o.std(), opera_reduction_pct=(1 - o.mean() / v.mean()) * 100,
                                     n_seeds=min(len(v), len(o))))
            style(ax)
            if i == 0:
                ax.set_title(FAMILIES[fam]["title"], fontsize=9)
            if i == len(bbs) - 1:
                ax.set_xlabel(FAMILIES[fam]["xlabel"], fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{bb}\ntest MSE", fontsize=8)
            if fam == "temporal_stride":
                ax.set_xticks(xs)
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color=SEEN_SHADE, alpha=0.25))
    labels.append("severities seen by OPERA in training")
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(ds, fontsize=10, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.955 if len(bbs) > 1 else 0.86))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"severity_{ds}.{ext}"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gap(rows_csv, fams, out):
    datasets = sorted({r["dataset"] for r in rows_csv})
    fig, axes = plt.subplots(1, len(fams), figsize=(3.2 * len(fams), 3.0), sharey=True)
    for j, fam in enumerate(fams):
        ax = axes[j]
        pts = [r for r in rows_csv if r["family"] == fam]
        xs = sorted({r["severity"] for r in pts})
        shade_seen(ax, fam, xs)
        ax.axhline(0, color="#9aa0a6", lw=0.8)
        for k, ds in enumerate(datasets):
            ys = [np.mean([r["opera_reduction_pct"] for r in pts if r["dataset"] == ds and r["severity"] == x])
                  for x in xs]
            ax.plot(xs, ys, color=DATASET_COLORS[k % len(DATASET_COLORS)], lw=2, marker="o", ms=4,
                    markeredgecolor="white", markeredgewidth=0.8, label=ds)
        style(ax)
        ax.set_title(FAMILIES[fam]["title"], fontsize=9)
        ax.set_xlabel(FAMILIES[fam]["xlabel"], fontsize=8)
        if fam == "temporal_stride":
            ax.set_xticks(xs)
    axes[0].set_ylabel("OPERA MSE reduction vs vanilla (%)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color=SEEN_SHADE, alpha=0.25))
    labels.append("seen in training")
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"severity_gap.{ext}"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="results/sweep")
    ap.add_argument("--out", default="figures/severity")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs, ladders = load(a.sweep)
    # a family is plotted only once every run has all of its points (the sweep may still be filling one in)
    fams = {fam: ladder(fam, ladders) for fam in FAMILIES if fam in ladders}
    fams = {fam: xs for fam, xs in fams.items()
            if all(cond(fam, x) in s for r in runs.values() for s in r.values() for x in xs)}
    datasets = sorted({ds for ds, _, _ in runs})
    rows = []
    for ds in datasets:
        plot_dataset(ds, runs, fams, a.out, rows)
    plot_gap(rows, fams, a.out)
    with open(os.path.join(a.out, "severity_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(datasets)} datasets -> {a.out}")


if __name__ == "__main__":
    main()
