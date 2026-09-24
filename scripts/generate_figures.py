"""OPERA figures (OPERA section 36), generated from results/raw + the datasets.

    python scripts/generate_figures.py [--raw results/raw] [--out figures]

Figure 1  framework diagram (trajectory -> observation process -> student/teacher -> prediction)
Figure 2  one trajectory under several observation processes
Figure 3  robustness curve: performance vs observation severity
Figure 4  unseen observation-process generalization
Figure 5  component ablation (E0 -> E4)
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.observation import UNSEEN_CONDITIONS  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
C = {"E0_baseline": "#4C72B0", "E1_augmentation": "#DD8452", "E2_distillation": "#55A868",
     "E3_hard_mining": "#C44E52", "E4_opera": "#8172B3"}
PRETTY = {"E0_baseline": "E0 baseline", "E1_augmentation": "E1 augmentation", "E2_distillation": "E2 +distill",
          "E3_hard_mining": "E3 +mining", "E4_opera": "E4 full OPERA"}
ORDER = list(PRETTY)


def load_runs(raw, dataset=None):
    runs = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(raw, "*", "results.json"))):
        r = json.load(open(f))
        if dataset and r.get("run", {}).get("dataset") != dataset:
            continue
        runs[r.get("run", {}).get("experiment", "custom")].append(r)
    return runs


def mean_std(runs, group, cond, key="mse"):
    vals = [r["per_condition"][cond][key] for r in runs.get(group, []) if cond in r["per_condition"]]
    return (np.mean(vals), np.std(vals)) if vals else (np.nan, 0.0)


# ------------------------------------------------------------------ Figure 1
def fig1(out):
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.axis("off")

    def box(x, y, w, h, text, fc, fs=7.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="#333", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, text=None, style="-|>", ls="-", color="#333"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, lw=0.9, color=color,
                                     linestyle=ls, mutation_scale=9, shrinkA=2, shrinkB=2))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.04, text, ha="center", va="bottom", fontsize=6.5,
                    color=color)

    box(0.00, 0.42, 0.15, 0.16, "underlying\ntrajectory $X$", "#EEEEEE")
    box(0.21, 0.42, 0.19, 0.16, "observation process\n$S \\sim P(S)$", "#FDE9D9")
    arrow(0.15, 0.50, 0.21, 0.50)

    box(0.47, 0.74, 0.19, 0.15, "rich view $S_{rich}(X)$", "#E3F0E3")
    box(0.47, 0.44, 0.19, 0.15, "sparse views $S_k(X)$", "#DCE6F5")
    box(0.47, 0.14, 0.19, 0.15, "hard view $S^*(X)$", "#F7DDDD")
    arrow(0.40, 0.52, 0.47, 0.81)
    arrow(0.40, 0.50, 0.47, 0.51)
    arrow(0.40, 0.48, 0.47, 0.21)

    box(0.72, 0.74, 0.15, 0.15, "teacher\n(frozen)", "#E3F0E3")
    box(0.72, 0.36, 0.15, 0.22, "student\n$f_\\theta$", "#DCE6F5")
    arrow(0.66, 0.81, 0.72, 0.81)
    arrow(0.66, 0.51, 0.72, 0.49)
    arrow(0.66, 0.21, 0.72, 0.41)
    arrow(0.795, 0.74, 0.795, 0.58, "$\\lambda_d\\,\\mathcal{L}_{distill}$", ls="--", color="#55A868")

    box(0.92, 0.40, 0.08, 0.14, "$\\hat Y$", "#FFFFFF")
    arrow(0.87, 0.47, 0.92, 0.47)
    ax.text(0.585, 0.055, "hard mining: search over valid $S$ for the worst case",
            ha="center", fontsize=6.5, color="#C44E52")
    ax.text(0.80, 0.95, "training time only", ha="center", fontsize=6.5, color="#55A868")
    ax.text(0.80, 0.28, "deployment: sparse observation $\\rightarrow$ student $\\rightarrow$ prediction",
            ha="center", fontsize=6.5, color="#333")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(0.0, 1.0)
    fig.savefig(os.path.join(out, "fig1_framework.pdf"))
    print("fig1 done")


# ------------------------------------------------------------------ Figure 2
def _panel_mse(conditions):
    """Real test MSE per condition, from the PhysioNet vanilla benchmark run.

    Never fabricated: if the run is absent the figure is drawn without annotations rather than with made-up
    numbers. The 5-condition benchmark run is preferred; the 15-condition exploratory run of the same config and
    seed is the fallback (their shared conditions agree to 5 decimals).
    """
    import json
    for run in ("results/raw/bench_physionet_itransformer_vanilla_s0/results.json",
                "results/raw/baseline_s0/results.json"):
        if not os.path.exists(run):
            continue
        per = json.load(open(run, encoding="utf-8")).get("per_condition") or {}
        if all(c in per for c in conditions):
            return {c: per[c]["mse"] for c in conditions}, os.path.basename(os.path.dirname(run))
    return None, None


def fig2(out, dataset="physionet"):
    """One ICU stay: the input the model receives, the output it must predict, and what the observation
    process costs.

    Message: same underlying history and same target, different observations, different prediction error. The
    faint trajectory and the whole right-hand region are identical in all four panels by construction; only the
    markers in the left region -- what the model actually receives -- change.

    Variables plotted are heart rate and systolic blood pressure, the two most-measured vitals in this stay. The
    observation-process draw uses the evaluation seed (12345) unchanged in every panel: at that seed variable
    removal happens to drop SysABP and keep HR, which is why one channel's markers vanish in (c). No seed was
    searched for to produce that.
    """
    import torch
    from src.observation import ObservationProcessGenerator
    path = {"physionet": "data/physionet/processed/set-a_0.0.pt"}.get(dataset)
    if not path or not os.path.exists(path):
        print("fig2 skipped (no raw dataset)"); return
    data = torch.load(path, map_location="cpu", weights_only=False)
    HR, SBP = 19, 35                      # 'HR' and 'SysABP' in data/physionet.py params
    HISTORY = 24.0                        # the model sees 0-24 h and forecasts 24-48 h
    best = max(range(len(data)), key=lambda i: (data[i][3][:, HR].sum() + data[i][3][:, SBP].sum()).item())
    _, tt, vals, mask = data[best]
    tt, vals, mask = tt.float(), vals.float(), mask.float()

    hist = tt < HISTORY
    th, vh, mh = tt[hist], vals[hist], mask[hist]
    fut = tt >= HISTORY
    tf, vf, mf = tt[fut], vals[fut], mask[fut]
    Th = th.view(1, -1, 1).expand(1, -1, vh.shape[1])

    views = [("(a) reference", "identity"),
             ("(b) temporal subsampling", "temporal_stride:3"),
             ("(c) variable removal", "variable_dropout:0.5"),
             ("(d) density reduction", "random_dropout:0.7")]
    mses, src = _panel_mse([spec for _, spec in views])
    print(f"fig2 MSE annotations from {src}" if mses else "fig2: no run found, drawing without MSE")

    # This record codes some missing vitals as 0, which is physiologically impossible and would stretch the
    # axis to zero. Crop the VIEW to the central range of the plotted channels; no observation is deleted, and
    # the observation processes act on the full data regardless.
    plotted = torch.cat([vh[mh[:, v] > 0, v] for v in (HR, SBP)] + [vf[mf[:, v] > 0, v] for v in (HR, SBP)])
    plotted = plotted[plotted > 0]
    lo, hi = float(np.percentile(plotted, 1)), float(np.percentile(plotted, 99))
    pad = 0.08 * (hi - lo)

    gen = ObservationProcessGenerator(device="cpu")
    TARGET = "#2f6f4e"                    # one colour for the output, whichever variable it belongs to
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.15), sharey=True)
    for k, (ax, (title, spec)) in enumerate(zip(axes, views)):
        _, m = gen.condition(vh.unsqueeze(0), Th, mh.unsqueeze(0), spec, seed=12345)
        m = m[0]
        ax.axvspan(HISTORY, 48, color="0.92", zorder=0)             # output region
        ax.axvline(HISTORY, color="0.55", lw=0.7, ls="--", zorder=1)
        for var, col, lab in [(HR, C["E0_baseline"], "HR"), (SBP, C["E1_augmentation"], "SysABP")]:
            idx = mh[:, var] > 0
            ax.plot(th[idx], vh[idx, var], color=col, lw=0.7, alpha=0.28, zorder=2)   # same in every panel
            got = m[:, var] > 0
            ax.scatter(th[got], vh[got, var], s=6, color=col, label=lab if k == 0 else None, zorder=4)
            fidx = mf[:, var] > 0                                    # target: identical in every panel
            ax.plot(tf[fidx], vf[fidx, var], color=TARGET, lw=0.7, alpha=0.75, zorder=3,
                    label="forecast target" if (k == 0 and var == HR) else None)
        ax.set_xlim(0, 48)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks([0, 12, 24, 36, 48])
        ax.tick_params(labelsize=6)
        ax.set_xlabel("time (h)", fontsize=7)
        head = title if not mses else title + "\n" + f"MSE {mses[spec]:.4f}"
        ax.set_title(head, fontsize=7.5, linespacing=1.35)
    axes[0].set_ylabel("value", fontsize=7)
    axes[0].legend(frameon=False, fontsize=6.2, loc="upper left", handletextpad=0.3, borderpad=0.1,
                   labelspacing=0.25)
    # name the two regions inside every panel, low enough not to collide with the data
    for ax in axes:
        ax.text(0.25, 0.04, "input", transform=ax.transAxes, ha="center", fontsize=6.2, color="0.45")
        ax.text(0.75, 0.04, "output", transform=ax.transAxes, ha="center", fontsize=6.2, color=TARGET)
    fig.savefig(os.path.join(out, "fig2_observation_shift.pdf"))
    print("fig2 done")


# ------------------------------------------------------------------ Figure 3
def fig3(out, runs):
    families = {
        "random dropout": [("random_dropout:0.3", 0.3), ("random_dropout:0.5", 0.5), ("random_dropout:0.7", 0.7)],
        "variable dropout": [("variable_dropout:0.3", 0.3), ("variable_dropout:0.5", 0.5)],
        "temporal stride": [("temporal_stride:2", 2), ("temporal_stride:3", 3), ("temporal_stride:4", 4)],
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.1))
    for ax, (fam, items) in zip(axes, families.items()):
        for g in [g for g in ORDER if g in runs]:
            xs = [sev for _, sev in items]
            ys = [mean_std(runs, g, c)[0] for c, _ in items]
            es = [mean_std(runs, g, c)[1] for c, _ in items]
            clean = mean_std(runs, g, "identity")[0]
            ax.errorbar([0] + xs, [clean] + ys, yerr=[mean_std(runs, g, "identity")[1]] + es,
                        marker="o", ms=3, lw=1.0, capsize=2, color=C.get(g), label=PRETTY.get(g, g))
        ax.set_title(fam, fontsize=8); ax.set_xlabel("severity")
    axes[0].set_ylabel("test MSE")
    axes[0].legend(frameon=False, fontsize=6)
    fig.savefig(os.path.join(out, "fig3_robustness_curve.pdf"))
    print("fig3 done")


# ------------------------------------------------------------------ Figure 4
def fig4(out, runs):
    seen = ["random_dropout:0.3", "temporal_stride:2", "burst:0.3", "variable_dropout:0.3"]
    unseen = [c for c in UNSEEN_CONDITIONS if any(c in r["per_condition"] for rs in runs.values() for r in rs)]
    groups = [g for g in ORDER if g in runs]
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    w = 0.8 / max(len(groups), 1)
    x = np.arange(3)
    for k, g in enumerate(groups):
        clean = mean_std(runs, g, "identity")[0]
        s = np.mean([mean_std(runs, g, c)[0] for c in seen if not np.isnan(mean_std(runs, g, c)[0])])
        u = np.mean([mean_std(runs, g, c)[0] for c in unseen if not np.isnan(mean_std(runs, g, c)[0])])
        ax.bar(x + (k - (len(groups) - 1) / 2) * w, [clean, s, u], w, color=C.get(g), label=PRETTY.get(g, g))
    ax.set_xticks(x); ax.set_xticklabels(["clean", "seen processes", "unseen processes"], fontsize=7)
    ax.set_ylabel("test MSE"); ax.legend(frameon=False, fontsize=6)
    ax.set_title("Generalization to observation processes absent from training", fontsize=7.5)
    fig.savefig(os.path.join(out, "fig4_unseen_generalization.pdf"))
    print("fig4 done")


# ------------------------------------------------------------------ Figure 5
def fig5(out, runs):
    groups = [g for g in ORDER if g in runs]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.2))
    labels = [PRETTY.get(g, g) for g in groups]
    deg = [100 * np.mean([r["summary"]["mean_degradation"] for r in runs[g]]) for g in groups]
    deg_e = [100 * np.std([r["summary"]["mean_degradation"] for r in runs[g]]) for g in groups]
    clean = [np.mean([r["summary"]["clean_mse"] for r in runs[g]]) for g in groups]
    clean_e = [np.std([r["summary"]["clean_mse"] for r in runs[g]]) for g in groups]
    ax1.bar(labels, deg, yerr=deg_e, color=[C.get(g) for g in groups], capsize=2)
    ax1.set_ylabel("mean degradation under shift (%)"); ax1.tick_params(axis="x", rotation=20, labelsize=6.5)
    ax2.bar(labels, clean, yerr=clean_e, color=[C.get(g) for g in groups], capsize=2)
    ax2.set_ylabel("clean test MSE"); ax2.tick_params(axis="x", rotation=20, labelsize=6.5)
    ax1.set_title("robustness", fontsize=8); ax2.set_title("in-distribution cost", fontsize=8)
    fig.savefig(os.path.join(out, "fig5_ablation.pdf"))
    print("fig5 done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--dataset", default="physionet")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = load_runs(a.raw, dataset=a.dataset)
    fig1(a.out)
    fig2(a.out, a.dataset)
    if runs:
        fig3(a.out, runs); fig4(a.out, runs); fig5(a.out, runs)
    else:
        print("no runs for figures 3-5")


if __name__ == "__main__":
    main()
