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
    """One ICU stay: what the model is given, what it must predict, and what it actually predicts.

    Message: same underlying history and same target, different observations, different forecast. The input
    markers are the only thing that changes between panels; the ground truth never does. The model's own
    forecast is overlaid so the reader can see the prediction drift away from the target as the observation
    process degrades, and the panel heading gives the test MSE of that condition.

    Runs the trained vanilla checkpoint on CPU; no GPU is touched, so this is safe to render during a sweep.
    """
    import torch
    if dataset != "physionet":
        print("fig2 skipped (physionet only)"); return
    ckpt_dir = "results/raw/bench_physionet_itransformer_vanilla_s0"
    ckpt = os.path.join(ckpt_dir, "model.pt")
    if not os.path.exists(ckpt):
        print("fig2 skipped (no vanilla checkpoint)"); return

    # The processed dataset was serialised with CUDA tensors; force CPU so nothing lands on the card.
    _load = torch.load
    torch.load = lambda *a, **k: _load(*a, **{**k, "map_location": "cpu"})
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
        from train import load_yaml
        from src.datasets.registry import load as load_data, next_batch
        from src.models.base import build_model
        from src.observation import ObservationProcessGenerator

        cfg = load_yaml("configs/baseline.yaml")
        data, info = load_data(cfg, "cpu", seed=0)
        model, _ = build_model(cfg, info, "cpu")
        model.load_state_dict(torch.load(ckpt)["model"])
        model.eval()
        batch = next_batch(data["test_dataloader"])
    finally:
        torch.load = _load

    HR = 19                       # 'HR' in data/physionet.py params
    HRS = 48.0                    # times are normalised by the 48 h window
    # the sample with the most heart-rate coverage in both windows
    obs_n = (batch["observed_mask"][..., HR] > 0).flatten(1).sum(1)
    tgt_n = (batch["mask_predicted_data"][..., HR] > 0).sum(1)
    i = int(torch.argmin(-(obs_n + tgt_n)))

    views = [("(a) reference", "identity"),
             ("(b) temporal subsampling", "temporal_stride:3"),
             ("(c) variable removal", "variable_dropout:0.5"),
             ("(d) density reduction", "random_dropout:0.7")]
    mses, src = _panel_mse([spec for _, spec in views])
    print(f"fig2 MSE from {src}" if mses else "fig2: no run found, drawing without MSE")

    gen = ObservationProcessGenerator(device="cpu")
    OBS = C["E0_baseline"]        # input observations AND ground truth: one colour, split by the shading
    PRED = "#c0504d"              # the model's own forecast

    tt_h = (batch["observed_tp"][i, ..., HR].reshape(-1) * HRS)
    v_h = batch["observed_data"][i, ..., HR].reshape(-1)
    m_h = batch["observed_mask"][i, ..., HR].reshape(-1) > 0
    tt_f = batch["tp_to_predict"][i] * HRS
    v_f = batch["data_to_predict"][i, :, HR]
    m_f = batch["mask_predicted_data"][i, :, HR] > 0

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.95), sharey=True)
    for k, (ax, (title, spec)) in enumerate(zip(axes, views)):
        d, m = gen.condition(batch["observed_data"], batch["observed_tp"], batch["observed_mask"],
                             spec, seed=12345)
        with torch.no_grad():
            pred = model(model.with_observation(batch, d, m))
        keep = m[i, ..., HR].reshape(-1) > 0

        ax.axvspan(24, 48, color="0.93", zorder=0)
        ax.axvline(24, color="0.6", lw=0.7, ls="--", zorder=1)
        # everything the stay contains, faint: identical in every panel
        ax.plot(tt_h[m_h], v_h[m_h], color=OBS, lw=0.7, alpha=0.25, zorder=2,
                label="recorded, not given to model" if k == 0 else None)
        ax.scatter(tt_h[keep], v_h[keep], s=7, color=OBS, zorder=4,
                   label="observed / ground truth" if k == 0 else None)
        ax.plot(tt_f[m_f], v_f[m_f], color=OBS, lw=0.9, zorder=3)          # ground truth, same colour
        ax.plot(tt_f[m_f], pred[i, m_f, HR], color=PRED, lw=1.0, ls="--", zorder=5,
                label="model forecast" if k == 0 else None)
        ax.set_xlim(0, 48); ax.set_xticks([0, 12, 24, 36, 48])
        ax.tick_params(labelsize=6)
        ax.set_xlabel("time (h)", fontsize=7)
        head = title + (chr(10) + f"MSE {mses[spec]:.4f}" if mses else "")
        ax.set_title(head, fontsize=7.5, linespacing=1.35)
        if int(keep.sum()) == 0:
            ax.text(0.24, 0.80, "HR removed" + chr(10) + f"0 of {int(m_h.sum())} kept",
                    transform=ax.transAxes, ha="center", va="center", fontsize=6.2, color=PRED)
        ax.text(0.24, 0.045, "input", transform=ax.transAxes, ha="center", fontsize=6.2, color="0.45")
        ax.text(0.76, 0.045, "output", transform=ax.transAxes, ha="center", fontsize=6.2, color="0.45")
    axes[0].set_ylabel("heart rate (norm.)", fontsize=7)
    # legend below the axes, clear of the x labels
    fig.subplots_adjust(bottom=0.34)
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.02),
               handlelength=1.6, columnspacing=1.6)
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
