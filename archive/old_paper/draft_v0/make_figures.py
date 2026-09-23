"""Generate the paper figures from the repo's results/ and data/ (run from the repo root):

    .venv/Scripts/python "Robust Representation Learning for Irregular Time Series under Observation-Process Shift/make_figures.py"

Outputs PDFs into <paper folder>/figures/.
  fig1_observation_shift.pdf : one PhysioNet trajectory under four observation processes
  fig3_robustness.pdf        : test MSE vs. observation view for the main configurations (PhysioNet)
  fig3b_consistency.pdf      : consistency-loss sweep vs. augmentation-only
  fig4_structural_prior.pdf  : synthetic graph + target-variable error under targeted removals
"""
import os, sys, json, glob, re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sampling_views as sv

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
C = {"plain": "#4C72B0", "aug": "#DD8452", "adapt": "#55A868", "both": "#8172B3", "grey": "#999999"}


def load_groups(pattern):
    g = defaultdict(list)
    for f in glob.glob(pattern):
        if f.endswith(".stuck.json"):
            continue
        r = json.load(open(f))
        mses = [v["mse"] for v in r.get("robustness", {}).values()]
        if len(mses) > 1 and max(mses) - min(mses) < 1e-4:   # constant predictor (plateau run)
            continue
        g[re.sub(r"_s\d+$", "", r["run_tag"])].append(r)
    return g


# ---------------------------------------------------------------- Figure 1
def fig1():
    data = torch.load("data/physionet/processed/set-a_0.0.pt", map_location="cpu", weights_only=False)
    HR, MAP, TEMP = 19, 23, 36
    best = max(range(len(data)), key=lambda i: (data[i][3][:, HR].sum() + data[i][3][:, MAP].sum()).item())
    _, tt, vals, mask = data[best]
    tt = tt.float(); vals = vals.float(); mask = mask.float()
    keep24 = tt < 24
    tt, vals, mask = tt[keep24], vals[keep24], mask[keep24]
    T = tt.view(1, -1, 1).expand(1, -1, vals.shape[1])
    views = [("(a) original", "none"), ("(b) random sparse, 50%", "random:0.5"),
             ("(c) bursty gaps, 50%", "bursty:0.5"), ("(d) variable-specific", "dropvars:23")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.7), sharey=True)
    for ax, (title, spec) in zip(axes, views):
        g = sv.make_generator(7, "cpu")
        _, m = sv.apply_view(vals.unsqueeze(0), T, mask.unsqueeze(0), spec, g)
        m = m[0]
        for var, col, lab in [(HR, C["plain"], "HR"), (MAP, C["aug"], "MAP")]:
            idx = (mask[:, var] > 0)
            ax.plot(tt[idx], vals[idx, var], color=col, lw=0.6, alpha=0.25)
            idx2 = (m[:, var] > 0)
            ax.scatter(tt[idx2], vals[idx2, var], s=6, color=col, label=lab, zorder=3)
        ax.set_title(title, fontsize=7.5); ax.set_xlabel("time (h)"); ax.set_xlim(0, 24)
    axes[0].set_ylabel("value"); axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.savefig(os.path.join(OUT, "fig1_observation_shift.pdf"))
    print("fig1 done (record", best, ")")


# ---------------------------------------------------------------- Figure 3
VIEWS = ["none", "random:0.3", "random:0.5", "random:0.7", "bursty:0.3", "bursty:0.5", "variable:0.3",
         "variable:0.5", "stride:2", "stride:3"]
LABELS = ["ID", "rnd .3", "rnd .5", "rnd .7", "burst .3", "burst .5", "var .3", "var .5*", "stride 2*", "stride 3*"]


def stat(rs, v):
    x = [r["robustness"][v]["mse"] for r in rs if v in r["robustness"]]
    return (np.mean(x), np.std(x), len(x)) if x else (np.nan, 0, 0)


def fig3():
    g = load_groups("results/p[012]_*.json")
    cfgs = [("p0_quite_itr", "no augmentation (plain)", C["plain"]),
            ("p2_adaptive_full", "observation-aware queries", C["adapt"]),
            ("p1_aug_only", "sampling-process augmentation", C["aug"]),
            ("p2_adaptive_full_mixed", "augmentation + observation-aware queries", C["both"])]
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    w = 0.2; x = np.arange(len(VIEWS))
    for k, (tag, lab, col) in enumerate(cfgs):
        m = [stat(g[tag], v) for v in VIEWS]
        ax.bar(x + (k - 1.5) * w, [a[0] for a in m], w, yerr=[a[1] for a in m], color=col, label=lab,
               error_kw=dict(lw=0.6, capsize=1.5))
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=7); ax.set_ylabel("test MSE")
    ax.set_title("PhysioNet forecasting: test inputs re-observed through different observation processes (* = unseen in training)", fontsize=7.5)
    ax.legend(frameon=False, fontsize=7, ncol=1, loc="upper left")
    fig.savefig(os.path.join(OUT, "fig3_robustness.pdf")); print("fig3 done")

    # consistency sweep
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    sweep = [("p1_aug_only", "aug-only ($\\lambda=0$)"), ("p1_cons_lam0.01", "$\\lambda=0.01$"),
             ("p1_cons_lam0.1", "$\\lambda=0.1$"), ("p1_cons_lam1.0", "$\\lambda=1$"), ("p1_cons_pred_lam0.1", "pred., $\\lambda=0.1$")]
    vs = ["none", "random:0.7", "variable:0.3"]
    x = np.arange(len(vs)); w = 0.15
    for k, (tag, lab) in enumerate(sweep):
        m = [stat(g[tag], v) for v in vs]
        ax.bar(x + (k - 2) * w, [a[0] for a in m], w, yerr=[a[1] for a in m], label=lab, error_kw=dict(lw=0.6, capsize=1.5))
    ax.set_xticks(x); ax.set_xticklabels(["ID", "rnd .7", "var .3"]); ax.set_ylabel("test MSE")
    ax.set_title("Representation-consistency weight", fontsize=7.5); ax.legend(frameon=False, fontsize=6)
    fig.savefig(os.path.join(OUT, "fig3b_consistency.pdf")); print("fig3b done")


# ---------------------------------------------------------------- Figure 4
def fig4():
    from data.synthetic import EDGES, N_VARS, TARGET
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[0.9, 1.4, 1.1]))
    pos = {0: (0, 1), 1: (1, 1), 2: (2, 1), 3: (1, 0), 4: (2, 2), 5: (3.2, 1.5), 6: (4.2, 1.5), 7: (3.2, 0.5), 8: (4.2, 0.5), 9: (5, 1)}
    for j, i in EDGES:
        (x0, y0), (x1, y1) = pos[j], pos[i]
        ax1.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.9, color="k", shrinkA=8, shrinkB=8))
    for n, (x, y) in pos.items():
        col = C["aug"] if n == TARGET else (C["plain"] if n in (0, 1, 4) else C["grey"])
        ax1.scatter([x], [y], s=260, color=col, zorder=3, edgecolor="k", lw=0.5)
        ax1.text(x, y, "Y" if n == TARGET else str(n), ha="center", va="center", fontsize=7, color="w", zorder=4)
    ax1.set_xlim(-0.6, 5.6); ax1.set_ylim(-0.5, 2.5); ax1.axis("off")
    ax1.set_title("(a) structural dynamics (target Y)", fontsize=7.5)

    g = load_groups("results/p3_*.json")
    cfgs = [("p3_quite", "no bias", C["plain"]), ("p3_soft_true_lam2", "oracle graph", C["aug"]),
            ("p3_soft_shuffled_lam2", "shuffled graph", C["adapt"]), ("p3_soft_random_lam2", "random graph", C["both"]),
            ("p3_soft_identity_lam2", "identity", C["grey"])]
    vs = ["none", "dropvars:5-6-7-8-9", "dropvars:1-4", "dropvars:0-3"]
    labs = ["clean", "non-causal\nremoved", "parents\nremoved", "anc.+sibling\nremoved"]
    x = np.arange(len(vs)); w = 0.16
    for k, (tag, lab, col) in enumerate(cfgs):
        rs = g.get(tag, [])
        m = []
        for v in vs:
            xs = [r["robustness"][v]["mse_per_var"][TARGET] for r in rs if v in r["robustness"]]
            m.append((np.mean(xs), np.std(xs)) if xs else (np.nan, 0))
        ax2.bar(x + (k - 2) * w, [a[0] for a in m], w, yerr=[a[1] for a in m], color=col, label=lab, error_kw=dict(lw=0.6, capsize=1.5))
    ax2.set_xticks(x); ax2.set_xticklabels(labs, fontsize=6.5); ax2.set_ylabel("test MSE on Y"); ax2.set_ylim(0, 0.075)
    ax2.set_title("(b) soft bias, 2,400 training traj.", fontsize=7.5); ax2.legend(frameon=False, fontsize=6, ncol=2, loc="upper left")

    # low-data regime, hard masks
    gl = load_groups("results/p3low_*.json")
    cfgs3 = [("p3low_quite", "no mask", C["plain"]), ("p3low_hard_true", "hard mask, oracle", C["aug"]),
             ("p3low_hard_shuffled", "hard mask, shuffled", C["adapt"]), ("p3low_hard_identity", "hard mask, identity", C["grey"]),
             ("p3low_hard_random", "hard mask, random", C["both"])]
    vs3 = ["none", "dropvars:5-6-7-8-9", "dropvars:1-4"]
    labs3 = ["clean", "non-causal\nremoved", "parents\nremoved"]
    x = np.arange(len(vs3)); w = 0.16
    for k, (tag, lab, col) in enumerate(cfgs3):
        rs = gl.get(tag, [])
        m = []
        for v in vs3:
            xs = [r["robustness"][v]["mse_per_var"][TARGET] for r in rs if v in r["robustness"]]
            m.append((np.mean(xs), np.std(xs)) if xs else (np.nan, 0))
        ax3.bar(x + (k - 2) * w, [a[0] for a in m], w, yerr=[a[1] for a in m], color=col, label=lab, error_kw=dict(lw=0.6, capsize=1.5))
    ax3.set_xticks(x); ax3.set_xticklabels(labs3, fontsize=6.5); ax3.set_ylim(0, 0.105)
    ax3.set_title("(c) hard mask, 900 training traj.", fontsize=7.5); ax3.legend(frameon=False, fontsize=5.8, ncol=1, loc="upper left")
    fig.savefig(os.path.join(OUT, "fig4_structural_prior.pdf")); print("fig4 done")


if __name__ == "__main__":
    fig1(); fig3(); fig4()
