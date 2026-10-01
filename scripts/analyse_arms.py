"""Regenerate every arm-comparison number reported for OPERA, from results/raw.

    python scripts/analyse_arms.py                      # -> stdout
    python scripts/analyse_arms.py --out results/analysis_summary.txt
    python scripts/analyse_arms.py --raw /scratch/.../raw

Nothing here is stored by a training run and nothing is hand-entered: every figure is derived from the
per-run `results.json` files, so re-running it after more cells land cannot drift from its source
(BENCHMARK_PROTOCOL section 5).

Conventions, matching the rest of the benchmark:
  primary_error   MSE for forecasting; 1 - AUROC/100 or 1 - acc/100 for classification. Lower is better.
  clean           the `identity` condition
  shifted         mean over the four protocol shift conditions
  paired          every comparison is within (setting, backbone, seed); a run enters only if every arm
                  being compared has that exact cell, so no arm is credited for an easier subset.

TMix is reported separately throughout. It fails intermittently on the classification settings (clean
AUROC 79.06 +/- 15.01 on P19, individual runs at or below chance) and those are failed optimisations,
not a mechanism effect; averaging them in reverses the sign of several conclusions.
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics as st
import sys

PRIMARY = {"P12": "auroc", "P19": "auroc", "PAM": "acc", "P19_hospB": "auroc", "eicu_demo": "auroc"}
CLASSIFICATION = tuple(PRIMARY)
FORECAST_MAIN = ("activity", "physionet", "ushcn_1mo", "ushcn_12mo")
CLEAN = "identity"
SHIFTS = ["random_dropout:0.7", "temporal_stride:3", "variable_dropout:0.5",
          "temporal_stride:3+variable_group_dropout:0.3"]
QUITE = ("itransformer", "patchtst", "tmix")
STABLE = ("itransformer", "patchtst")
IRREGULAR = ("grud", "mtand", "warpformer")
ARM_LABEL = {"vanilla": "baseline", "opera": "OPERA", "operapp": "abl_mining",
             "operapp2": "OPERA++ full", "operapp2_scalar": "OPERA++ scalar"}
RUN_RE = re.compile(
    r"bench_(?P<setting>.+)_(?P<backbone>itransformer|patchtst|tmix|grud|mtand|warpformer)"
    r"_(?P<arm>vanilla|opera|operapp|operapp2|operapp2_scalar)_s(?P<seed>\d+)$")


def primary_error(setting, cond):
    metric = PRIMARY.get(setting, "mse")
    v = cond.get(metric)
    if v is None:
        return None
    return v if metric == "mse" else 1.0 - v / 100.0


def load(raw):
    """-> {(setting, backbone, seed, arm): {'clean':e, 'shift':e, 'raw_clean':v}}"""
    out = {}
    for f in glob.glob(os.path.join(raw, "bench_*", "results.json")):
        m = RUN_RE.match(os.path.basename(os.path.dirname(f)))
        if not m:
            continue
        s = m.group("setting")
        try:
            pc = json.load(open(f, encoding="utf-8")).get("per_condition", {})
        except (json.JSONDecodeError, OSError):
            continue
        if CLEAN not in pc or not all(x in pc for x in SHIFTS):
            continue
        cl = primary_error(s, pc[CLEAN])
        sh = [primary_error(s, pc[x]) for x in SHIFTS]
        if cl is None or any(x is None for x in sh):
            continue
        out[(s, m.group("backbone"), int(m.group("seed")), m.group("arm"))] = {
            "clean": cl, "shift": st.mean(sh),
            "raw_clean": pc[CLEAN].get(PRIMARY.get(s, "mse")),
        }
    return out


def tests(diffs):
    """(median, mean, wins, n, wilcoxon p, paired-t p). Falls back to a sign test without scipy."""
    n = len(diffs)
    wins = sum(1 for x in diffs if x > 0)
    pw = pt = float("nan")
    try:
        from scipy.stats import wilcoxon, ttest_1samp
        if n >= 6 and any(x != 0 for x in diffs):
            pw = float(wilcoxon(diffs).pvalue)
            pt = float(ttest_1samp(diffs, 0.0).pvalue)
    except ImportError:
        from math import comb
        pw = min(1.0, sum(comb(n, i) for i in range(max(wins, n - wins), n + 1)) / 2 ** n * 2)
    return st.median(diffs), st.mean(diffs), wins, n, pw, pt


def paired(D, arms, settings, backbones):
    """Cells where EVERY listed arm exists, so comparisons cannot be credited an easier subset."""
    return sorted({(s, b, sd) for (s, b, sd, a) in D
                   if s in settings and b in backbones
                   and all((s, b, sd, x) in D for x in arms)})


def gain(D, a, ref, k, metric):
    """Relative improvement of `a` over `ref` on primary_error; positive = `a` better."""
    return (D[(*k, ref)][metric] - D[(*k, a)][metric]) / D[(*k, ref)][metric] * 100


def block(w, D, title, arms, settings, backbones, ref="vanilla"):
    ks = paired(D, arms, settings, backbones)
    if not ks:
        return
    w(f"\n{title}  (n={len(ks)} paired runs)")
    w(f"  {'arm':16} {'CLEAN vs ' + ref:>34} {'SHIFTED vs ' + ref:>34}")
    for a in arms:
        if a == ref:
            continue
        dc = [gain(D, a, ref, k, "clean") for k in ks]
        ds = [gain(D, a, ref, k, "shift") for k in ks]
        mc, ac, wc, n, pwc, _ = tests(dc)
        ms, _, ws, _, pws, _ = tests(ds)
        w(f"  {ARM_LABEL.get(a, a):16} med {mc:+6.2f}% mean {ac:+6.2f}% {wc:>3}/{n} p={pwc:<6.3f}"
          f"   med {ms:+7.2f}% {ws:>3}/{n} p={pws:<6.3f}")


def head_to_head(w, D, title, pairs, settings, backbones):
    w(f"\n{title}")
    for a, ref in pairs:
        ks = paired(D, (a, ref), settings, backbones)
        if not ks:
            continue
        for metric in ("clean", "shift"):
            d = [gain(D, a, ref, k, metric) for k in ks]
            m, _, wins, n, pw, pt = tests(d)
            w(f"  {ARM_LABEL.get(a, a):16} vs {ARM_LABEL.get(ref, ref):16} {metric:6} "
              f"med {m:+6.2f}%  {wins:>3}/{n}  wilcoxon p={pw:<6.3f} t p={pt:.3f}")


def per_setting(w, D, title, arms, settings, backbones, ref="vanilla"):
    w(f"\n{title}")
    hdr = f"  {'setting':14} {'n':>3} " + " ".join(f"{ARM_LABEL.get(a, a)[:13]:>14}" for a in arms if a != ref)
    w(hdr)
    for s in sorted(settings):
        ks = paired(D, arms, {s}, backbones)
        if not ks:
            continue
        row = f"  {s:14} {len(ks):>3} "
        for a in arms:
            if a == ref:
                continue
            row += f" clean{st.median([gain(D, a, ref, k, 'clean') for k in ks]):+6.2f}%"
            row += f"/sh{st.median([gain(D, a, ref, k, 'shift') for k in ks]):+7.2f}%"
        w(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/scratch/s223669184/opera_bench/raw")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    D = load(a.raw)
    lines = []
    w = lines.append
    w("OPERA arm comparison — regenerated from results.json, never hand-entered")
    w(f"source: {a.raw}")
    w(f"cells loaded: {len(D)}")
    w("primary_error: MSE (forecasting) or 1-AUROC/1-acc (classification); lower is better")
    w("every comparison is paired within (setting, backbone, seed)")
    w("=" * 100)

    present = {k[3] for k in D}
    w("\n## 1. HEADLINE — OPERA vs baseline, QuITE backbones")
    settings_all = {k[0] for k in D}
    block(w, D, "all settings, all three backbones", ("vanilla", "opera"), settings_all, QUITE)
    block(w, D, "all settings, EXCLUDING TMix", ("vanilla", "opera"), settings_all, STABLE)

    w("\n\n## 2. FORECASTING vs CLASSIFICATION — where the clean cost lives")
    fc = {s for s in settings_all if s not in CLASSIFICATION}
    cls = {s for s in settings_all if s in CLASSIFICATION}
    block(w, D, "forecasting, all backbones", ("vanilla", "opera"), fc, QUITE)
    block(w, D, "forecasting, EXCLUDING TMix", ("vanilla", "opera"), fc, STABLE)
    block(w, D, "classification, all backbones", ("vanilla", "opera"), cls, QUITE)
    block(w, D, "classification, EXCLUDING TMix", ("vanilla", "opera"), cls, STABLE)

    if "operapp2" in present:
        w("\n\n## 3. OPERA++ — conditioning on the observation descriptor r = g(S)")
        w("   hypothesis: if the clean cost is observation-process mixture bias, telling the model which")
        w("   process it is in should remove it.")
        arms = ("vanilla", "opera", "operapp2_scalar", "operapp2")
        block(w, D, "forecasting, all backbones", arms, FORECAST_MAIN, QUITE)
        block(w, D, "forecasting, EXCLUDING TMix", arms, FORECAST_MAIN, STABLE)
        head_to_head(w, D, "head-to-head (all backbones):",
                     (("operapp2", "opera"), ("operapp2_scalar", "opera"),
                      ("operapp2", "operapp2_scalar")), FORECAST_MAIN, QUITE)
        head_to_head(w, D, "head-to-head (EXCLUDING TMix):",
                     (("operapp2", "opera"), ("operapp2_scalar", "opera")), FORECAST_MAIN, STABLE)
        per_setting(w, D, "per setting (all backbones):", arms, FORECAST_MAIN, QUITE)
        w("\n  ablation A vs B: 'OPERA++ scalar' conditions on observation density alone; 'full' adds the")
        w("  structure (gaps, coverage, timestep ratio). random_dropout:0.7 and variable_dropout:0.5 have")
        w("  near-identical density (0.075 vs 0.087) but variable coverage 0.95 vs 0.33, so the full")
        w("  descriptor can separate them and the scalar cannot. A null here is therefore informative.")

    if "operapp" in present:
        w("\n\n## 4. abl_mining (formerly labelled opera++) — harsh-pool candidate mining")
        block(w, D, "all settings, all backbones", ("vanilla", "opera", "operapp"), settings_all, QUITE)
        head_to_head(w, D, "head-to-head:", (("operapp", "opera"),), settings_all, QUITE)

    if present & set(IRREGULAR):
        w("\n\n## 5. IRREGULAR-TS BACKBONES — does OPERA transfer beyond the QuITE encoders?")
        w("   GRU-D (Che et al. 2018), mTAND (Shukla & Marlin 2021), Warpformer (Zhang et al. 2023),")
        w("   re-implemented at this benchmark's shared budget. Absolute levels are NOT comparable to the")
        w("   published tables; the vanilla-vs-OPERA contrast within each backbone is.")
        for b in IRREGULAR:
            block(w, D, f"{b}", ("vanilla", "opera"), settings_all, (b,))

    text = "\n".join(lines) + "\n"
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {a.out} ({len(lines)} lines, {len(D)} cells)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
