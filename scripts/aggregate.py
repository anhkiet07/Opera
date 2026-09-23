"""Aggregate results/raw/*/results.json into results/aggregated/ and the required Markdown tables.

    python scripts/aggregate.py
Produces results/aggregated/tables.md (Tables 1-4 of OPERA section 37) and results/aggregated/runs.json.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.observation import TEST_CONDITIONS, MATCHED_DENSITY  # noqa: E402

MAIN = ["identity", "random_dropout:0.5", "random_dropout:0.7", "variable_dropout:0.3", "variable_dropout:0.5",
        "temporal_stride:3", "long_gap:0.4", "burst:0.5", "variable_group_dropout:0.3"]
LABEL = {"identity": "clean", "random_dropout:0.5": "rnd .5", "random_dropout:0.7": "rnd .7",
         "variable_dropout:0.3": "var .3", "variable_dropout:0.5": "var .5*", "temporal_stride:3": "stride 3*",
         "long_gap:0.4": "long gap*", "burst:0.5": "burst .5*", "variable_group_dropout:0.3": "var group*"}
ORDER = ["E0_baseline", "E1_augmentation", "E2_distillation", "E3_hard_mining", "E4_opera"]
PRETTY = {"E0_baseline": "E0 baseline (no OPERA)", "E1_augmentation": "E1 + augmentation",
          "E2_distillation": "E2 + distillation", "E3_hard_mining": "E3 + hard mining",
          "E4_opera": "E4 full OPERA"}


def fmt(vals, nd=4):
    vals = [x for x in vals if x is not None and np.isfinite(x)]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{np.mean(vals):.{nd}f}"
    return f"{np.mean(vals):.{nd}f} +- {np.std(vals):.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--out", default="results/aggregated")
    ap.add_argument("--dataset", default=None, help="restrict to one dataset (default: every dataset found)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    runs = defaultdict(list)
    datasets = set()
    for f in sorted(glob.glob(os.path.join(a.raw, "*", "results.json"))):
        r = json.load(open(f))
        ds = r.get("run", {}).get("dataset", "physionet")
        datasets.add(ds)
        if a.dataset and ds != a.dataset:
            continue
        runs[r.get("run", {}).get("experiment", "custom")].append(r)
    if not runs:
        print("no results found in " + a.raw)
        return

    groups = [g for g in ORDER if g in runs] + [g for g in sorted(runs) if g not in ORDER]
    L = ["# OPERA aggregated results", ""]

    L += ["## Table 1 - Robustness to observation-process shift (test MSE, mean +- std over seeds)", "",
          "Conditions marked * never occur in the training observation-process pool.", "",
          "| method | seeds | " + " | ".join(LABEL.get(c, c) for c in MAIN) + " |",
          "|---|---|" + "---|" * len(MAIN)]
    for g in groups:
        rs = runs[g]
        cells = [fmt([r["per_condition"][c]["mse"] for r in rs if c in r["per_condition"]]) for c in MAIN]
        L.append(f"| {PRETTY.get(g, g)} | {len(rs)} | " + " | ".join(cells) + " |")

    L += ["", "## Table 2 - Component contribution", "",
          "| method | seeds | clean MSE | mean shifted MSE | mean degradation | unseen MSE | unseen degradation | worst condition |",
          "|---|---|---|---|---|---|---|---|"]
    for g in groups:
        rs = runs[g]
        s = [r["summary"] for r in rs]
        worst = max(((r["summary"]["worst_mse"], r["summary"]["worst_condition"]) for r in rs), default=(0, "-"))
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            PRETTY.get(g, g), len(rs), fmt([x["clean_mse"] for x in s]), fmt([x["mean_shift_mse"] for x in s]),
            fmt([100 * x["mean_degradation"] for x in s], 1) + " %",
            fmt([x.get("mean_unseen_mse") for x in s]),
            fmt([100 * x.get("mean_unseen_degradation", float("nan")) for x in s], 1) + " %", worst[1]))

    dens = list(MATCHED_DENSITY)[0]
    key, specs = f"matched_density_{dens}_values", MATCHED_DENSITY[dens]
    L += ["", f"## Table 3 - Matched density ({dens}), different structure (test MSE)", "",
          "| method | seeds | " + " | ".join(specs) + " | spread |", "|---|---|" + "---|" * (len(specs) + 1)]
    for g in groups:
        rs = [r for r in runs[g] if key in r["summary"]]
        if not rs:
            continue
        cells = [fmt([r["summary"][key].get(s) for r in rs]) for s in specs]
        spread = fmt([r["summary"][f"matched_density_{dens}_spread"] for r in rs])
        L.append(f"| {PRETTY.get(g, g)} | {len(rs)} | " + " | ".join(cells) + f" | {spread} |")

    L += ["", "## Table 4 - Compute and parameter overhead", "",
          "| method | seeds | params | s / epoch | epochs | peak GPU MB | inference ms / series |",
          "|---|---|---|---|---|---|---|"]
    for g in groups:
        rs = runs[g]
        c = [r["cost"] for r in rs]
        L.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            PRETTY.get(g, g), len(rs), c[0]["n_params"], fmt([x["train_time_per_epoch_s"] for x in c], 1),
            fmt([x["epochs_run"] for x in c], 0), fmt([x["peak_gpu_mem_mb"] for x in c], 0),
            fmt([x["infer_ms_per_series"] for x in c], 3)))

    L += ["", "## All test conditions (test MSE)", "",
          "| method | seeds | " + " | ".join(TEST_CONDITIONS) + " |",
          "|---|---|" + "---|" * len(TEST_CONDITIONS)]
    for g in groups:
        rs = runs[g]
        cells = [fmt([r["per_condition"][c]["mse"] for r in rs if c in r["per_condition"]])
                 for c in TEST_CONDITIONS]
        L.append(f"| {PRETTY.get(g, g)} | {len(rs)} | " + " | ".join(cells) + " |")

    md = "\n".join(L) + "\n"
    stem = f"tables_{a.dataset}" if a.dataset else "tables"
    with open(os.path.join(a.out, f"{stem}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    dump = {g: [{"run": r["run"], "summary": r["summary"], "cost": r["cost"],
                 "per_condition": {c: {"mse": m["mse"], "mae": m["mae"]}
                                   for c, m in r["per_condition"].items()}} for r in runs[g]] for g in groups}
    with open(os.path.join(a.out, f"{stem.replace('tables','runs')}.json"), "w") as f:
        json.dump(dump, f, indent=2)
    print(md)
    if not a.dataset and len(datasets) > 1:
        print(f"note: {len(datasets)} datasets present ({', '.join(sorted(datasets))}); "
              f"use --dataset to get per-dataset tables")


if __name__ == "__main__":
    main()
