"""Backfill robustness bar charts into existing W&B runs from results/*.json.

Usage: python wandb_backfill_plots.py [--project quite] [--glob "results/p*.json"]
Finds each run by its display name (== run_tag), resumes it, logs the bar charts.
"""
import argparse
import glob
import json

import wandb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="quite")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--glob", default="results/p*.json")
    a = ap.parse_args()

    api = wandb.Api()
    path = f"{a.entity}/{a.project}" if a.entity else a.project
    runs = {r.name: r for r in api.runs(path) if r.state == "finished"}

    for f in sorted(glob.glob(a.glob)):
        res = json.load(open(f))
        tag = res["run_tag"]
        rob = res.get("robustness") or res.get("robustness_mean")
        if not rob or tag not in runs:
            print(f"skip {tag}: {'no robustness' if not rob else 'run not found'}")
            continue
        r = runs[tag]
        if "robustness/mse_by_view" in r.summary or "robustness/auroc_by_view" in r.summary:
            print(f"skip {tag}: plots already present")
            continue
        metrics = [m for m in next(iter(rob.values())).keys() if not m.endswith("_std")]
        tbl = wandb.Table(columns=["view"] + metrics, data=[[v] + [rob[v][m] for m in metrics] for v in rob])
        run = wandb.init(project=r.project, entity=r.entity, id=r.id, resume="must")
        run.log({f"robustness/{m}_by_view": wandb.plot.bar(tbl, "view", m, title=f"Test {m.upper()} by observation view")
                 for m in metrics})
        run.finish()
        print(f"backfilled {tag} -> {r.url}")


if __name__ == "__main__":
    main()
