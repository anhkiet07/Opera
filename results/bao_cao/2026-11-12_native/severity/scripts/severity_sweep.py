"""Severity sweep: re-score finished runs on a graded ladder of observation-process shifts (Figure 3).

    python scripts/severity_sweep.py --glob 'results/raw/bench_*_grud_*_s[0-9]'
    python scripts/severity_sweep.py --glob '...' --out results/sweep

Each run is replayed from its saved checkpoint, with no retraining. The protocol conditions already stored in
results.json are part of the ladder and must reproduce (scripts/reevaluate.replay_mismatch) before anything is
written, so the sweep is scored on byte-identical inputs to the main benchmark. Output goes to
<out>/<run name>.json; results.json is never touched. A run whose output already holds every ladder condition is
skipped; one that lacks some (the ladder grew) is replayed on the protocol plus the missing ones and merged.

The ladder spans the severities OPERA trains on (TRAIN_POOL: random dropout <= 0.4, stride 2, variable
dropout 0.2) and the harsher, unseen ones it is tested on, so one curve shows both regimes.
"""
import argparse
import glob as globmod
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.reevaluate import CONDITIONS as PROTOCOL, load_trainer, replay_mismatch  # noqa: E402

LADDER = {
    "random_dropout": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "temporal_stride": [2, 3, 4],
    "variable_dropout": [0.2, 0.3, 0.4, 0.5],
    # the protocol's composed shift, graded on its unseen component; its x = 0 point is stride 3 alone
    "composed": [0.1, 0.2, 0.3, 0.4, 0.5],
}
COMPOSED_BASE = "temporal_stride:3"


def condition(fam, sev):
    """Condition spec for a ladder point; `sev` = None is the family's starting point."""
    if fam == "composed":
        return COMPOSED_BASE if sev is None else f"{COMPOSED_BASE}+variable_group_dropout:{sev:g}"
    return "identity" if sev is None else f"{fam}:{sev:g}"


CONDITIONS = list(dict.fromkeys(PROTOCOL + [condition(fam, sev) for fam, sevs in LADDER.items() for sev in sevs]))


def sweep(run_dir, out_dir, device):
    out_path = os.path.join(out_dir, os.path.basename(run_dir.rstrip("/")) + ".json")
    done = json.load(open(out_path))["per_condition"] if os.path.exists(out_path) else {}
    missing = [c for c in CONDITIONS if c not in done]
    if not missing:
        return "skip (complete)"
    for f in ("results.json", "model.pt"):
        if not os.path.exists(os.path.join(run_dir, f)):
            return f"no {f}"
    trainer, old = load_trainer(run_dir, device)
    # the protocol conditions are always replayed: they are what proves the inputs match the benchmark
    new = trainer.test(conditions=list(dict.fromkeys(PROTOCOL + missing)))
    bad = replay_mismatch(old, new)
    if bad:
        return bad
    per = dict(done)
    per.update({c: {k: v for k, v in m.items()
                    if isinstance(v, (int, float)) and k not in ("n_series", "infer_ms_per_series")}
                for c, m in new["per_condition"].items()})
    rec = {"run": os.path.basename(run_dir.rstrip("/")), "ladder": LADDER, "per_condition": per}
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=2)
    os.replace(tmp, out_path)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, nargs="+", help="one or more shell globs of run dirs")
    ap.add_argument("--out", default="results/sweep")
    a = ap.parse_args()

    import torch
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)

    dirs = sorted({d for g in a.glob for d in globmod.glob(g) if os.path.isdir(d)})
    print(f"sweeping {len(dirs)} run(s) x {len(CONDITIONS)} conditions on {device}", flush=True)
    bad = 0
    for i, d in enumerate(dirs, 1):
        try:
            msg = sweep(d, a.out, device)
        except Exception as e:
            msg = f"ERROR {type(e).__name__}: {e}"
        if not (msg.startswith(("ok", "skip"))):
            bad += 1
        print(f"[{i}/{len(dirs)}] {os.path.basename(d)}: {msg}", flush=True)
    print(f"\n{len(dirs) - bad} ok, {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
