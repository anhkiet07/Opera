"""Evaluation under observation-process shift (OPERA sections 22-23, 30).

`evaluate` runs one pass over a split; `evaluate_conditions` re-observes the *history* of the test inputs
through each named observation process with a fixed seed (so every model sees the same shifted inputs) and
leaves the prediction targets untouched.
"""
import time

import torch

from src.evaluation.metrics import compute_error, aggregate, degradation
from src.observation import ObservationProcessGenerator, TEST_CONDITIONS, UNSEEN_CONDITIONS, MATCHED_DENSITY


@torch.no_grad()
def evaluate(model, loader, n_batches, next_batch, condition="identity", gen=None, obs_gen=None, seed=12345):
    """Mean masked MSE/MAE/MAPE over a split, optionally re-observing the inputs through `condition`."""
    model.eval()
    obs_gen = obs_gen or ObservationProcessGenerator(device="cpu")
    sums = {k: 0 for k in ("mse", "mae", "mape")}
    counts = {k: 0 for k in ("mse", "mae", "mape")}
    n_series, infer_s = 0, 0.0

    for _ in range(n_batches):
        batch = next_batch(loader)
        if batch is None:
            continue
        if condition not in ("identity", "none", "clean"):
            d, m = obs_gen.condition(batch["observed_data"], batch["observed_tp"], batch["observed_mask"],
                                     condition, seed=seed)
            batch = model.with_observation(batch, d, m)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        pred = model(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_s += time.time() - t0
        n_series += pred.shape[0]

        target, mask = batch["data_to_predict"], batch["mask_predicted_data"]
        for key, func in (("mse", "MSE"), ("mae", "MAE"), ("mape", "MAPE")):
            s, c = compute_error(target, pred, mask, func, "sum")
            sums[key] = sums[key] + s
            counts[key] = counts[key] + c

    out = aggregate(sums, counts)
    out["condition"] = condition
    out["n_series"] = n_series
    out["infer_ms_per_series"] = 1000.0 * infer_s / max(n_series, 1)
    return out


@torch.no_grad()
def evaluate_conditions(model, loader, n_batches, next_batch, conditions=None, obs_gen=None, seed=12345,
                        log=print):
    """Evaluate every condition and summarise robustness.

    Returns {'per_condition': {spec: metrics}, 'summary': {...}} where the summary reports mean degradation over
    all shifted conditions and over the unseen-only subset.
    """
    conditions = list(conditions or TEST_CONDITIONS)
    if "identity" not in conditions:
        conditions = ["identity"] + conditions
    per = {}
    for c in conditions:
        per[c] = evaluate(model, loader, n_batches, next_batch, condition=c, obs_gen=obs_gen, seed=seed)
        if log:
            log(f"  [{c:42s}] MSE {per[c]['mse']:.5f}  MAE {per[c]['mae']:.5f}")

    clean = per["identity"]["mse"]
    shifted = [c for c in conditions if c != "identity"]
    unseen = [c for c in shifted if c in UNSEEN_CONDITIONS]
    summary = {
        "clean_mse": clean,
        "mean_shift_mse": float(sum(per[c]["mse"] for c in shifted) / max(len(shifted), 1)),
        "mean_degradation": float(sum(degradation(clean, per[c]["mse"]) for c in shifted) / max(len(shifted), 1)),
        "worst_mse": max((per[c]["mse"] for c in shifted), default=float("nan")),
        "worst_condition": max(shifted, key=lambda c: per[c]["mse"]) if shifted else None,
    }
    if unseen:
        summary["mean_unseen_mse"] = float(sum(per[c]["mse"] for c in unseen) / len(unseen))
        summary["mean_unseen_degradation"] = float(
            sum(degradation(clean, per[c]["mse"]) for c in unseen) / len(unseen))
    # density-vs-structure: conditions with the same expected density but different structure
    for dens, specs in MATCHED_DENSITY.items():
        have = [s for s in specs if s in per]
        if len(have) > 1:
            vals = [per[s]["mse"] for s in have]
            summary[f"matched_density_{dens}_spread"] = float(max(vals) - min(vals))
            summary[f"matched_density_{dens}_values"] = {s: per[s]["mse"] for s in have}
    return {"per_condition": per, "summary": summary}
