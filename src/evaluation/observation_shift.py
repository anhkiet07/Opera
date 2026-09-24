"""Evaluation under observation-process shift (OPERA sections 22-23, 30).

`evaluate` runs one pass over a split; `evaluate_conditions` re-observes the *history* of the test inputs
through each named observation process with a fixed seed (so every model sees the same shifted inputs) and
leaves the prediction targets untouched.
"""
import time

import torch

from src.evaluation.metrics import compute_error, aggregate, degradation, classification_metrics
from src.observation import ObservationProcessGenerator, TEST_CONDITIONS, UNSEEN_CONDITIONS, MATCHED_DENSITY


@torch.no_grad()
def evaluate(model, loader, n_batches, next_batch, condition="identity", gen=None, obs_gen=None, seed=12345,
             task="forecasting", n_class=0):
    """Metrics over a split, optionally re-observing the *history* of the inputs through `condition`.

    Forecasting: masked MSE / MAE / MAPE. Classification: cross-entropy, accuracy and AUROC/AUPRC (binary) or
    macro F1 (multi-class); the 'mse' key carries the cross-entropy so the shared tables stay lower-is-better.
    """
    model.eval()
    obs_gen = obs_gen or ObservationProcessGenerator(device="cpu")
    sums = {k: 0 for k in ("mse", "mae", "mape")}
    counts = {k: 0 for k in ("mse", "mae", "mape")}
    n_series, infer_s = 0, 0.0
    logits_all, labels_all = [], []

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

        if task == "classification":
            logits_all.append(pred.detach())
            labels_all.append(batch["labels"].detach())
            continue

        target, mask = batch["data_to_predict"], batch["mask_predicted_data"]
        for key, func in (("mse", "MSE"), ("mae", "MAE"), ("mape", "MAPE")):
            s, c = compute_error(target, pred, mask, func, "sum")
            sums[key] = sums[key] + s
            counts[key] = counts[key] + c

    if task == "classification":
        out = classification_metrics(torch.cat(logits_all), torch.cat(labels_all), n_class)
    else:
        out = aggregate(sums, counts)
    out["condition"] = condition
    out["n_series"] = n_series
    out["infer_ms_per_series"] = 1000.0 * infer_s / max(n_series, 1)
    return out


@torch.no_grad()
def evaluate_conditions(model, loader, n_batches, next_batch, conditions=None, obs_gen=None, seed=12345,
                        log=print, task="forecasting", n_class=0):
    """Evaluate every condition and summarise robustness.

    Returns {'per_condition': {spec: metrics}, 'summary': {...}} where the summary reports mean degradation over
    all shifted conditions and over the unseen-only subset.
    """
    conditions = list(conditions or TEST_CONDITIONS)
    if "identity" not in conditions:
        conditions = ["identity"] + conditions
    per = {}
    for c in conditions:
        per[c] = evaluate(model, loader, n_batches, next_batch, condition=c, obs_gen=obs_gen, seed=seed,
                          task=task, n_class=n_class)
        if log:
            if task == "classification":
                extra = (f"AUROC {per[c]['auroc']:.2f} AUPRC {per[c]['auprc']:.2f}" if n_class == 2
                         else f"F1 {per[c]['f1']:.2f}")
                log(f"  [{c:42s}] CE {per[c]['ce']:.4f}  acc {per[c]['acc']:.2f}  {extra}")
            else:
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
