"""Masked forecasting metrics (OPERA section 27).

These are the metrics of the existing protocol, kept bit-identical so OPERA numbers are comparable to the
published QuITE setup: errors are averaged over observed target entries, per variable, then over variables that
have at least one observed target.
"""
import torch


def compute_error(truth, pred, mask, func, reduce):
    """truth/pred/mask: (B, L, N) (pred may carry a leading sample axis). Returns (value, per_variable)."""
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    n_dim = pred.size(-1)
    truth = truth.repeat(pred.size(0), 1, 1, 1)
    mask = mask.repeat(pred.size(0), 1, 1, 1)

    if func == "MSE":
        err = (truth - pred) ** 2 * mask
    elif func == "MAE":
        err = torch.abs(truth - pred) * mask
    elif func == "MAPE":
        mask = (truth != 0) * mask
        denom = truth + (truth == 0) * 1e-8
        err = torch.abs(truth - pred) / denom * mask
    else:
        raise ValueError(func)

    err_sum = err.reshape(-1, n_dim).sum(dim=0)
    cnt = torch.count_nonzero(mask.reshape(-1, n_dim), dim=0)
    if reduce == "mean":
        avg = err_sum / (cnt + 1e-8)
        return avg.sum() / torch.count_nonzero(cnt), avg
    if reduce == "sum":
        return err_sum, cnt
    raise ValueError(reduce)


def task_loss(pred, batch):
    """Training loss (masked MSE) plus detached MSE/MAE for logging."""
    target, mask = batch["data_to_predict"], batch["mask_predicted_data"]
    mse, per_var = compute_error(target, pred, mask, "MSE", "mean")
    mae, _ = compute_error(target, pred, mask, "MAE", "mean")
    return mse, {"mse": mse.item(), "mae": mae.item(), "rmse": mse.sqrt().item()}


def aggregate(sums, counts):
    """Turn accumulated per-variable sums/counts into the reported scalars."""
    out = {}
    n_var = torch.count_nonzero(counts["mse"])
    for k in ("mse", "mae"):
        per_var = sums[k] / (counts[k] + 1e-8)
        out[k] = (per_var.sum() / n_var).item()
        out[f"{k}_per_var"] = [float(x) for x in per_var.tolist()]
    n_var_mape = torch.count_nonzero(counts["mape"])
    out["mape"] = ((sums["mape"] / (counts["mape"] + 1e-8)).sum() / n_var_mape).item()
    out["rmse"] = out["mse"] ** 0.5
    return out


def degradation(clean, shifted):
    """Relative degradation (OPERA section 27); positive means worse than clean."""
    return (shifted - clean) / clean if clean else float("nan")


# ------------------------------------------------------------------ classification
def classification_loss(logits, batch):
    """Cross-entropy on the observed labels, plus accuracy for logging."""
    import torch.nn.functional as F
    y = batch["labels"]
    ce = F.cross_entropy(logits, y)
    acc = (logits.argmax(-1) == y).float().mean()
    return ce, {"ce": ce.item(), "acc": acc.item()}


def classification_metrics(logits, labels, n_class):
    """AUROC / AUPRC for binary tasks, macro F1 / precision / recall for multi-class; accuracy always.

    The reported 'mse' key mirrors the forecasting interface so the shared evaluation and aggregation code can
    treat 'lower is better': for classification it holds the cross-entropy.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                                 precision_score, recall_score)
    y = labels.detach().cpu().numpy().ravel()
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    pred = probs.argmax(-1)
    ce = float(F.cross_entropy(logits, labels).item())
    out = {"mse": ce, "ce": ce, "acc": float((y == pred).mean() * 100)}
    if n_class == 2:
        out["auroc"] = float(roc_auc_score(y, probs[:, 1]) * 100)
        out["auprc"] = float(average_precision_score(y, probs[:, 1]) * 100)
        out["mae"] = 100.0 - out["auroc"]          # keep 'lower is better' for the shared tables
    else:
        out["f1"] = float(f1_score(y, pred, average="macro") * 100)
        out["precision"] = float(precision_score(y, pred, average="macro", zero_division=0) * 100)
        out["recall"] = float(recall_score(y, pred, average="macro", zero_division=0) * 100)
        out["mae"] = 100.0 - out["f1"]
    return out
