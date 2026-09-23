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
