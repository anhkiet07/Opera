import time

import torch
from tqdm import tqdm

import utils
import sampling_views as sv


def compute_error(truth, pred_y, mask, func, reduce, norm_dict=None):
    """Compute MSE / MAE / MAPE per variable.

    Returns either (mean_error, per_variable_mean) when reduce='mean',
    or (per_variable_sum, mask_count) when reduce='sum'.
    """
    if pred_y.dim() == 3:
        pred_y = pred_y.unsqueeze(0)
    _, _, _, n_dim = pred_y.size()

    truth = truth.repeat(pred_y.size(0), 1, 1, 1)
    mask = mask.repeat(pred_y.size(0), 1, 1, 1)

    if func == "MSE":
        error = (truth - pred_y) ** 2 * mask
    elif func == "MAE":
        error = torch.abs(truth - pred_y) * mask
    elif func == "MAPE":
        if norm_dict is None:
            mask = (truth != 0) * mask
            truth_div = truth + (truth == 0) * 1e-8
            error = torch.abs(truth - pred_y) / truth_div * mask
        else:
            scale = norm_dict["data_max"] - norm_dict["data_min"]
            offset = norm_dict["data_min"]
            truth_rescale = truth * scale + offset
            pred_rescale = pred_y * scale + offset
            mask = (truth_rescale != 0) * mask
            truth_rescale_div = truth_rescale + (truth_rescale == 0) * 1e-8
            error = torch.abs(truth_rescale - pred_rescale) / truth_rescale_div * mask
    else:
        raise ValueError(f"Unknown error function: {func}")

    error_var_sum = error.reshape(-1, n_dim).sum(dim=0)
    mask_count = torch.count_nonzero(mask.reshape(-1, n_dim), dim=0)

    if reduce == "mean":
        error_var_avg = error_var_sum / (mask_count + 1e-8)
        n_avai_var = torch.count_nonzero(mask_count)
        return error_var_avg.sum() / n_avai_var, error_var_avg
    elif reduce == "sum":
        return error_var_sum, mask_count
    raise ValueError(f"Unknown reduce: {reduce}")


def _forward(model, batch_dict, return_repr=False):
    return model.forecasting(
        batch_dict["tp_to_predict"], batch_dict["observed_data"],
        batch_dict["observed_tp"], batch_dict["observed_mask"],
        **({"return_repr": True} if return_repr else {}),
    )


def _task_loss(pred_y, batch_dict):
    target = batch_dict["data_to_predict"]
    target_mask = batch_dict["mask_predicted_data"]
    mse, mse_var_avg = compute_error(target, pred_y, mask=target_mask, func="MSE", reduce="mean")
    mae, _ = compute_error(target, pred_y, mask=target_mask, func="MAE", reduce="mean")
    return mse, mae, mse_var_avg


def compute_all_losses(model, batch_dict, args=None, gen=None):
    """Single training step: forward + MSE loss (+ optional sampling-consistency).

    Plain QuITE (default): loss = MSE on the original observation pattern.

    Sampling-invariant training (args.train_view != 'none'):
        n_views observation views of the same batch are drawn; the task loss is
        averaged over the views and, when args.consistency_lambda > 0, a
        consistency term D(z_A, z_B) between the two views' representations
        (or predictions, args.consistency_target='pred') is added.
    """
    train_view = getattr(args, "train_view", "none") if args is not None else "none"
    lam = getattr(args, "consistency_lambda", 0.0) if args is not None else 0.0
    n_views = getattr(args, "n_views", 1) if args is not None else 1

    if train_view == "none" and lam <= 0:
        pred_y = _forward(model, batch_dict)
        mse, mae, mse_var_avg = _task_loss(pred_y, batch_dict)
        return {"loss": mse, "mse": mse.item(), "rmse": torch.sqrt(mse).item(),
                "mae": mae.item(), "mse_var_avg": mse_var_avg}

    n_views = max(n_views, 2 if lam > 0 else 1)
    views = [sv.view_batch(batch_dict, train_view, gen) for _ in range(n_views)]
    preds, reprs, mses, maes = [], [], [], []
    for v in views:
        pred_y, z = _forward(model, v, return_repr=True)
        mse, mae, mse_var_avg = _task_loss(pred_y, v)
        preds.append(pred_y); reprs.append(z); mses.append(mse); maes.append(mae)

    task = torch.stack(mses).mean()
    out = {"mse": task.item(), "rmse": torch.sqrt(task).item(),
           "mae": torch.stack(maes).mean().item(), "mse_var_avg": mse_var_avg}
    loss = task
    if lam > 0:
        target = getattr(args, "consistency_target", "repr")
        dist = getattr(args, "consistency_dist", "cos")
        if target == "repr":
            cons = sv.consistency_loss(reprs[0], reprs[1], dist)
        else:  # prediction-level consistency (masked to predicted positions)
            m = batch_dict["mask_predicted_data"]
            cons = sv.consistency_loss(preds[0] * m, preds[1] * m, "mse")
        loss = task + lam * cons
        out["consistency"] = cons.item()
    if n_views >= 2:
        out.update(sv.repr_diagnostics(reprs[0], reprs[1]))
    out["loss"] = loss
    return out


def evaluation(model, dataloader, n_batches, view="none", gen=None):
    """Full-pass evaluation; returns aggregated MSE/MAE/RMSE/MAPE.

    `view` applies a sampling view to the *observed* inputs only (targets are
    untouched), which is how cross-sampling robustness is measured.
    """
    totals = {"loss": 0, "mse": 0, "mae": 0, "rmse": 0, "mape": 0}
    n_eval_samples = 0
    n_eval_samples_mape = 0
    n_series = 0
    infer_time = 0.0

    for _ in tqdm(range(n_batches)):
        batch_dict = utils.get_next_batch(dataloader)
        if batch_dict is None:
            continue
        batch_dict = sv.view_batch(batch_dict, view, gen)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        st = time.time()
        pred_y = _forward(model, batch_dict)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_time += time.time() - st
        n_series += pred_y.shape[0]

        target = batch_dict["data_to_predict"]
        target_mask = batch_dict["mask_predicted_data"]

        se_var_sum, mask_count = compute_error(target, pred_y, mask=target_mask, func="MSE", reduce="sum")
        ae_var_sum, _ = compute_error(target, pred_y, mask=target_mask, func="MAE", reduce="sum")
        ape_var_sum, mask_count_mape = compute_error(target, pred_y, mask=target_mask, func="MAPE", reduce="sum")

        totals["loss"] += se_var_sum
        totals["mse"] += se_var_sum
        totals["mae"] += ae_var_sum
        totals["mape"] += ape_var_sum
        n_eval_samples += mask_count
        n_eval_samples_mape += mask_count_mape

    n_avai_var = torch.count_nonzero(n_eval_samples)
    n_avai_var_mape = torch.count_nonzero(n_eval_samples_mape)

    var_list = totals["loss"] / (n_eval_samples + 1e-8)
    totals["loss"] = (totals["loss"] / (n_eval_samples + 1e-8)).sum() / n_avai_var
    totals["mse"] = (totals["mse"] / (n_eval_samples + 1e-8)).sum() / n_avai_var
    totals["mae"] = (totals["mae"] / (n_eval_samples + 1e-8)).sum() / n_avai_var
    totals["rmse"] = torch.sqrt(totals["mse"])
    totals["mape"] = (totals["mape"] / (n_eval_samples_mape + 1e-8)).sum() / n_avai_var_mape

    for k, v in totals.items():
        if isinstance(v, torch.Tensor):
            totals[k] = v.item()
    totals["infer_time_s"] = infer_time
    totals["n_series"] = n_series
    totals["infer_ms_per_series"] = 1000.0 * infer_time / max(n_series, 1)
    return totals, var_list
