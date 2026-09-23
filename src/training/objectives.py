"""The OPERA training objective (OPERA section 16).

    L = L_task  +  lambda_d * L_distill

where L_task is averaged over the observation views the batch is trained on (random, mined, or both) and the
distillation term compares the student's prediction under a sparse view with a frozen/EMA teacher's prediction
under a richer observation of the *same history*. No representation-consistency term exists (section 17).
"""
import torch

from src.distillation.losses import prediction_distillation, representation_distillation, teacher_gap
from src.evaluation.metrics import task_loss


def opera_step(model, batch, views, teacher=None, cfg=None, collect_diag=False):
    """One optimisation step's loss.

    views   : list of (data, mask, spec) observation views to train on; empty means train on the batch as-is.
    teacher : optional teacher module (already on the right device), or None.
    cfg     : the run config (uses cfg['distillation']).
    Returns (loss, logs).
    """
    dc = (cfg or {}).get("distillation", {}) or {}
    use_distill = bool(dc.get("enabled", False)) and teacher is not None
    lam = float(dc.get("weight", 1.0))
    dkind = dc.get("loss", "prediction")
    pred_kind = dc.get("prediction_loss", "mse")
    repr_kind = dc.get("representation_loss", "mse")
    need_repr = use_distill and dkind in ("representation", "prediction+representation")

    if not views:
        views = [(batch["observed_data"], batch["observed_mask"], "identity")]

    logs = {}
    task_terms, distill_terms = [], []
    teacher_pred = teacher_repr = None

    if use_distill:
        from src.observation import ObservationProcessGenerator  # noqa: F401  (documentation of the source)
        rich_data, rich_mask = batch["_rich"]
        tbatch = model.with_observation(batch, rich_data, rich_mask)
        with torch.no_grad():
            tout = teacher(tbatch, return_repr=need_repr)
        teacher_pred, teacher_repr = (tout if need_repr else (tout, None))

    for i, (data, mask, spec) in enumerate(views):
        vbatch = model.with_observation(batch, data, mask)
        out = model(vbatch, return_repr=need_repr)
        pred, z = (out if need_repr else (out, None))
        mse, m = task_loss(pred, vbatch)
        task_terms.append(mse)
        if i == 0:
            logs.update({f"train/{k}": v for k, v in m.items()})
            logs["train/view"] = spec

        if use_distill:
            tmask = batch["mask_predicted_data"]
            if dkind in ("prediction", "prediction+representation"):
                distill_terms.append(prediction_distillation(pred, teacher_pred, tmask, pred_kind))
            if dkind in ("representation", "prediction+representation"):
                distill_terms.append(representation_distillation(z, teacher_repr, repr_kind))

    loss = torch.stack(task_terms).mean()
    logs["train/task_loss"] = loss.item()
    logs["train/n_views"] = len(views)
    if distill_terms:
        d = torch.stack(distill_terms).mean()
        loss = loss + lam * d
        logs["train/distill_loss"] = d.item()
        if collect_diag:
            with torch.no_grad():
                p0 = model(model.with_observation(batch, *views[0][:2]))
            logs.update({f"diag/{k}": v for k, v in teacher_gap(teacher_pred, p0, batch).items()})
    logs["train/loss"] = loss.item()
    return loss, logs
