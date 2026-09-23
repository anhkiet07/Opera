"""Distillation losses (OPERA section 10).

The primary mechanism is **prediction** distillation: the student's forecast under a sparse observation is
pulled toward the teacher's forecast of the same future under a richer observation of the same history. The
teacher output is always detached by the caller.

Representation distillation exists only as an optional ablation and is deliberately not the main objective:
forcing representations to match between observation views is the formulation that previously failed.
"""
import torch
import torch.nn.functional as F


def prediction_distillation(pred_student, pred_teacher, mask=None, kind="mse"):
    """Task-level distillation on the forecast.

    mask: (B, L_pred, N) target mask. Distillation is applied only where the target exists, so the student is
    never asked to imitate the teacher on entries that carry no supervision.
    """
    t = pred_teacher.detach()
    if mask is None:
        mask = torch.ones_like(pred_student)
    denom = mask.sum().clamp_min(1.0)
    if kind == "mse":
        return (((pred_student - t) ** 2) * mask).sum() / denom
    if kind == "huber":
        return (F.smooth_l1_loss(pred_student, t, reduction="none") * mask).sum() / denom
    if kind == "kl":       # for probabilistic heads: treat the last axis as logits
        logp = F.log_softmax(pred_student, dim=-1)
        q = F.softmax(t, dim=-1)
        return (F.kl_div(logp, q, reduction="none").sum(-1) * mask.any(-1).float()).sum() / \
            mask.any(-1).float().sum().clamp_min(1.0)
    raise ValueError(kind)


def representation_distillation(z_student, z_teacher, kind="mse"):
    """Optional ablation only (OPERA section 10 / 17)."""
    t = z_teacher.detach()
    a = z_student.reshape(z_student.shape[0], -1)
    b = t.reshape(t.shape[0], -1)
    if kind == "mse":
        return F.mse_loss(a, b)
    if kind == "cos":
        return (1.0 - F.cosine_similarity(a, b, dim=-1)).mean()
    raise ValueError(kind)


@torch.no_grad()
def teacher_gap(pred_teacher, pred_student, batch):
    """Diagnostic for OPERA section 39: is the teacher actually better than the student on this batch?"""
    from src.evaluation.metrics import compute_error
    target, mask = batch["data_to_predict"], batch["mask_predicted_data"]
    mse_t, _ = compute_error(target, pred_teacher, mask, "MSE", "mean")
    mse_s, _ = compute_error(target, pred_student, mask, "MSE", "mean")
    return {"teacher_mse": mse_t.item(), "student_mse": mse_s.item(),
            "teacher_advantage": (mse_s - mse_t).item()}
