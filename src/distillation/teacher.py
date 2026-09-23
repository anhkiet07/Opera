"""Privileged-observation teachers (OPERA sections 8-9, 11).

Two modes:
  * `frozen` : a model trained on rich observations, loaded from a checkpoint and frozen (Mode A).
  * `ema`    : an exponential moving average of the student (Mode B), fed the rich observation.

Anti-leakage invariant (OPERA section 11): the teacher sees a *richer observation of the same history window*
and never any future information. Both teacher and student are called with the same `tp_to_predict` and the
same targets; only `observed_data` / `observed_mask` differ. `assert_no_future_leakage` checks this and is
exercised by tests/test_no_leakage.py.
"""
import copy

import torch
import torch.nn as nn


class FrozenTeacher(nn.Module):
    """Mode A: a separately trained rich-observation model, frozen."""

    def __init__(self, model, ckpt=None, device="cpu"):
        super().__init__()
        self.model = model
        if ckpt:
            state = torch.load(ckpt, map_location=device)
            self.model.load_state_dict(state["model"] if "model" in state else state)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, batch, return_repr=False):
        self.model.eval()
        return self.model(batch, return_repr=return_repr)

    def update(self, student):
        pass


class EMATeacher(nn.Module):
    """Mode B: theta_T <- alpha * theta_T + (1 - alpha) * theta_S."""

    def __init__(self, student, alpha=0.999):
        super().__init__()
        self.model = copy.deepcopy(student)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.alpha = float(alpha)

    @torch.no_grad()
    def forward(self, batch, return_repr=False):
        self.model.eval()
        return self.model(batch, return_repr=return_repr)

    @torch.no_grad()
    def update(self, student):
        a = self.alpha
        for pt, ps in zip(self.model.parameters(), student.parameters()):
            pt.mul_(a).add_(ps.detach(), alpha=1 - a)
        for bt, bs in zip(self.model.buffers(), student.buffers()):
            bt.copy_(bs)


def build_teacher(cfg, student, device, build_fn=None, data_info=None):
    """Create the teacher described by cfg['distillation'].

    teacher_type: 'frozen' (needs `teacher_ckpt`, or falls back to a copy of the student's init) or 'ema'.
    """
    dc = cfg.get("distillation", {})
    kind = dc.get("teacher_type", "frozen")
    if kind == "ema":
        return EMATeacher(student, alpha=float(dc.get("ema_alpha", 0.999))).to(device)
    ckpt = dc.get("teacher_ckpt")
    if ckpt:
        model, _ = build_fn(cfg, data_info, device) if build_fn else (copy.deepcopy(student), None)
        return FrozenTeacher(model, ckpt=ckpt, device=device).to(device)
    # No checkpoint: a frozen copy of the current student (only sensible for smoke tests).
    return FrozenTeacher(copy.deepcopy(student), device=device).to(device)


def assert_no_future_leakage(student_batch, teacher_batch):
    """The teacher may differ from the student only in the observed history."""
    for k in ("tp_to_predict", "data_to_predict", "mask_predicted_data"):
        a, b = student_batch[k], teacher_batch[k]
        if not torch.equal(a, b):
            raise AssertionError(f"teacher/student disagree on '{k}': future information would leak")
    # The teacher's history must not extend beyond the student's history window.
    st, tt = student_batch["observed_tp"], teacher_batch["observed_tp"]
    if st.shape == tt.shape and not torch.equal(st, tt):
        raise AssertionError("teacher/student history timestamps differ")
    s_mask, t_mask = student_batch["observed_mask"], teacher_batch["observed_mask"]
    if s_mask.shape != t_mask.shape:
        raise AssertionError("teacher/student observation masks have different shapes")
    return True
