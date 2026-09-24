"""Anti-leakage tests (OPERA section 11): the teacher may see a richer *history*, never the future.

Also covers the invariant that every mechanism (augmentation, mining, evaluation conditions) alters only the
observed history and leaves the prediction targets untouched.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.distillation.losses import prediction_distillation  # noqa: E402
from src.distillation.teacher import assert_no_future_leakage  # noqa: E402
from src.models.base import BaseIrregularEncoder  # noqa: E402
from src.observation import ObservationProcessGenerator  # noqa: E402

TARGET_KEYS = ("tp_to_predict", "data_to_predict", "mask_predicted_data")


def make_batch(B=4, L=20, N=5, Lp=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    tp = torch.rand(B, L, generator=g).sort(dim=1).values
    mask = (torch.rand(B, L, N, generator=g) < 0.6).float()
    return {
        "observed_data": torch.randn(B, L, N, generator=g) * mask,
        "observed_tp": tp,
        "observed_mask": mask,
        "tp_to_predict": torch.rand(B, Lp, generator=g).sort(dim=1).values + 1.0,
        "data_to_predict": torch.randn(B, Lp, N, generator=g),
        "mask_predicted_data": (torch.rand(B, Lp, N, generator=g) < 0.5).float(),
    }


def test_with_observation_leaves_targets_untouched():
    b = make_batch()
    gen = ObservationProcessGenerator(seed=0)
    d, m = gen.condition(b["observed_data"], b["observed_tp"], b["observed_mask"], "random_dropout:0.5")
    b2 = BaseIrregularEncoder.with_observation(b, d, m)
    for k in TARGET_KEYS:
        assert torch.equal(b[k], b2[k]), f"{k} was modified"
    assert not torch.equal(b["observed_mask"], b2["observed_mask"])


def test_rich_view_is_history_only_and_passes_the_leakage_check():
    b = make_batch()
    gen = ObservationProcessGenerator(seed=0, rich_process="identity")
    rich = gen.rich_view(b["observed_data"], b["observed_tp"], b["observed_mask"])
    tb = BaseIrregularEncoder.with_observation(b, *rich)
    assert assert_no_future_leakage(b, tb)
    # the teacher's history never extends past the student's window
    assert float(b["observed_tp"].max()) < float(b["tp_to_predict"].min())
    assert rich[1].shape == b["observed_mask"].shape


def test_leakage_check_catches_future_targets():
    b = make_batch()
    bad = dict(b)
    bad["data_to_predict"] = b["data_to_predict"] + 1.0
    with pytest.raises(AssertionError):
        assert_no_future_leakage(b, bad)
    bad2 = dict(b)
    bad2["tp_to_predict"] = b["tp_to_predict"] * 2
    with pytest.raises(AssertionError):
        assert_no_future_leakage(b, bad2)


def test_teacher_view_is_richer_than_the_student_view():
    b = make_batch()
    gen = ObservationProcessGenerator(seed=0, train_processes=["random_dropout:0.6"], num_views=1)
    (sd, sm), = gen.views(b["observed_data"], b["observed_tp"], b["observed_mask"], k=1)
    rd, rm = gen.rich_view(b["observed_data"], b["observed_tp"], b["observed_mask"])
    assert rm.sum() > sm.sum(), "the teacher's observation is not richer than the student's"
    assert torch.all(sm <= rm), "the student observed something the teacher did not"


def test_prediction_distillation_is_masked_and_detached():
    b = make_batch()
    ps = torch.randn_like(b["data_to_predict"], requires_grad=True)
    pt = torch.randn_like(b["data_to_predict"], requires_grad=True)
    loss = prediction_distillation(ps, pt, b["mask_predicted_data"])
    loss.backward()
    assert pt.grad is None, "gradient flowed into the teacher"
    assert ps.grad is not None
    # entries with no target must not contribute
    ps2 = ps.detach().clone()
    ps2[b["mask_predicted_data"] == 0] += 1000.0
    assert torch.allclose(prediction_distillation(ps2, pt, b["mask_predicted_data"]),
                          prediction_distillation(ps.detach(), pt, b["mask_predicted_data"]))


def test_identical_predictions_give_zero_distillation():
    b = make_batch()
    p = torch.randn_like(b["data_to_predict"])
    assert float(prediction_distillation(p, p.clone(), b["mask_predicted_data"])) == pytest.approx(0.0)


# ---------------------------------------------------------------- classification batches
def make_cls_batch(B=4, M=4, L=8, N=6, n_class=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = (torch.rand(B, M, L, N, generator=g) < 0.6).float()
    return {
        "observed_data": torch.randn(B, M, L, N, generator=g) * mask,
        "observed_tp": torch.rand(B, M, L, N, generator=g),
        "observed_mask": mask,
        "static": None,
        "labels": torch.randint(n_class, (B,), generator=g),
    }


def test_leakage_check_handles_classification():
    b = make_cls_batch()
    gen = ObservationProcessGenerator(seed=0)
    rich = gen.rich_view(b["observed_data"], b["observed_tp"], b["observed_mask"])
    tb = BaseIrregularEncoder.with_observation(b, *rich)
    assert assert_no_future_leakage(b, tb)


def test_leakage_check_catches_changed_labels():
    b = make_cls_batch()
    bad = dict(b)
    bad["labels"] = 1 - b["labels"]
    with pytest.raises(AssertionError):
        assert_no_future_leakage(b, bad)


def test_classification_views_leave_labels_untouched():
    b = make_cls_batch()
    gen = ObservationProcessGenerator(seed=0)
    d, m = gen.condition(b["observed_data"], b["observed_tp"], b["observed_mask"], "variable_dropout:0.5")
    b2 = BaseIrregularEncoder.with_observation(b, d, m)
    assert torch.equal(b["labels"], b2["labels"])
    assert torch.all(m <= b["observed_mask"])
