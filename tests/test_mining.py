"""Tests for candidate-pool hard mining (OPERA sections 12-15)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mining.hard_miner import HardMiner  # noqa: E402
from src.models.base import BaseIrregularEncoder  # noqa: E402
from src.observation import ObservationProcessGenerator  # noqa: E402
from tests.test_no_leakage import make_batch  # noqa: E402


class DensityModel(BaseIrregularEncoder):
    """A stand-in whose error grows as observations are removed, so the hardest candidate is the sparsest."""

    def __init__(self, n_pred, n_var):
        super().__init__()
        self.n_pred, self.n_var = n_pred, n_var
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch, return_repr=False, obs_descriptor=None):
        m = batch["observed_mask"]
        density = (m > 0).float().reshape(m.shape[0], -1).mean(dim=1)
        # prediction error scales with missingness
        err = (1.0 - density).view(-1, 1, 1) * 10.0
        pred = batch["data_to_predict"] + err + self.dummy
        return (pred, pred.reshape(pred.shape[0], -1)) if return_repr else pred


def test_miner_selects_the_hardest_candidate():
    b = make_batch(B=8)
    gen = ObservationProcessGenerator(
        seed=0, candidate_processes=["random_dropout:0.1", "random_dropout:0.5", "random_dropout:0.85"],
        constraints={"min_density": 0.0})
    model = DensityModel(b["data_to_predict"].shape[1], b["data_to_predict"].shape[2])
    miner = HardMiner(gen, top_k=1, alpha=1.0)
    (d, m, spec), = miner.mine(model, b, n_select=1)
    assert spec == "random_dropout:0.85", f"picked {spec} instead of the hardest candidate"


def test_constraints_prevent_degenerate_selection():
    """With a density floor the miner must not pick the (invalid) near-empty observation."""
    b = make_batch(B=8)
    gen = ObservationProcessGenerator(
        seed=0, candidate_processes=["random_dropout:0.1", "random_dropout:0.5", "random_dropout:0.95"],
        constraints={"min_density": 0.4})
    model = DensityModel(b["data_to_predict"].shape[1], b["data_to_predict"].shape[2])
    miner = HardMiner(gen, top_k=1, alpha=1.0)
    (_, m, spec), = miner.mine(model, b, n_select=1)
    assert spec != "random_dropout:0.95"
    assert float((m > 0).float().mean()) > 0.0


def test_alpha_mixes_random_and_hard():
    b = make_batch(B=6)
    gen = ObservationProcessGenerator(seed=0, train_processes=["random_dropout:0.2"],
                                      candidate_processes=["random_dropout:0.5", "random_dropout:0.8"],
                                      constraints={"min_density": 0.0})
    model = DensityModel(b["data_to_predict"].shape[1], b["data_to_predict"].shape[2])
    # the pool holds 2 candidates, so n_hard is capped at 2 even when alpha * n_select is larger
    for alpha, exp_hard in [(0.0, 0), (0.5, 2), (1.0, 2)]:
        miner = HardMiner(gen, top_k=4, alpha=alpha)
        views = miner.mine(model, b, n_select=4)
        assert len(views) == 4, "the miner must always return the requested number of views"
        assert miner.stats["n_mined"] == exp_hard
        assert miner.stats["n_random"] == 4 - exp_hard


def test_warmup_disables_mining():
    gen = ObservationProcessGenerator(seed=0)
    miner = HardMiner(gen, alpha=0.5, warmup_epochs=5)
    assert not miner.active(0) and not miner.active(4) and miner.active(5)
    assert not HardMiner(gen, alpha=0.0).active(10)


def test_mined_views_leave_targets_untouched():
    b = make_batch(B=4)
    gen = ObservationProcessGenerator(seed=0, candidate_processes=["random_dropout:0.5"],
                                      constraints={"min_density": 0.0})
    model = DensityModel(b["data_to_predict"].shape[1], b["data_to_predict"].shape[2])
    miner = HardMiner(gen, top_k=1, alpha=1.0)
    (d, m, _), = miner.mine(model, b, n_select=1)
    b2 = model.with_observation(b, d, m)
    for k in ("tp_to_predict", "data_to_predict", "mask_predicted_data"):
        assert torch.equal(b[k], b2[k])
    assert torch.all(m <= b["observed_mask"])


def test_mining_does_not_touch_model_mode_or_grads():
    b = make_batch(B=4)
    gen = ObservationProcessGenerator(seed=0, candidate_processes=["random_dropout:0.3", "random_dropout:0.6"],
                                      constraints={"min_density": 0.0})
    model = DensityModel(b["data_to_predict"].shape[1], b["data_to_predict"].shape[2])
    model.train()
    miner = HardMiner(gen, top_k=1, alpha=1.0)
    miner.mine(model, b, n_select=1)
    assert model.training, "the miner left the model in eval mode"
    assert model.dummy.grad is None, "candidate scoring produced gradients"


class CEDensityModel(BaseIrregularEncoder):
    """Classification stand-in whose logits get less confident as observations are removed."""

    def __init__(self, n_class=2):
        super().__init__()
        self.task = "classification"
        self.n_class = n_class
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch, return_repr=False, obs_descriptor=None):
        m = batch["observed_mask"]
        density = (m > 0).float().reshape(m.shape[0], -1).mean(dim=1)
        y = batch["labels"]
        logits = torch.zeros(y.shape[0], self.n_class, device=m.device)
        logits[torch.arange(y.shape[0]), y] = 6.0 * density      # sparser -> less confident -> higher CE
        return (logits + self.dummy, logits) if return_repr else logits + self.dummy


def test_miner_handles_classification_batches():
    from tests.test_no_leakage import make_cls_batch
    b = make_cls_batch(B=8)
    gen = ObservationProcessGenerator(
        seed=0, candidate_processes=["random_dropout:0.1", "random_dropout:0.5", "random_dropout:0.85"],
        constraints={"min_density": 0.0})
    model = CEDensityModel()
    miner = HardMiner(gen, top_k=1, alpha=1.0)
    (_, _, spec), = miner.mine(model, b, n_select=1)
    assert spec == "random_dropout:0.85", f"picked {spec} instead of the hardest classification candidate"


def test_classification_candidate_losses_are_per_sample():
    from tests.test_no_leakage import make_cls_batch
    b = make_cls_batch(B=6)
    gen = ObservationProcessGenerator(seed=0, candidate_processes=["random_dropout:0.2", "random_dropout:0.8"],
                                      constraints={"min_density": 0.0})
    model = CEDensityModel()
    miner = HardMiner(gen, top_k=1, alpha=1.0)
    items, _ = gen.candidates(b["observed_data"], b["observed_tp"], b["observed_mask"])
    losses = miner._candidate_losses(model, b, items)
    assert losses.shape == (6, 2)
    assert (losses[:, 1] > losses[:, 0]).all(), "the sparser candidate should have the higher loss"
