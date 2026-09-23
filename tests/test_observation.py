"""Tests for the observation-process abstraction (OPERA section 5 / Phase B)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.observation import (  # noqa: E402
    ObservationConstraints, ObservationProcessGenerator, REGISTRY, apply, compute_metadata, descriptor,
    make_process, DESCRIPTOR_FIELDS, TRAIN_POOL, TEST_CONDITIONS, UNSEEN_CONDITIONS,
)

SPECS = ["identity", "random_dropout:0.5", "temporal_stride:3", "burst:0.5", "long_gap:0.4",
         "variable_dropout:0.3", "variable_group_dropout:0.5", "local_mask:0.4", "density:0.5",
         "random_dropout:0.3+variable_dropout:0.2", "random_dropout:0.2|burst:0.3"]


def batch3(B=6, L=40, N=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    tp = torch.rand(B, L, generator=g).sort(dim=1).values
    mask = (torch.rand(B, L, N, generator=g) < 0.6).float()
    data = torch.randn(B, L, N, generator=g) * mask
    return data, tp, mask


def batch4(B=4, M=4, L=10, N=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    tp = torch.rand(B, M, L, N, generator=g)
    mask = (torch.rand(B, M, L, N, generator=g) < 0.6).float()
    data = torch.randn(B, M, L, N, generator=g) * mask
    return data, tp, mask


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("maker", [batch3, batch4])
def test_processes_only_remove_observations(spec, maker):
    """S(X) must be a subset of the original observation and must zero the values it removes."""
    data, tp, mask = maker()
    g = torch.Generator().manual_seed(1)
    d2, m2 = apply(data, tp, mask, spec, g)
    assert m2.shape == mask.shape
    assert torch.all((m2 == 0) | (m2 == 1))
    assert torch.all(m2 <= mask), "a process invented observations"
    assert torch.all(d2[m2 == 0] == 0), "values left behind an unobserved entry"
    kept = (m2 > 0) & (mask > 0)
    assert torch.allclose(d2[kept], data[kept]), "a process modified observed values"


def test_registry_covers_required_processes():
    for name in ["random_dropout", "temporal_stride", "burst", "long_gap", "variable_dropout",
                 "variable_group_dropout", "local_mask", "density", "identity"]:
        assert name in REGISTRY


def test_severity_is_monotone():
    data, tp, mask = batch3()
    g = torch.Generator().manual_seed(2)
    prev = mask.sum()
    for p in [0.2, 0.4, 0.6, 0.8]:
        _, m = apply(data, tp, mask, f"random_dropout:{p}", g)
        assert m.sum() <= prev
        prev = m.sum()


def test_stride_and_density_have_expected_rates():
    data, tp, mask = batch3(B=32, L=60, N=8)
    _, m2 = apply(data, tp, mask, "temporal_stride:2")
    assert 0.45 <= float(m2.sum() / mask.sum()) <= 0.58
    _, m3 = apply(data, tp, mask, "density:0.5")
    assert 0.45 <= float(m3.sum() / mask.sum()) <= 0.60


def test_variable_dropout_removes_whole_variables_and_keeps_one():
    data, tp, mask = batch3(B=32, N=6)
    g = torch.Generator().manual_seed(3)
    _, m = apply(data, tp, mask, "variable_dropout:0.5", g)
    per_var = m.sum(dim=1)
    orig = mask.sum(dim=1)
    dropped = (per_var == 0) & (orig > 0)
    kept = (per_var > 0)
    assert kept.any(dim=1).all(), "some sample lost every variable"
    assert torch.all(per_var[kept] == orig[kept]), "variable dropout thinned a kept variable"
    assert dropped.any(), "no variable was dropped at p=0.5"


def test_identity_is_a_noop():
    data, tp, mask = batch3()
    d2, m2 = apply(data, tp, mask, "identity")
    assert torch.equal(m2, mask) and torch.equal(d2, data * mask)


def test_metadata_reflects_the_process():
    data, tp, mask = batch3(B=16, L=80, N=6)
    clean = compute_metadata(tp, mask)
    _, m_sparse = apply(data, tp, mask, "random_dropout:0.7", torch.Generator().manual_seed(4))
    _, m_gap = apply(data, tp, mask, "long_gap:0.5", torch.Generator().manual_seed(4))
    _, m_var = apply(data, tp, mask, "variable_dropout:0.5", torch.Generator().manual_seed(4))
    assert compute_metadata(tp, m_sparse).density.mean() < clean.density.mean()
    assert compute_metadata(tp, m_gap).max_gap.mean() > clean.max_gap.mean()
    assert compute_metadata(tp, m_var).variable_coverage.mean() < clean.variable_coverage.mean()


def test_descriptor_shape_and_range():
    for maker in (batch3, batch4):
        data, tp, mask = maker()
        r = descriptor(tp, mask)
        assert r.shape == (mask.shape[0], len(DESCRIPTOR_FIELDS))
        assert torch.isfinite(r).all()
        assert float(r.min()) >= 0.0 and float(r.max()) <= 1.01


def test_constraints_reject_over_aggressive_processes():
    data, tp, mask = batch3(B=16)
    c = ObservationConstraints(min_density=0.5, min_variable_coverage=0.9, max_gap=0.3)
    _, m_ok = apply(data, tp, mask, "random_dropout:0.1", torch.Generator().manual_seed(5))
    _, m_bad = apply(data, tp, mask, "random_dropout:0.9", torch.Generator().manual_seed(5))
    assert c.check(tp, m_ok, mask).float().mean() > 0.5
    assert c.check(tp, m_bad, mask).float().mean() == 0.0


def test_generator_is_reproducible_and_conditions_are_fixed():
    data, tp, mask = batch3()
    g1 = ObservationProcessGenerator(seed=7)
    g2 = ObservationProcessGenerator(seed=7)
    v1 = g1.views(data, tp, mask, k=3, with_meta=True)
    v2 = g2.views(data, tp, mask, k=3, with_meta=True)
    assert [x[3] for x in v1] == [x[3] for x in v2]
    for a, b in zip(v1, v2):
        assert torch.equal(a[1], b[1])
    # evaluation conditions must not depend on generator state, so every model sees the same shifted input
    a1 = g1.condition(data, tp, mask, "random_dropout:0.5")[1]
    a2 = g2.condition(data, tp, mask, "random_dropout:0.5")[1]
    g1.views(data, tp, mask, k=5)
    a3 = g1.condition(data, tp, mask, "random_dropout:0.5")[1]
    assert torch.equal(a1, a2) and torch.equal(a1, a3)


def test_candidate_pool_is_validated():
    data, tp, mask = batch3(B=8)
    g = ObservationProcessGenerator(seed=0, candidate_processes=["random_dropout:0.1", "random_dropout:0.99"],
                                    constraints={"min_density": 0.3})
    items, valid = g.candidates(data, tp, mask)
    assert len(items) == 2 and valid.shape == (8, 2)
    assert valid[:, 0].all() and not valid[:, 1].any()


def test_train_and_test_pools_are_disjoint_in_severity():
    """OPERA section 23: the unseen conditions must not appear in the training pool."""
    assert set(TRAIN_POOL).isdisjoint(set(UNSEEN_CONDITIONS))
    assert set(UNSEEN_CONDITIONS).issubset(set(TEST_CONDITIONS))


def test_spec_roundtrip():
    for spec in SPECS:
        assert make_process(spec).spec.replace(".0", "") or True   # constructible
    assert make_process("temporal_stride:3").spec == "temporal_stride:3"
    assert make_process("random_dropout:0.5").spec == "random_dropout:0.5"
