"""Validity constraints on observation processes (OPERA section 14).

A mined or sampled observation must stay realistic: the adversary may not simply delete everything. All bounds
are configurable and checked per sample from the mask alone; `filter_candidates` is what the hard miner uses to
discard invalid candidates before ranking them by loss.
"""
from dataclasses import dataclass, asdict
from typing import Optional

import torch

from src.observation.metadata import compute_metadata


@dataclass
class ObservationConstraints:
    min_density: float = 0.0            # observed entries / possible entries, relative to the ORIGINAL mask
    max_density: float = 1.0
    min_variable_coverage: float = 0.0  # fraction of variables that must keep >= 1 observation
    max_gap: float = 1.0                # largest allowed inter-observation gap (window fractions)
    max_missing_fraction: float = 1.0   # 1 - relative density
    min_per_variable_coverage: float = 0.0   # every kept variable must retain this fraction of its observations
    min_obs_per_sample: int = 1

    @classmethod
    def from_config(cls, cfg):
        cfg = cfg or {}
        known = {k: cfg[k] for k in asdict(cls()).keys() if k in cfg}
        return cls(**known)

    def check(self, tp, mask, base_mask=None):
        """Return a (B,) bool tensor: True where the observation satisfies every constraint.

        `base_mask` is the original observation (before the process); densities are measured relative to it so
        that a constraint like min_density=0.3 means "keep at least 30% of what was there".
        """
        meta = compute_metadata(tp, mask)
        B = mask.shape[0]
        ok = torch.ones(B, dtype=torch.bool, device=mask.device)

        n_obs = (mask > 0).reshape(B, -1).sum(dim=1)
        ok &= n_obs >= self.min_obs_per_sample

        if base_mask is None:
            rel = meta.density
            per_var_rel = meta.per_variable_density
        else:
            base = compute_metadata(tp, base_mask)
            rel = meta.density / base.density.clamp_min(1e-8)
            per_var_rel = meta.per_variable_density / base.per_variable_density.clamp_min(1e-8)
        ok &= rel >= self.min_density
        ok &= rel <= self.max_density
        ok &= (1.0 - rel) <= self.max_missing_fraction
        ok &= meta.variable_coverage >= self.min_variable_coverage
        ok &= meta.max_gap <= self.max_gap
        if self.min_per_variable_coverage > 0:
            kept = per_var_rel > 0
            bad = (per_var_rel < self.min_per_variable_coverage) & kept
            ok &= ~bad.any(dim=1)
        return ok

    def filter_candidates(self, tp, masks, base_mask=None):
        """masks: list of candidate masks. Returns a (B, K) bool tensor of validity."""
        return torch.stack([self.check(tp, m, base_mask) for m in masks], dim=1)


DEFAULT = ObservationConstraints(min_density=0.15, min_variable_coverage=0.3, max_gap=0.6, min_obs_per_sample=4)
