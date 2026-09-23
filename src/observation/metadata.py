"""Observation-process metadata and the descriptor r = g(S) (OPERA sections 5-6).

Everything here is computed from the observation mask and timestamps only, per sample, and never from the
values, so it can be handed to the model, logged, or used by mining constraints without leaking the target.

Tensor layouts supported (the last axis is always the variable axis):
    (B, L, N)          variate-family batches
    (B, M, L_in, N)    patch-family batches (the time axis is the second-to-last)
"""
from dataclasses import dataclass, asdict, field
from typing import Optional

import torch

# Descriptor field order, also the order of the vector returned by `descriptor`.
DESCRIPTOR_FIELDS = (
    "density",              # observed entries / possible entries
    "log_n_obs",            # log1p(number of observed entries per variable, mean over variables)
    "mean_gap",             # mean inter-observation time gap (normalised by the window)
    "max_gap",              # largest gap
    "gap_std",              # gap standard deviation
    "temporal_coverage",    # observed span / window
    "variable_coverage",    # fraction of variables with >= 1 observation
    "recent_density",       # density in the last quarter of the window
)


@dataclass
class ObservationMetadata:
    """Per-sample summary of one observation process realisation."""
    density: torch.Tensor                       # (B,)
    mean_gap: torch.Tensor                      # (B,)
    max_gap: torch.Tensor                       # (B,)
    gap_std: torch.Tensor                       # (B,)
    variable_coverage: torch.Tensor             # (B,)
    temporal_coverage: torch.Tensor             # (B,)
    missing_fraction: torch.Tensor              # (B,)
    per_variable_density: torch.Tensor          # (B, N)
    recent_density: torch.Tensor                # (B,)
    log_n_obs: torch.Tensor                     # (B,)
    sampling_regime: Optional[str] = None       # the spec string that produced it, when known
    extra: dict = field(default_factory=dict)

    def summary(self):
        """Scalar means, for logging."""
        out = {}
        for k, v in asdict(self).items():
            if torch.is_tensor(v) and v.numel():
                out[k] = float(v.float().mean())
        if self.sampling_regime is not None:
            out["sampling_regime"] = self.sampling_regime
        return out


def _time_axis(mask):
    return mask.dim() - 2


def compute_metadata(tp, mask, regime=None, window=1.0):
    """Compute ObservationMetadata for a batch.

    tp   : timestamps, broadcastable to `mask` (e.g. (B, L) or (B, L, N) or (B, M, L_in, N))
    mask : {0,1} observation mask
    window: length of the observation window in the same units as `tp` (normalised time -> 1.0)
    """
    m = (mask > 0)
    B = mask.shape[0]
    N = mask.shape[-1]
    flat_m = m.reshape(B, -1, N)                       # (B, T, N)
    T = flat_m.shape[1]

    per_var = flat_m.float().mean(dim=1)               # (B, N)
    density = flat_m.float().reshape(B, -1).mean(dim=1)
    n_obs = flat_m.float().sum(dim=1)                  # (B, N)
    log_n_obs = torch.log1p(n_obs).mean(dim=1)
    variable_coverage = (n_obs > 0).float().mean(dim=1)

    while tp.dim() < mask.dim():
        tp = tp.unsqueeze(-1)
    tp = tp.expand_as(mask)
    # collapse the variable axis: a time index counts as observed if any variable is observed there, and the
    # timestamp of that index is shared across variables.
    flat_tp = tp.reshape(B, -1, N)[..., 0]             # (B, T)
    obs = flat_m.any(-1)                                # (B, T)
    big = torch.full_like(flat_tp, float("inf"))
    t_first = torch.where(obs, flat_tp, big).min(dim=1).values
    t_last = torch.where(obs, flat_tp, -big).max(dim=1).values
    has_any = obs.any(dim=1)
    span = torch.where(has_any, (t_last - t_first).clamp_min(0), torch.zeros_like(t_last))

    # gaps between consecutive observed time indices
    sorted_t = torch.where(obs, flat_tp, big).sort(dim=1).values
    d = torch.diff(sorted_t, dim=1)
    valid = torch.isfinite(d)
    d = torch.where(valid, d, torch.zeros_like(d))
    cnt = valid.sum(dim=1).clamp_min(1).float()
    mean_gap = d.sum(dim=1) / cnt
    var_gap = (((d - mean_gap.unsqueeze(1)) ** 2) * valid).sum(dim=1) / cnt
    gap_std = var_gap.clamp_min(0).sqrt()
    max_gap = torch.where(valid, d, torch.full_like(d, -1.0)).max(dim=1).values.clamp_min(0)

    tail = max(T // 4, 1)
    recent_density = flat_m[:, -tail:, :].float().reshape(B, -1).mean(dim=1)

    return ObservationMetadata(
        density=density,
        mean_gap=mean_gap / window,
        max_gap=max_gap / window,
        gap_std=gap_std / window,
        variable_coverage=variable_coverage,
        temporal_coverage=(span / window).clamp(0, 1),
        missing_fraction=1.0 - density,
        per_variable_density=per_var,
        recent_density=recent_density,
        log_n_obs=log_n_obs,
        sampling_regime=regime,
    )


def descriptor(tp, mask, window=1.0, meta=None):
    """r = g(S): a normalised (B, len(DESCRIPTOR_FIELDS)) descriptor of the observation process."""
    meta = meta or compute_metadata(tp, mask, window=window)
    cols = [
        meta.density,
        meta.log_n_obs / 8.0,                 # log1p(#obs) rarely exceeds 8 in these datasets
        meta.mean_gap.clamp(0, 1),
        meta.max_gap.clamp(0, 1),
        meta.gap_std.clamp(0, 1),
        meta.temporal_coverage,
        meta.variable_coverage,
        meta.recent_density,
    ]
    return torch.stack(cols, dim=-1).float()
