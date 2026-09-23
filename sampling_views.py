"""Sampling-view generator for Sampling-Invariant QuITE (Phase 1).

A *view* is a different irregular observation pattern of the same underlying
trajectory. Views are produced purely by editing the observation mask (and
zeroing the corresponding values), so they work with every input layout used
in this repo:

    forecasting / classification, patch family : (B, M, L_in, N)
    forecasting, variate family                 : (B, L, N)

View specs are strings so they can be passed on the command line:

    none            identity (the original observation pattern)
    random:p        drop each observed point independently with prob p
    bursty:p        drop contiguous time windows covering a fraction p of each
                    sample's observed time range (2 windows per sample)
    variable:p      drop entire variables with prob p (per sample; at least
                    one variable is always kept)
    mixed           per call, pick uniformly among random:0.3, random:0.6,
                    bursty:0.3, bursty:0.5, variable:0.3 (training mixture)
    a|b|c           per call, pick uniformly among the listed specs

Prediction targets are never touched: the model must still forecast the same
future from a different observation of the same past.
"""
import torch

MIXED_POOL = ["random:0.3", "random:0.6", "bursty:0.3", "bursty:0.5", "variable:0.3"]

# Views used for the cross-sampling robustness table.
DEFAULT_EVAL_VIEWS = ["none", "random:0.3", "random:0.5", "random:0.7",
                      "bursty:0.3", "bursty:0.5", "variable:0.3"]
# Phase 2: adds regimes that are NOT in the 'mixed' training pool (unseen severity / unseen type).
UNSEEN_EVAL_VIEWS = ["variable:0.5", "stride:2", "stride:3"]
EXTENDED_EVAL_VIEWS = DEFAULT_EVAL_VIEWS + UNSEEN_EVAL_VIEWS


def parse_view(spec):
    """'random:0.5' -> ('random', 0.5); 'none' -> ('none', 0.0)."""
    spec = spec.strip()
    if spec in ("none", "dense", ""):
        return "none", 0.0
    if ":" in spec:
        kind, p = spec.split(":", 1)
        try:
            return kind, float(p)
        except ValueError:          # non-numeric parameter, e.g. 'dropvars:5-6-7'
            return kind, 1.0
    return spec, 0.0


def _choose(spec, gen):
    """Resolve 'mixed' / 'a|b' specs to a single concrete spec."""
    if spec == "mixed":
        pool = MIXED_POOL
    elif "|" in spec:
        pool = [s.strip() for s in spec.split("|")]
    else:
        return spec
    i = int(torch.randint(len(pool), (1,), generator=gen, device=gen.device if gen else None))
    return pool[i]


def _rand(shape, gen, device):
    return torch.rand(shape, generator=gen, device=device)


def keep_mask(tp, mask, spec, gen=None):
    """Return a {0,1} tensor (same shape as `mask`) selecting which observed
    entries to KEEP under the view `spec`."""
    spec = _choose(spec, gen)
    kind, p = parse_view(spec)
    device = mask.device
    if kind == "none" or p <= 0:
        return torch.ones_like(mask)

    if kind == "random":
        return (_rand(mask.shape, gen, device) >= p).to(mask.dtype)

    if kind == "variable":
        # mask: (..., N); decide per (sample, variable)
        B, N = mask.shape[0], mask.shape[-1]
        drop = _rand((B, N), gen, device) < p
        # always keep at least one variable per sample
        all_dropped = drop.all(dim=1)
        if all_dropped.any():
            j = torch.randint(N, (int(all_dropped.sum()),), generator=gen, device=device)
            drop[all_dropped.nonzero(as_tuple=True)[0], j] = False
        shape = [B] + [1] * (mask.dim() - 2) + [N]
        return (~drop).to(mask.dtype).view(shape).expand_as(mask).clone()

    if kind == "dropvars":
        # Remove specific variables entirely, e.g. 'dropvars:5-6-7' (Phase 3 critical experiment).
        idx = [int(x) for x in spec.split(":")[1].split("-") if x != ""]
        k = torch.ones_like(mask)
        k[..., idx] = 0
        return k

    if kind == "stride":
        # Systematic subsampling: keep every k-th observation of each variable (in time order).
        # Lower effective sampling rate with regular spacing - a regime type absent from 'mixed'.
        k = max(int(p), 1)
        tdim = mask.dim() - 2                       # time axis: (B, L, N) -> 1, (B, M, L, N) -> 2
        order = (mask > 0).float().cumsum(dim=tdim) - 1
        return ((order % k == 0) & (mask > 0)).to(mask.dtype)

    if kind == "bursty":
        # Drop 2 contiguous time windows per sample, total width = p * range.
        B = mask.shape[0]
        while tp.dim() < mask.dim():       # e.g. (B, L) time steps vs (B, L, N) mask
            tp = tp.unsqueeze(-1)
        tp = tp.expand_as(mask)
        flat_tp = tp.reshape(B, -1)
        flat_m = mask.reshape(B, -1) > 0
        big = torch.full_like(flat_tp, float("inf"))
        t_min = torch.where(flat_m, flat_tp, big).min(dim=1).values
        t_max = torch.where(flat_m, flat_tp, -big).max(dim=1).values
        rng = (t_max - t_min).clamp_min(1e-6)
        n_win = 2
        width = (p * rng / n_win).unsqueeze(1)                       # (B,1)
        starts = t_min.unsqueeze(1) + _rand((B, n_win), gen, device) * (rng.unsqueeze(1) - width)
        ends = starts + width
        keep = torch.ones_like(flat_tp, dtype=torch.bool)
        for w in range(n_win):
            inside = (flat_tp >= starts[:, w:w + 1]) & (flat_tp <= ends[:, w:w + 1])
            keep &= ~inside
        return keep.to(mask.dtype).reshape(mask.shape)

    raise ValueError(f"Unknown view spec: {spec}")


def apply_view(data, tp, mask, spec, gen=None):
    """Return (data, mask) observed through view `spec`. Time is unchanged."""
    if spec in ("none", "dense", ""):
        return data, mask
    k = keep_mask(tp, mask, spec, gen)
    new_mask = mask * k
    return data * new_mask, new_mask


def view_batch(batch_dict, spec, gen=None):
    """Forecasting batch dict -> shallow copy with observed_data / observed_mask viewed."""
    if spec in ("none", "dense", ""):
        return batch_dict
    out = dict(batch_dict)
    out["observed_data"], out["observed_mask"] = apply_view(
        batch_dict["observed_data"], batch_dict["observed_tp"], batch_dict["observed_mask"], spec, gen)
    return out


def make_generator(seed, device):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def consistency_loss(za, zb, dist="cos"):
    """Distance between two representations of the same samples.

    za, zb: (B*, ..., D) — flattened per sample.
    """
    za = za.reshape(za.shape[0], -1)
    zb = zb.reshape(zb.shape[0], -1)
    if dist == "cos":
        return (1.0 - torch.nn.functional.cosine_similarity(za, zb, dim=-1)).mean()
    if dist == "mse":
        return torch.nn.functional.mse_loss(za, zb)
    raise ValueError(f"Unknown consistency distance: {dist}")


@torch.no_grad()
def repr_diagnostics(za, zb):
    """Collapse diagnostics: similarity of positive pairs (same sample, two
    views) vs. shuffled negatives (different samples), and feature std."""
    a = za.reshape(za.shape[0], -1)
    b = zb.reshape(zb.shape[0], -1)
    pos = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean()
    perm = torch.roll(torch.arange(a.shape[0], device=a.device), shifts=1)
    neg = torch.nn.functional.cosine_similarity(a, b[perm], dim=-1).mean()
    std = a.std(dim=0).mean()
    return {"repr/pos_cos": pos.item(), "repr/neg_cos": neg.item(), "repr/std": std.item()}
