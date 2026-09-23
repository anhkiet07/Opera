"""Observation processes S: mask-level transformations of an observed irregular time series (OPERA section 5).

Every process only *removes* observations, i.e. it returns a keep-mask of the same shape as the input mask. No
process invents values, so O = S(X) always remains a subset of the original observation. Values are zeroed
wherever the mask is zero by `apply`, matching the convention of the datasets in this repo.

Tensor layouts (last axis = variables, second-to-last = time):
    (B, L, N)          variate family
    (B, M, L_in, N)    patch family

Each process has a `spec` string so an observation regime is fully described by text and can live in a YAML
config: 'random_dropout:0.5', 'temporal_stride:3', 'burst:0.5', 'long_gap:0.4', 'variable_dropout:0.3',
'variable_group_dropout:0.3', 'local_mask:0.3', 'density:0.4', 'identity'. Composition uses '+'
('random_dropout:0.3+variable_dropout:0.2'); a uniform mixture uses '|'.
"""
import torch

REGISTRY = {}


def register(name):
    def deco(cls):
        REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


def _time_axis(mask):
    return mask.dim() - 2


def _rand(shape, device, gen):
    return torch.rand(shape, device=device, generator=gen)


class ObservationProcess:
    """Base class. Subclasses implement keep_mask(tp, mask, gen) -> {0,1} tensor like `mask`."""
    name = "base"

    def __init__(self, p=0.0):
        self.p = float(p)

    @property
    def spec(self):
        return f"{self.name}:{self.p:g}"

    def keep_mask(self, tp, mask, gen=None):
        raise NotImplementedError

    def __call__(self, tp, mask, gen=None):
        return self.keep_mask(tp, mask, gen)

    def __repr__(self):
        return f"<{self.spec}>"


@register("identity")
class Identity(ObservationProcess):
    @property
    def spec(self):
        return "identity"

    def keep_mask(self, tp, mask, gen=None):
        return torch.ones_like(mask)


@register("random_dropout")
class RandomDropout(ObservationProcess):
    """Drop each observed entry independently with probability p."""

    def keep_mask(self, tp, mask, gen=None):
        if self.p <= 0:
            return torch.ones_like(mask)
        return (_rand(mask.shape, mask.device, gen) >= self.p).to(mask.dtype)


@register("temporal_stride")
class TemporalStride(ObservationProcess):
    """Keep every k-th observation of each variable in time order (systematic subsampling, k = p)."""

    def __init__(self, p=2):
        super().__init__(p)
        self.k = max(int(round(float(p))), 1)

    @property
    def spec(self):
        return f"{self.name}:{self.k}"

    def keep_mask(self, tp, mask, gen=None):
        if self.k <= 1:
            return torch.ones_like(mask)
        tdim = _time_axis(mask)
        order = (mask > 0).to(mask.dtype).cumsum(dim=tdim) - 1
        return ((order % self.k == 0) & (mask > 0)).to(mask.dtype)


@register("burst")
class BurstMissingness(ObservationProcess):
    """Remove `n_windows` contiguous time windows covering a fraction p of each sample's observed span."""

    def __init__(self, p=0.5, n_windows=2):
        super().__init__(p)
        self.n_windows = int(n_windows)

    @property
    def spec(self):
        return f"{self.name}:{self.p:g}" + (f":{self.n_windows}" if self.n_windows != 2 else "")

    def keep_mask(self, tp, mask, gen=None):
        if self.p <= 0:
            return torch.ones_like(mask)
        B, N = mask.shape[0], mask.shape[-1]
        while tp.dim() < mask.dim():
            tp = tp.unsqueeze(-1)
        tp = tp.expand_as(mask)
        flat_tp = tp.reshape(B, -1)
        flat_m = (mask > 0).reshape(B, -1)
        big = torch.full_like(flat_tp, float("inf"))
        t_min = torch.where(flat_m, flat_tp, big).min(dim=1).values
        t_max = torch.where(flat_m, flat_tp, -big).max(dim=1).values
        rng = (t_max - t_min).clamp_min(1e-6)
        width = (self.p * rng / self.n_windows).unsqueeze(1)
        starts = t_min.unsqueeze(1) + _rand((B, self.n_windows), mask.device, gen) * (rng.unsqueeze(1) - width)
        keep = torch.ones_like(flat_tp, dtype=torch.bool)
        for w in range(self.n_windows):
            inside = (flat_tp >= starts[:, w:w + 1]) & (flat_tp <= starts[:, w:w + 1] + width)
            keep &= ~inside
        return keep.to(mask.dtype).reshape(mask.shape)


@register("long_gap")
class LongGapMissingness(BurstMissingness):
    """One long contiguous gap covering a fraction p of the observed span (the extreme case of burst)."""

    def __init__(self, p=0.4):
        super().__init__(p, n_windows=1)

    @property
    def spec(self):
        return f"{self.name}:{self.p:g}"


@register("variable_dropout")
class VariableDropout(ObservationProcess):
    """Remove whole variables, each with probability p; at least one variable is always kept."""

    def keep_mask(self, tp, mask, gen=None):
        if self.p <= 0:
            return torch.ones_like(mask)
        B, N = mask.shape[0], mask.shape[-1]
        drop = _rand((B, N), mask.device, gen) < self.p
        all_dropped = drop.all(dim=1)
        if all_dropped.any():
            j = torch.randint(N, (int(all_dropped.sum()),), device=mask.device, generator=gen)
            drop[all_dropped.nonzero(as_tuple=True)[0], j] = False
        shape = [B] + [1] * (mask.dim() - 2) + [N]
        return (~drop).to(mask.dtype).view(shape).expand_as(mask).clone()


@register("variable_group_dropout")
class VariableGroupDropout(ObservationProcess):
    """Correlated variable dropout: variables are split into `n_groups` contiguous blocks and whole blocks are
    removed with probability p (models a failing monitor / an unavailable lab panel)."""

    def __init__(self, p=0.3, n_groups=4):
        super().__init__(p)
        self.n_groups = int(n_groups)

    @property
    def spec(self):
        return f"{self.name}:{self.p:g}" + (f":{self.n_groups}" if self.n_groups != 4 else "")

    def keep_mask(self, tp, mask, gen=None):
        if self.p <= 0:
            return torch.ones_like(mask)
        B, N = mask.shape[0], mask.shape[-1]
        g = min(self.n_groups, N)
        drop_g = _rand((B, g), mask.device, gen) < self.p
        all_dropped = drop_g.all(dim=1)
        if all_dropped.any():
            j = torch.randint(g, (int(all_dropped.sum()),), device=mask.device, generator=gen)
            drop_g[all_dropped.nonzero(as_tuple=True)[0], j] = False
        bounds = torch.linspace(0, N, g + 1, device=mask.device).long()
        keep_var = torch.ones((B, N), device=mask.device, dtype=mask.dtype)
        for i in range(g):
            keep_var[:, bounds[i]:bounds[i + 1]] = (~drop_g[:, i:i + 1]).to(mask.dtype)
        shape = [B] + [1] * (mask.dim() - 2) + [N]
        return keep_var.view(shape).expand_as(mask).clone()


@register("local_mask")
class LocalTemporalMask(ObservationProcess):
    """Per-variable local blackout: for each (sample, variable) remove one contiguous run covering a fraction p
    of that variable's observations. Same density as random dropout but clustered per variable."""

    def keep_mask(self, tp, mask, gen=None):
        if self.p <= 0:
            return torch.ones_like(mask)
        tdim = _time_axis(mask)
        m = (mask > 0)
        order = m.to(mask.dtype).cumsum(dim=tdim) - 1              # observation index per variable
        n = m.to(mask.dtype).sum(dim=tdim, keepdim=True)           # count per variable
        width = (self.p * n)
        start_shape = list(mask.shape)
        start_shape[tdim] = 1
        start = _rand(tuple(start_shape), mask.device, gen) * (n - width).clamp_min(0)
        inside = (order >= start) & (order < start + width) & m
        return (~inside).to(mask.dtype)


@register("density")
class DensityControlledSampling(ObservationProcess):
    """Keep a target fraction p of the observations of every variable, uniformly at random (density control:
    the resulting density is p * original density regardless of the original pattern)."""

    def keep_mask(self, tp, mask, gen=None):
        if self.p >= 1:
            return torch.ones_like(mask)
        tdim = _time_axis(mask)
        m = (mask > 0)
        noise = _rand(mask.shape, mask.device, gen)
        noise = torch.where(m, noise, torch.full_like(noise, 2.0))
        ranks = noise.argsort(dim=tdim).argsort(dim=tdim).to(mask.dtype)   # 0..T-1 among observed first
        n = m.to(mask.dtype).sum(dim=tdim, keepdim=True)
        keep_n = (self.p * n).ceil()
        return ((ranks < keep_n) & m).to(mask.dtype)


class Composite(ObservationProcess):
    """Sequential composition: the keep-masks of all parts are multiplied."""

    def __init__(self, parts):
        super().__init__(0.0)
        self.parts = list(parts)

    @property
    def spec(self):
        return "+".join(p.spec for p in self.parts)

    def keep_mask(self, tp, mask, gen=None):
        keep = torch.ones_like(mask)
        for p in self.parts:
            keep = keep * p.keep_mask(tp, mask * keep, gen)
        return keep


class MixedObservationProcess(ObservationProcess):
    """Uniform (or weighted) mixture: one member is drawn per call."""

    def __init__(self, members, weights=None):
        super().__init__(0.0)
        self.members = list(members)
        self.weights = weights
        self.last = None

    @property
    def spec(self):
        return "|".join(m.spec for m in self.members)

    def pick(self, gen=None):
        if self.weights is None:
            i = int(torch.randint(len(self.members), (1,), device="cpu", generator=None if gen is None or gen.device.type != "cpu" else gen))
        else:
            w = torch.tensor(self.weights, dtype=torch.float)
            i = int(torch.multinomial(w, 1))
        self.last = self.members[i]
        return self.last

    def keep_mask(self, tp, mask, gen=None):
        return self.pick(gen).keep_mask(tp, mask, gen)


def make_process(spec):
    """Build an ObservationProcess from a spec string (see the module docstring)."""
    spec = str(spec).strip()
    if "|" in spec:
        return MixedObservationProcess([make_process(s) for s in spec.split("|")])
    if "+" in spec:
        return Composite([make_process(s) for s in spec.split("+")])
    if spec in ("", "none", "identity", "clean"):
        return Identity()
    parts = spec.split(":")
    name, args = parts[0], parts[1:]
    if name not in REGISTRY:
        raise ValueError(f"unknown observation process '{name}'; known: {sorted(REGISTRY)}")
    return REGISTRY[name](*[float(a) for a in args])


def apply(data, tp, mask, process, gen=None, return_meta=False):
    """Observe (data, mask) through `process`. Returns (data', mask') and optionally the metadata of mask'."""
    if isinstance(process, str):
        process = make_process(process)
    keep = process.keep_mask(tp, mask, gen)
    new_mask = mask * keep
    new_data = data * new_mask
    if return_meta:
        from src.observation.metadata import compute_metadata
        return new_data, new_mask, compute_metadata(tp, new_mask, regime=process.spec)
    return new_data, new_mask
