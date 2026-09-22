"""QuITE (mode='quite'): query-based irregular time series embedding (paper main method).

For each "structured target" (variable or patch), a learnable query token is
prepended to the irregular observation tokens and aggregated via a single
masked self-attention layer (Eq. 5-13 in the paper). The updated query token
is returned as the structured embedding.

Phase 2 extension (args.adaptive_query != 'none'): the query token becomes
observation-aware,  q(X) = q_base + MLP(s(X)),  where s(X) summarises the
observation pattern seen by that query (number of observations, density,
time-gap statistics, availability). The MLP's last layer is zero-initialised,
so every adaptive variant starts exactly at plain QuITE.
"""
import torch
import torch.nn as nn

from models.modules import SelfAttentionBlock
from models.embeddings._base import LearnableTE


# ------------------------------------------------------------------------------------------
# Observation-pattern statistics
# ------------------------------------------------------------------------------------------
STAT_GROUPS = {
    # name -> indices into the 7-dim statistic vector produced by obs_stats()
    'density': [0, 1],          # log1p(n_obs), n_obs / L_max
    'gaps':    [2, 3, 4],       # mean dt, std dt, observed span
    'avail':   [5, 6],          # this target has >= 1 obs, fraction of targets with >= 1 obs
}
STAT_GROUPS['full'] = STAT_GROUPS['density'] + STAT_GROUPS['gaps'] + STAT_GROUPS['avail']
STAT_GROUPS['random'] = STAT_GROUPS['full']   # same width, contents replaced by noise
N_STATS = 7


def obs_stats(time, mask, tdim, target_dim):
    """Per-target observation statistics.

    time, mask : same shape; `tdim` is the time axis that is reduced;
                 `target_dim` is the axis over which "availability fraction" is taken
                 (variables for the variate family, variables within a patch for the patch family).
    Returns a tensor with the time axis removed and a trailing axis of N_STATS features.
    """
    m = mask > 0
    n = m.sum(dim=tdim).float()                                            # (...)
    l_max = float(mask.shape[tdim])
    inf = torch.full_like(time, float('inf'))
    t_sorted = torch.where(m, time, inf).sort(dim=tdim).values
    d = torch.diff(t_sorted, dim=tdim)
    valid = torch.isfinite(d)
    d = torch.where(valid, d, torch.zeros_like(d))
    cnt = valid.sum(dim=tdim).float()
    mean_dt = d.sum(dim=tdim) / cnt.clamp_min(1.0)
    var_dt = (((d - mean_dt.unsqueeze(tdim)) ** 2) * valid).sum(dim=tdim) / cnt.clamp_min(1.0)
    std_dt = var_dt.clamp_min(0).sqrt()
    t_first = t_sorted.narrow(tdim, 0, 1).squeeze(tdim)
    t_last = torch.where(m, time, -inf).max(dim=tdim).values
    has = (n > 0).float()
    span = torch.where(n > 0, t_last - t_first, torch.zeros_like(t_last))
    # target_dim indexes the reduced tensor (time axis already removed)
    avail_frac = has.mean(dim=target_dim, keepdim=True).expand_as(has)
    feats = torch.stack([torch.log1p(n), n / l_max, mean_dt, std_dt, span, has, avail_frac], dim=-1)
    return feats


class AdaptiveQuery(nn.Module):
    """q_adaptive = q_base + MLP(selected statistics)."""
    def __init__(self, d_model, kind):
        super().__init__()
        self.kind = kind
        self.idx = STAT_GROUPS[kind]
        self.mlp = nn.Sequential(nn.Linear(len(self.idx), d_model), nn.GELU(), nn.Linear(d_model, d_model))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, q_base, stats):
        s = stats[..., self.idx]
        if self.kind == 'random':          # control: same capacity, uninformative input
            s = torch.randn_like(s)
        return q_base + self.mlp(s)


class QuITEEmbedding(nn.Module):
    """Query-based embedding.

    Args:
        args: must contain `hid_dim`, `device`, `nhead`, `dropout`, plus
              `ndim` if family='variate' or `npatch` if family='patch'.
              Optional `adaptive_query` in {'none','full','density','gaps','avail','random'}.
        family: 'variate' (one query per variable) or 'patch'
                (one query per (patch, variable) pair).
    """
    def __init__(self, args, family='variate'):
        super().__init__()
        self.d_model = args.hid_dim
        self.device = args.device
        self.family = family

        self.val_emb = nn.Linear(1, self.d_model)
        self.te = LearnableTE(self.d_model)

        if family == 'variate':
            self.query_emb = nn.Embedding(args.ndim, self.d_model)
        elif family == 'patch':
            self.query_emb = nn.Embedding(args.npatch, self.d_model)
        else:
            raise ValueError(f"Unknown family: {family}")

        kind = getattr(args, 'adaptive_query', 'none') or 'none'
        self.adaptive = AdaptiveQuery(self.d_model, kind) if kind != 'none' else None

        self.attn = SelfAttentionBlock(self.d_model, args.nhead, dropout=args.dropout)

    def forward(self, X, time, mask):
        return self._variate(X, time, mask) if self.family == 'variate' \
                                            else self._patch(X, time, mask)

    # ----------------------------------------------------------------
    def _variate(self, X, time, mask):
        """Variate-level aggregation.

        Inputs:
            X    : (B, L, N)
            time : (B, L, N)
            mask : (B, L, N)
        Output:
            (B, N, D)
        """
        B, L, N = X.shape
        x_emb = self.val_emb(X.unsqueeze(-1))           # (B, L, N, D)
        t_emb = self.te(time.unsqueeze(-1))             # (B, L, N, D)

        z = (x_emb + t_emb).permute(0, 2, 1, 3).reshape(B * N, L, self.d_model)
        m = mask.permute(0, 2, 1).reshape(B * N, L)

        q = self.query_emb(torch.arange(N, device=self.device))
        q = q.unsqueeze(0).expand(B, -1, -1)                                  # (B, N, D)
        if self.adaptive is not None:
            stats = obs_stats(time, mask, tdim=1, target_dim=1)               # (B, N, F)
            q = self.adaptive(q, stats)
        q = q.reshape(B * N, 1, self.d_model)

        z = torch.cat([q, z], dim=1)
        m = torch.cat([torch.ones(B * N, 1, device=self.device), m], dim=1)

        out = self.attn(z, attn_mask=m.unsqueeze(-1))
        return out[:, 0, :].reshape(B, N, self.d_model)

    # ----------------------------------------------------------------
    def _patch(self, X, time, mask):
        """Patch-level aggregation.

        Inputs (after permute to N-on-axis-2):
            X    : (B, M, L_in, N) → internally permuted to (B, M, N, L_in)
            time : same
            mask : same
        Output:
            (B*N, M, D)
        """
        B, M, L_in, N = X.shape
        X = X.permute(0, 1, 3, 2)               # (B, M, N, L_in)
        time = time.permute(0, 1, 3, 2)
        mask = mask.permute(0, 1, 3, 2)

        x_emb = self.val_emb(X.unsqueeze(-1))   # (B, M, N, L_in, D)
        t_emb = self.te(time.unsqueeze(-1))

        q = self.query_emb(torch.arange(M, dtype=torch.long, device=self.device))
        q = q.reshape(1, M, 1, self.d_model).expand(B, -1, N, -1)             # (B, M, N, D)
        if self.adaptive is not None:
            stats = obs_stats(time, mask, tdim=3, target_dim=2)               # (B, M, N, F)
            q = self.adaptive(q, stats)
        q = q.unsqueeze(3)                                                    # (B, M, N, 1, D)

        z = torch.cat([q, x_emb + t_emb], dim=3)                # (B, M, N, L_in+1, D)
        z = z.reshape(B * M * N, L_in + 1, self.d_model)

        m = torch.cat([torch.ones(B, M, N, 1, device=self.device), mask], dim=3)
        m = m.reshape(B * M * N, L_in + 1)

        out = self.attn(z, attn_mask=m.unsqueeze(-1))
        out = out.reshape(B, M, N, L_in + 1, self.d_model)
        return out[:, :, :, 0, :].transpose(1, 2).reshape(B * N, M, self.d_model)
