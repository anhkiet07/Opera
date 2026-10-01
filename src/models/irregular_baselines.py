"""Established irregular-time-series architectures exposed as OPERA backbones (CLAUDE.md section 18).

Three methods from the irregular-multivariate-time-series literature, re-implemented against
`BaseIrregularEncoder` so the OPERA machinery (augmentation, privileged distillation) wraps them
unchanged. None of them sees the observation-process code: they receive a history observation and
return a prediction, exactly like the QuITE backbones.

| key           | method     | family                  | reference                                        |
|---------------|------------|-------------------------|--------------------------------------------------|
| `grud`        | GRU-D      | recurrent + decay       | Che et al., Sci. Rep. 2018                       |
| `mtand`       | mTAND      | attention to a reference grid | Shukla & Marlin, ICLR 2021 (reml-lab/mTAN, MIT) |
| `warpformer`  | Warpformer | learned temporal warping + attention | Zhang et al., KDD 2023 (imJiawen/Warpformer, MIT) |

These are re-implementations following the published descriptions and the authors' reference code, adapted
to this repo's batch format. They are *not* the authors' code verbatim, and hyper-parameters follow this
benchmark's shared settings (hidden width, heads, layers) rather than each paper's tuned values, so absolute
numbers are not comparable to the published tables. What is comparable -- and what the benchmark asks -- is
the vanilla-vs-OPERA contrast *within* each backbone, since both arms share the architecture and budget.

Batch format. The datasets emit patch-shaped tensors `(B, M, L, N)` for values, timestamps and mask, where
timestamps are constant across the variable dimension. These methods all want a flat observation sequence, so
`_flatten` collapses `(M, L) -> T` and takes the shared time axis. Padding positions carry mask 0 and are
excluded by every reduction below.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseIrregularEncoder


# --------------------------------------------------------------------------------------- shared helpers
def _flatten(batch):
    """(B, M, L, N) -> values/mask (B, T, N) and shared times (B, T), T = M*L.

    Timestamps are constant across the variable axis (verified for both task families), so taking channel 0
    loses nothing. Returns times already zeroed where nothing is observed, so padded slots cannot move a
    time-difference computation.
    """
    x, tp, m = batch["observed_data"], batch["observed_tp"], batch["observed_mask"]
    if x.dim() == 4:
        B, M, L, N = x.shape
        x = x.reshape(B, M * L, N)
        m = m.reshape(B, M * L, N)
        tp = tp.reshape(B, M * L, N)[..., 0]
    else:
        tp = tp[..., 0] if tp.dim() == 3 else tp
    obs = (m.sum(-1) > 0).float()                     # (B, T) 1 where the step has any observation
    return x.float(), m.float(), tp.float() * obs, obs


def _time_scale(batch, times):
    """Normalise the time axis to ~[0, 1] using the batch's own span, for numerical stability."""
    hi = times.amax(dim=1, keepdim=True).clamp(min=1e-6)
    return times / hi


class _TimeEmbedding(nn.Module):
    """mTAND's learned time embedding: one linear channel plus (d-1) learned periodic channels."""

    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, d - 1)

    def forward(self, t):                              # (B, T) -> (B, T, d)
        t = t.unsqueeze(-1)
        return torch.cat([self.linear(t), torch.sin(self.periodic(t))], dim=-1)


class _IrregularBackbone(BaseIrregularEncoder):
    """Shared base for the three methods below.

    Records `self.task`, which the training loop reads to choose masked MSE vs cross-entropy
    (`src/training/objectives._loss_for`). A backbone that omits it silently defaults to "forecasting" and
    fails on the classification settings with a missing `data_to_predict` key.
    """

    #: these methods consume a flat observation sequence, not patch-shaped input
    patch_input = False

    def __init__(self, args):
        super().__init__()
        self.task = getattr(args, "task", "forecasting")


class _Head(nn.Module):
    """Task head shared by all three backbones.

    classification -> logits (B, C), optionally concatenating the static vector
    forecasting    -> values at arbitrary query times (B, L_pred, N), by conditioning a decoder on a time
                      embedding of each query time, so the backbone can be asked for any horizon.
    """

    def __init__(self, d_in, args, embed_time=32):
        super().__init__()
        self.task = args.task
        self.ndim = args.ndim
        if self.task == "classification":
            d = d_in + int(getattr(args, "d_static", 0) or 0)
            self.net = nn.Sequential(nn.Linear(d, d_in), nn.ReLU(), nn.Linear(d_in, args.n_class))
        else:
            self.temb = _TimeEmbedding(embed_time)
            self.net = nn.Sequential(nn.Linear(d_in + embed_time, d_in), nn.ReLU(),
                                     nn.Linear(d_in, args.ndim))

    def forward(self, z, batch):
        if self.task == "classification":
            static = batch.get("static")
            if static is not None:
                z = torch.cat([z, static.float()], dim=-1)
            return self.net(z)
        tq = batch["tp_to_predict"].float()            # (B, Lp)
        e = self.temb(tq / tq.amax(dim=1, keepdim=True).clamp(min=1e-6))
        z = z.unsqueeze(1).expand(-1, tq.shape[1], -1)
        return self.net(torch.cat([z, e], dim=-1))     # (B, Lp, N)


# --------------------------------------------------------------------------------------------- GRU-D
class GRUDEncoder(_IrregularBackbone):
    """GRU-D (Che et al. 2018): a GRU whose input and hidden state decay toward an empirical mean as the
    time since each variable was last observed grows.

        delta_t    time since that variable was last seen
        gamma_x    = exp(-relu(W_x delta + b_x))        input decay, per variable
        gamma_h    = exp(-relu(W_h delta + b_h))        hidden decay
        x_hat      = m*x + (1-m)*(gamma_x*x_last + (1-gamma_x)*x_mean)

    The empirical mean is a learned parameter rather than a dataset statistic, so that no information
    crosses from the training split into the module state.
    """

    def __init__(self, args):
        super().__init__(args)
        n, h = args.ndim, args.hid_dim
        self.ndim = n
        self.x_mean = nn.Parameter(torch.zeros(n))
        self.w_dg_x = nn.Linear(n, n)
        self.w_dg_h = nn.Linear(n, h)
        self.cell = nn.GRUCell(2 * n, h)               # [x_hat, mask]
        self.head = _Head(h, args)

    def forward(self, batch, return_repr=False, obs_descriptor=None):
        x, m, t, obs = _flatten(batch)
        t = _time_scale(batch, t)
        B, T, N = x.shape
        h = x.new_zeros(B, self.w_dg_h.out_features)
        x_last = self.x_mean.expand(B, N).clone()
        t_last = t.new_zeros(B, N)
        for i in range(T):
            xi, mi, ti = x[:, i], m[:, i], t[:, i].unsqueeze(-1)
            delta = (ti - t_last).clamp(min=0.0)
            gx = torch.exp(-F.relu(self.w_dg_x(delta)))
            gh = torch.exp(-F.relu(self.w_dg_h(delta)))
            x_hat = mi * xi + (1 - mi) * (gx * x_last + (1 - gx) * self.x_mean)
            h_new = self.cell(torch.cat([x_hat, mi], dim=-1), gh * h)
            # a padded step carries no observation, so it must not advance the state at all
            step = obs[:, i].unsqueeze(-1)
            h = step * h_new + (1 - step) * h
            # likewise, only advance the "last seen" value/time where something was observed
            x_last = torch.where(mi > 0, xi, x_last)
            t_last = torch.where(mi > 0, ti.expand_as(t_last), t_last)
        out = self.head(h, batch)
        return (out, h) if return_repr else out


# ----------------------------------------------------------------------------------------------- mTAND
class _MultiTimeAttention(nn.Module):
    """mTAND's multi-time attention (Shukla & Marlin 2021).

    Attention is computed between time embeddings, but the softmax is taken per *variable* with the
    observation mask applied, so an unobserved variable contributes nothing at that timestamp. Queries are a
    fixed reference grid, which turns a variable-length irregular series into a fixed-size representation.
    """

    def __init__(self, input_dim, nhidden, embed_time, num_heads):
        super().__init__()
        assert embed_time % num_heads == 0
        self.h, self.dk, self.dim = num_heads, embed_time // num_heads, input_dim
        self.q = nn.Linear(embed_time, embed_time)
        self.k = nn.Linear(embed_time, embed_time)
        self.out = nn.Linear(input_dim * num_heads, nhidden)

    def forward(self, query, key, value, mask):
        # query (B,R,E) key (B,T,E) value (B,T,D) mask (B,T,D)
        B, T, D = value.shape
        q = self.q(query).view(B, -1, self.h, self.dk).transpose(1, 2)      # (B,h,R,dk)
        k = self.k(key).view(B, -1, self.h, self.dk).transpose(1, 2)        # (B,h,T,dk)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)  # (B,h,R,T)
        scores = scores.unsqueeze(-1).expand(-1, -1, -1, -1, D)             # (B,h,R,T,D)
        scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        attn = torch.softmax(scores, dim=-2)
        x = torch.sum(attn * value.unsqueeze(1).unsqueeze(2), dim=-2)       # (B,h,R,D)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * D)
        return self.out(x)                                                  # (B,R,nhidden)


class MTANDEncoder(_IrregularBackbone):
    """mTAND: embed the irregular series onto a learned reference grid by time attention, then run a GRU
    over that grid. Values and their mask are concatenated, as in the reference implementation, so the
    module can distinguish "observed zero" from "missing"."""

    def __init__(self, args, n_ref=32, embed_time=32):
        super().__init__(args)
        n, h = args.ndim, args.hid_dim
        self.temb = _TimeEmbedding(embed_time)
        self.register_buffer("ref", torch.linspace(0.0, 1.0, n_ref).unsqueeze(0))
        self.att = _MultiTimeAttention(2 * n, h, embed_time, args.nhead)
        self.gru = nn.GRU(h, h, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * h, h)
        self.head = _Head(h, args)

    def forward(self, batch, return_repr=False, obs_descriptor=None):
        x, m, t, _ = _flatten(batch)
        t = _time_scale(batch, t)
        key = self.temb(t)                                   # (B,T,E)
        query = self.temb(self.ref.expand(x.shape[0], -1))   # (B,R,E)
        val = torch.cat([x * m, m], dim=-1)                  # (B,T,2N)
        mask = torch.cat([m, m], dim=-1)
        out = self.att(query, key, val, mask)                # (B,R,h)
        out, _ = self.gru(out)
        z = self.proj(out).mean(dim=1)                       # (B,h) pooled over the reference grid
        pred = self.head(z, batch)
        return (pred, z) if return_repr else pred


# ------------------------------------------------------------------------------------------ Warpformer
class WarpformerEncoder(_IrregularBackbone):
    """Warpformer (Zhang et al. 2023): represent each observation as a (value, variable, time) token, learn
    a monotone warping of the sequence onto a coarser set of segments, and attend at the warped resolution.

    The warping follows the reference implementation: a scoring network emits a non-negative increment per
    position, the masked cumulative sum gives a monotone alignment `gamma` normalised to [0, 1], and a
    triangular kernel turns `gamma` into a soft assignment of positions to `n_warp` segments. Attention then
    runs over segments rather than raw timestamps, which is what makes the cost independent of how densely
    the series happens to be sampled.
    """

    def __init__(self, args, n_warp=16, embed_time=32):
        super().__init__(args)
        n, h = args.ndim, args.hid_dim
        self.n_warp = n_warp
        self.value_enc = nn.Linear(1, h)
        self.var_emb = nn.Parameter(torch.randn(n, h) * 0.02)
        self.temb = _TimeEmbedding(embed_time)
        self.time_proj = nn.Linear(embed_time, h)
        self.score = nn.Sequential(nn.Linear(h, 2 * h), nn.ReLU(), nn.Linear(2 * h, 1, bias=False))
        layer = nn.TransformerEncoderLayer(h, args.nhead, 2 * h, args.dropout,
                                           batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, max(1, args.nlayer - 1))
        self.head = _Head(h, args)

    def forward(self, batch, return_repr=False, obs_descriptor=None):
        x, m, t, obs = _flatten(batch)
        t = _time_scale(batch, t)
        B, T, N = x.shape
        # (value, variable, time) tokens, summed over variables present at each step
        v = self.value_enc((x * m).unsqueeze(-1))                    # (B,T,N,h)
        v = (v + self.var_emb) * m.unsqueeze(-1)
        # Mean, not sum, over the observed variables. A sum makes the token magnitude scale with how many
        # variables happen to be recorded at that step, which diverges on the wide settings (P19 has 34
        # variables, P12 36) and produced non-finite logits. A masked mean is invariant to that count,
        # which is also the right behaviour when the observation process itself changes how many variables
        # are present -- the quantity this benchmark deliberately varies.
        denom = m.sum(dim=2, keepdim=True).clamp(min=1.0)            # (B,T,1)
        tok = v.sum(dim=2) / denom + self.time_proj(self.temb(t))    # (B,T,h)
        tok = tok * obs.unsqueeze(-1)

        # monotone alignment gamma in [0,1] (reference: masked cumsum of non-negative increments)
        dg = F.softplus(self.score(tok).squeeze(-1)) * obs           # (B,T)
        dg = dg.clone()
        dg[:, 0] = 0.0
        gamma = torch.cumsum(dg, dim=-1) * obs
        gmax = gamma.amax(dim=1, keepdim=True)
        gamma = gamma / torch.where(gmax > 0, gmax, torch.ones_like(gmax))

        # triangular soft assignment of positions to n_warp segments
        centres = torch.linspace(0.0, 1.0, self.n_warp, device=x.device).view(1, self.n_warp, 1)
        w = F.relu(1.0 - (self.n_warp - 1) * (gamma.unsqueeze(1) - centres).abs())   # (B,S,T)
        w = w * obs.unsqueeze(1)
        seg = torch.bmm(w, tok) / w.sum(-1, keepdim=True).clamp(min=1e-6)            # (B,S,h)

        z = self.tr(seg).mean(dim=1)
        pred = self.head(z, batch)
        return (pred, z) if return_repr else pred


BACKBONES = {"grud": GRUDEncoder, "mtand": MTANDEncoder, "warpformer": WarpformerEncoder}
