"""Established irregular-time-series architectures exposed as OPERA backbones (CLAUDE.md section 18).

Six methods from the irregular-multivariate-time-series literature, re-implemented against
`BaseIrregularEncoder` so the OPERA machinery (augmentation, privileged distillation) wraps them
unchanged. None of them sees the observation-process code: they receive a history observation and
return a prediction, exactly like the QuITE backbones.

| key           | method     | family                  | reference                                        |
|---------------|------------|-------------------------|--------------------------------------------------|
| `grud`        | GRU-D      | recurrent + decay       | Che et al., Sci. Rep. 2018                       |
| `mtand`       | mTAND      | attention to a reference grid | Shukla & Marlin, ICLR 2021 (reml-lab/mTAN, MIT) |
| `warpformer`  | Warpformer | learned temporal warping + attention | Zhang et al., KDD 2023 (imJiawen/Warpformer, MIT) |
| `tpatchgnn`   | t-PatchGNN | time patches + transformer + adaptive graph | Zhang et al., ICML 2024 |
| `grafiti`     | GraFITi    | bipartite variable-time graph, edge regression | Yalavarthy et al., AAAI 2024 |
| `cru`         | CRU        | latent linear SDE + Kalman filter | Schirmer et al., ICML 2022 |

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


# ------------------------------------------------------------------------------- shared by the three below
# t-PatchGNN, GraFITi and CRU forecast by querying a time. The datasets already place history and query times
# on one axis in [0, 1] (history first, horizon after), so these three read raw times and never rescale
# them per batch: a query at 0.9 must stay after a last observation at 0.7.

def _masked_softmax(scores, mask, dim):
    """Softmax over `dim` restricted to mask == 1; a slice with no valid entry returns all zeros."""
    scores = scores.masked_fill(mask == 0, -1e9)
    return torch.softmax(scores, dim=dim) * mask


class _QueryDecoder(nn.Module):
    """Per-variable forecasting head: value of variable n at query time q = MLP([h_n, time_emb(q)])."""

    def __init__(self, h, embed_time=32):
        super().__init__()
        self.temb = _TimeEmbedding(embed_time)
        self.net = nn.Sequential(nn.Linear(h + embed_time, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, hv, tq):                          # hv (B,N,h), tq (B,Lp) -> (B,Lp,N)
        B, N, h = hv.shape
        e = self.temb(tq)                               # (B,Lp,E)
        z = torch.cat([hv.unsqueeze(1).expand(-1, tq.shape[1], -1, -1),
                       e.unsqueeze(2).expand(-1, -1, N, -1)], dim=-1)
        return self.net(z).squeeze(-1)


# ------------------------------------------------------------------------------------------- t-PatchGNN
class TPatchGNNEncoder(_IrregularBackbone):
    """t-PatchGNN (Zhang et al., ICML 2024): split each variable's series into time patches, encode every
    (variable, patch) with a time-aware convolution over its own irregular points, then alternate a
    Transformer across a variable's patches with a graph network across variables whose adjacency is learned
    and varies with time. A variable's patches are pooled into one state that a query decoder reads.

        TTCN       filter_d = softmax over the patch's observed points of MLP_d([x, time_emb(t)]);
                   patch_d  = sum_points filter_d * [x, time_emb(t)]
        graph      A_m = softmax(relu((E1 + H_m W1)(E2 + H_m W2)^T)), 2-hop propagation, per patch m

    The patches are this repo's own patch axis (data.patch_size / stride), which is the windowing the
    reference code uses. Deviation from the reference: a variable's patches are pooled by a mean instead of
    flattened, so the module does not hard-code the patch count.
    """

    def __init__(self, args, embed_time=7, node_dim=10):
        super().__init__(args)
        n, h = args.ndim, args.hid_dim
        d_in = 1 + embed_time
        self.h, self.d_in = h, d_in
        self.temb = _TimeEmbedding(embed_time)
        self.filter_gen = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h * d_in))
        self.patch_proj = nn.Linear(h + 1, h)          # + "patch has any observation" indicator
        self.n_layer = max(1, args.nlayer)
        self.tr = nn.ModuleList(
            nn.TransformerEncoderLayer(h, args.nhead, 2 * h, args.dropout, batch_first=True, norm_first=True)
            for _ in range(self.n_layer))
        self.e1 = nn.Parameter(torch.randn(n, node_dim) * 0.1)
        self.e2 = nn.Parameter(torch.randn(n, node_dim) * 0.1)
        self.adj_w1 = nn.ModuleList(nn.Linear(h, node_dim) for _ in range(self.n_layer))
        self.adj_w2 = nn.ModuleList(nn.Linear(h, node_dim) for _ in range(self.n_layer))
        self.gconv = nn.ModuleList(nn.Linear(3 * h, h) for _ in range(self.n_layer))
        self.gnorm = nn.ModuleList(nn.LayerNorm(h) for _ in range(self.n_layer))
        if self.task == "classification":
            self.head = _Head(h, args)
        else:
            self.decoder = _QueryDecoder(h)

    @staticmethod
    def _patches(batch):
        x, tp, m = batch["observed_data"].float(), batch["observed_tp"].float(), batch["observed_mask"].float()
        if x.dim() == 3:                                # flat input: one patch
            x, m = x.unsqueeze(1), m.unsqueeze(1)
            tp = tp.unsqueeze(1) if tp.dim() == 3 else tp.unsqueeze(1).unsqueeze(-1).expand_as(x)
        if tp.dim() == 3:
            tp = tp.unsqueeze(-1).expand_as(x)
        return x, tp, m                                 # (B,M,L,N)

    def _pos(self, M, device):
        pos = torch.arange(M, device=device).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, self.h, 2, device=device).float() * (-math.log(10000.0) / self.h))
        pe = torch.zeros(M, self.h, device=device)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        return pe

    def forward(self, batch, return_repr=False, obs_descriptor=None):
        x, tp, m = self._patches(batch)
        B, M, L, N = x.shape
        z = torch.cat([x.unsqueeze(-1), self.temb(tp)], dim=-1) * m.unsqueeze(-1)    # (B,M,L,N,d_in)
        z = z.permute(0, 3, 1, 2, 4)                                                  # (B,N,M,L,d_in)
        mk = m.permute(0, 3, 1, 2)                                                    # (B,N,M,L)
        filt = self.filter_gen(z).view(B, N, M, L, self.h, self.d_in)
        filt = _masked_softmax(filt, mk[..., None, None], dim=3)
        hp = (filt * z.unsqueeze(-2)).sum(dim=(3, 5))                                 # (B,N,M,h)
        has = (mk.sum(-1) > 0).float().unsqueeze(-1)                                  # (B,N,M,1)
        hp = self.patch_proj(torch.cat([hp * has, has], dim=-1)) + self._pos(M, x.device)

        for i in range(self.n_layer):
            hp = self.tr[i](hp.reshape(B * N, M, self.h)).view(B, N, M, self.h)
            hm = hp.transpose(1, 2)                                                   # (B,M,N,h)
            a = torch.einsum("bmnd,bmkd->bmnk", self.e1 + self.adj_w1[i](hm), self.e2 + self.adj_w2[i](hm))
            a = torch.softmax(F.relu(a), dim=-1)
            a1 = a @ hm
            hm = self.gnorm[i](hm + F.relu(self.gconv[i](torch.cat([hm, a1, a @ a1], dim=-1))))
            hp = hm.transpose(1, 2)

        # every patch, empty ones included: a variable with no observation at all is represented only by what
        # the graph layers carried into its patches from the other variables, so masking empty patches out
        # would zero it and make it impossible to forecast from its neighbours
        hv = hp.mean(dim=2)                                                           # (B,N,h)
        zr = hv.mean(dim=1)
        if self.task == "classification":
            pred = self.head(zr, batch)
        else:
            pred = self.decoder(hv, batch["tp_to_predict"].float())
        return (pred, zr) if return_repr else pred


# ---------------------------------------------------------------------------------------------- GraFITi
class GraFITiEncoder(_IrregularBackbone):
    """GraFITi (Yalavarthy et al., AAAI 2024): forecasting as edge regression on a bipartite graph between
    variable nodes and time nodes. Every observed value is an edge (time k, variable n) carrying the value;
    every value to forecast is an edge carrying a "target" flag and no value. Each layer lets a variable
    node attend over its edges, a time node attend over its edges, and refreshes every edge from its two
    endpoints. The forecast is read off the target edges.

    The graph is held densely as (B, time nodes, variables, h) with an edge mask. Target edges are placed at
    every (query time, variable) pair rather than only where the target mask is 1, so the input does not
    reveal which future values exist; the loss masks the rest, as for every other backbone here.
    """

    def __init__(self, args, embed_time=32):
        super().__init__(args)
        n, h = args.ndim, args.hid_dim
        self.h, self.nh = h, args.nhead
        self.var_emb = nn.Parameter(torch.randn(n, h) * 0.1)
        self.temb = _TimeEmbedding(embed_time)
        self.time_proj = nn.Linear(embed_time, h)
        self.edge_in = nn.Linear(2, h)                  # [value, is_target]
        L = max(1, args.nlayer)
        self.var_q = nn.ModuleList(nn.Linear(h, h) for _ in range(L))
        self.var_kv = nn.ModuleList(nn.Linear(2 * h, 2 * h) for _ in range(L))
        self.time_q = nn.ModuleList(nn.Linear(h, h) for _ in range(L))
        self.time_kv = nn.ModuleList(nn.Linear(2 * h, 2 * h) for _ in range(L))
        self.edge_mlp = nn.ModuleList(nn.Sequential(nn.Linear(3 * h, h), nn.ReLU(), nn.Linear(h, h))
                                      for _ in range(L))
        self.norms = nn.ModuleList(nn.ModuleList(nn.LayerNorm(h) for _ in range(3)) for _ in range(L))
        self.out = nn.Linear(h, 1)
        if self.task == "classification":
            self.head = _Head(h, args)

    def _attend(self, q, kv, mask):
        """q (B,P,h) attends over kv (B,P,S,2h) under mask (B,P,S): multi-head, one query per node."""
        B, P, S, _ = kv.shape
        dk = self.h // self.nh
        k, v = kv.split(self.h, dim=-1)
        q = q.view(B, P, self.nh, dk)
        k, v = k.view(B, P, S, self.nh, dk), v.view(B, P, S, self.nh, dk)
        s = torch.einsum("bphd,bpshd->bphs", q, k) / math.sqrt(dk)
        a = _masked_softmax(s, mask.unsqueeze(2), dim=-1)
        return torch.einsum("bphs,bpshd->bphd", a, v).reshape(B, P, self.h)

    def forward(self, batch, return_repr=False, obs_descriptor=None):
        x, m, t, _ = _flatten(batch)
        B, T, N = x.shape
        if self.task == "classification":
            tq = t.new_zeros(B, 0)
        else:
            tq = batch["tp_to_predict"].float()
        Q = tq.shape[1]
        times = torch.cat([t, tq], dim=1)                                             # (B,K)
        val = torch.cat([x * m, x.new_zeros(B, Q, N)], dim=1)
        is_tgt = torch.cat([x.new_zeros(B, T, N), x.new_ones(B, Q, N)], dim=1)
        emask = torch.cat([m, x.new_ones(B, Q, N)], dim=1)                            # (B,K,N)

        e = self.edge_in(torch.stack([val, is_tgt], dim=-1)) * emask.unsqueeze(-1)    # (B,K,N,h)
        vn = self.var_emb.expand(B, -1, -1)                                           # (B,N,h)
        tn = self.time_proj(self.temb(times))                                         # (B,K,h)
        for i, (n1, n2, n3) in enumerate(self.norms):
            kv_v = self.var_kv[i](torch.cat([e, tn.unsqueeze(2).expand(-1, -1, N, -1)], dim=-1))
            vn = n1(vn + self._attend(self.var_q[i](vn), kv_v.transpose(1, 2), emask.transpose(1, 2)))
            kv_t = self.time_kv[i](torch.cat([e, vn.unsqueeze(1).expand(-1, times.shape[1], -1, -1)], dim=-1))
            tn = n2(tn + self._attend(self.time_q[i](tn), kv_t, emask))
            upd = self.edge_mlp[i](torch.cat([e, vn.unsqueeze(1).expand_as(e), tn.unsqueeze(2).expand_as(e)], -1))
            e = n3(e + upd) * emask.unsqueeze(-1)

        zr = vn.mean(dim=1)
        if self.task == "classification":
            pred = self.head(zr, batch)
        else:
            pred = self.out(e[:, T:]).squeeze(-1)                                     # (B,Q,N)
        return (pred, zr) if return_repr else pred


# -------------------------------------------------------------------------------------------------- CRU
class CRUEncoder(_IrregularBackbone):
    """CRU, Continuous Recurrent Units (Schirmer et al., ICML 2022): a Kalman filter in a learned latent
    space whose state follows a linear SDE between observations.

        latent state   mean mu (2*lod), covariance in CRU's factorised form: diagonal upper, lower and side
                       blocks (sigma_u, sigma_l, sigma_s); only the upper half is observed (H = [I 0])
        transition     A = sum_k alpha_k(mu) A_k  (locally linear, K basis matrices)
        predict        mu <- exp(A dt) mu ;  Sigma <- Phi Sigma Phi^T + Q dt, then re-factorised
        update         encoder maps each step's (values, mask) to a latent observation w with variance r,
                       then the closed-form Kalman update of the factorised covariance

    Forecasting runs the predict step from the last filtered state to each query time in order and decodes
    the latent mean. dt is on the datasets' shared [0, 1] axis, unscaled: stretching it (e.g. by the sequence
    length) makes a decaying state vanish before a long horizon is reached, and the decoder sees zeros.
    """

    def __init__(self, args, n_basis=15):
        super().__init__(args)
        n, h = args.ndim, args.hid_dim
        self.lod = h // 2
        d = 2 * self.lod
        self.enc = nn.Sequential(nn.Linear(2 * n, h), nn.ReLU(), nn.Linear(h, 2 * self.lod))
        # start every basis matrix near -I so the latent state decays between observations; a random init
        # has eigenvalues with positive real part and exp(A dt) overflows over a long forecast horizon
        self.basis = nn.Parameter(torch.randn(n_basis, d, d) * 0.05 / math.sqrt(d) - torch.eye(d))
        self.coef = nn.Linear(d, n_basis)
        self.log_q = nn.Parameter(torch.full((d,), -2.0))
        self.dec = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, n))
        if self.task == "classification":
            self.head = _Head(d, args)

    def _predict(self, mu, su, sl, ss, dt):
        lod = self.lod
        A = torch.einsum("bk,kij->bij", torch.softmax(self.coef(mu), -1), self.basis)
        phi = torch.matrix_exp(A * dt.view(-1, 1, 1))
        mu = torch.einsum("bij,bj->bi", phi, mu)
        cov = torch.cat([torch.cat([torch.diag_embed(su), torch.diag_embed(ss)], -1),
                         torch.cat([torch.diag_embed(ss), torch.diag_embed(sl)], -1)], -2)
        cov = phi @ cov @ phi.transpose(1, 2) + torch.diag_embed(F.softplus(self.log_q)) * dt.view(-1, 1, 1)
        dg = torch.diagonal(cov, dim1=1, dim2=2)
        side = torch.diagonal(cov[:, :lod, lod:], dim1=1, dim2=2)
        return mu, dg[:, :lod], dg[:, lod:], side

    def _update(self, mu, su, sl, ss, w, r):
        lod = self.lod
        denom = su + r
        ku, kl = su / denom, ss / denom
        resid = w - mu[:, :lod]
        mu = torch.cat([mu[:, :lod] + ku * resid, mu[:, lod:] + kl * resid], -1)
        return mu, (1 - ku) * su, (sl - kl * ss).clamp(min=1e-6), (1 - ku) * ss

    def forward(self, batch, return_repr=False, obs_descriptor=None):
        x, m, t, obs = _flatten(batch)
        B, T, N = x.shape
        wr = self.enc(torch.cat([x * m, m], dim=-1))
        w_all, r_all = wr[..., :self.lod], F.softplus(wr[..., self.lod:]) + 1e-4
        mu = x.new_zeros(B, 2 * self.lod)
        su, sl, ss = x.new_ones(B, self.lod), x.new_ones(B, self.lod), x.new_zeros(B, self.lod)
        t_last = t.new_zeros(B)
        for i in range(T):
            step = obs[:, i]
            dt = (t[:, i] - t_last).clamp(min=0.0) * step
            p = self._predict(mu, su, sl, ss, dt)
            u = self._update(*p, w_all[:, i], r_all[:, i])
            keep = step.unsqueeze(-1)
            mu, su, sl, ss = (keep * a + (1 - keep) * b for a, b in zip(u, (mu, su, sl, ss)))
            t_last = torch.where(step > 0, t[:, i], t_last)

        if self.task == "classification":
            pred = self.head(mu, batch)
        else:
            tq = batch["tp_to_predict"].float()
            outs, cur, tc = [], (mu, su, sl, ss), t_last
            for j in range(tq.shape[1]):
                dt = (tq[:, j] - tc).clamp(min=0.0)
                cur = self._predict(*cur, dt)
                tc = torch.maximum(tc, tq[:, j])
                outs.append(self.dec(cur[0]))
            pred = torch.stack(outs, dim=1)                                           # (B,Lp,N)
        return (pred, mu) if return_repr else pred


BACKBONES = {"grud": GRUDEncoder, "mtand": MTANDEncoder, "warpformer": WarpformerEncoder,
             "tpatchgnn": TPatchGNNEncoder, "grafiti": GraFITiEncoder, "cru": CRUEncoder}
