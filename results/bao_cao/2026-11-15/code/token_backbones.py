"""DLinear, SegRNN and Crossformer as encoders over QuITE patch tokens.

Both receive the patch-family token grid that every QuITE backbone sees, (B*N, M, D): M patch tokens per
variable, D channels each, and return a grid of the same shape for the shared forecasting decoder. They are
re-implementations of each method's core mechanism on that grid, not ports of the authors' code.

DLinear (Zeng et al., AAAI 2023): split the sequence into a moving-average trend and a seasonal remainder and
map each along the time axis with one linear layer shared across channels ("individual = False"). Here the
time axis is the M patch tokens, so on the short patch grids of this benchmark (M = 2..16) the decomposition
is coarse and the model is close to a learned linear mixing of patches.

SegRNN (Lin et al., 2023): replace point-wise recurrence with recurrence over segments: each segment is
embedded, and a GRU steps once per segment, so the recurrence length is the number of segments rather than
the number of time steps. QuITE's patch tokens are those segment embeddings, so the encoder is a GRU over the
M tokens of each variable (channel-independent, weights shared across variables). The reference decodes from
the last hidden state with per-position queries; here every hidden state is handed to the shared query
decoder, which plays that role for arbitrary forecast times.

Crossformer (Zhang & Yan, ICLR 2023): a two-stage attention layer. Cross-time: each variable attends over its
own segments. Cross-dimension: per segment, a small set of learned router tokens gathers from all variables
and the variables read back from the routers, which keeps the cost linear in the number of variables. The
QuITE patches play the role of Crossformer's segments. The hierarchical segment merging between layers is
left out, because the grids here hold at most 16 segments.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DLinearEncoder(nn.Module):
    def __init__(self, n_tokens, max_kernel=25):
        super().__init__()
        self.kernel = min(max_kernel, 2 * (n_tokens // 2) + 1)       # odd, so the trend keeps length M
        self.seasonal = nn.Linear(n_tokens, n_tokens)
        self.trend = nn.Linear(n_tokens, n_tokens)

    def forward(self, x):                                            # (B*N, M, D)
        xt = x.transpose(1, 2)                                       # (B*N, D, M)
        pad = (self.kernel - 1) // 2
        trend = F.avg_pool1d(F.pad(xt, (pad, pad), mode="replicate"), self.kernel, stride=1)
        return (self.seasonal(xt - trend) + self.trend(trend)).transpose(1, 2)


class SegRNNEncoder(nn.Module):
    def __init__(self, d, n_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(d, d, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)

    def forward(self, x):                                            # (B*N, M, D) -> (B*N, M, D)
        return self.gru(x)[0]


class _MLP(nn.Sequential):
    def __init__(self, d, dropout):
        super().__init__(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))


class TwoStageAttention(nn.Module):
    """One Crossformer encoder layer on a (B, N, M, D) grid."""

    def __init__(self, d, n_heads, n_routers, dropout):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.router = nn.Parameter(torch.randn(n_routers, d) * 0.02)
        self.dim_send = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.dim_recv = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.mlp1, self.mlp2 = _MLP(d, dropout), _MLP(d, dropout)
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(4))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                            # (B, N, M, D)
        B, N, M, D = x.shape
        # pre-norm residual blocks: the post-norm form sat on a plateau for ~14 epochs before learning,
        # long enough for early stopping to end a run there
        t = x.reshape(B * N, M, D)
        h = self.norms[0](t)
        t = t + self.drop(self.time_attn(h, h, h, need_weights=False)[0])
        t = t + self.drop(self.mlp1(self.norms[1](t)))
        s = t.view(B, N, M, D).transpose(1, 2).reshape(B * M, N, D)   # one row per segment
        h = self.norms[2](s)
        r = self.router.expand(B * M, -1, -1)
        gathered = self.dim_send(r, h, h, need_weights=False)[0]
        s = s + self.drop(self.dim_recv(h, gathered, gathered, need_weights=False)[0])
        s = s + self.drop(self.mlp2(self.norms[3](s)))
        return s.view(B, M, N, D).transpose(1, 2)


class CrossformerEncoder(nn.Module):
    def __init__(self, d, n_heads, n_layers, dropout, n_routers=10):
        super().__init__()
        self.layers = nn.ModuleList(TwoStageAttention(d, n_heads, n_routers, dropout) for _ in range(n_layers))

    def forward(self, x, B, N):                                      # (B*N, M, D) -> (B*N, M, D)
        M, D = x.shape[1], x.shape[2]
        x = x.reshape(B, N, M, D)
        for layer in self.layers:
            x = layer(x)
        return x.reshape(B * N, M, D)
