"""Phase 3: causal structure as an inductive bias for the variable-level attention.

Baseline attention (iTransformer, variable tokens):  A_ij = softmax(Q_i K_j^T / sqrt(d))
Causal QuITE:                                        A_ij = softmax(Q_i K_j^T / sqrt(d) + lambda * M_ij)

M_ij = 1 when variable j causally influences variable i (edge j -> i) or i == j, else 0, so a variable's
token is encouraged to attend to its causes. Modes:
    soft     : additive bias lambda * M
    hard     : -inf where M == 0 (attend only to causes and self)
    learned  : free N x N parameter, zero-initialised (no graph used) - "learned causal matrix"
Graph sources (args.causal_graph), all derived from the dataset's true adjacency A (A[j, i] = 1 for j -> i):
    true, identity (self only), random (same edge count, random positions), shuffled (variables permuted),
    noisy:p (fraction p of off-diagonal entries flipped)
"""
import numpy as np
import torch
import torch.nn as nn


def build_M(adj, source, seed=0):
    """adj: (N, N) numpy, adj[j, i] = 1 for edge j -> i. Returns M with M[i, j] = 1 for j -> i, plus diagonal."""
    N = adj.shape[0]
    rng = np.random.RandomState(seed)
    src = source.split(':')
    kind, p = src[0], (float(src[1]) if len(src) > 1 else 0.0)
    M = adj.T.astype(np.float32).copy()
    np.fill_diagonal(M, 0)
    if kind == 'true':
        pass
    elif kind == 'identity':
        M[:] = 0
    elif kind == 'random':
        n_edges = int(M.sum())
        M[:] = 0
        off = [(i, j) for i in range(N) for j in range(N) if i != j]
        for k in rng.choice(len(off), n_edges, replace=False):
            M[off[k]] = 1
    elif kind == 'shuffled':
        perm = rng.permutation(N)
        M = M[perm][:, perm]
    elif kind == 'noisy':
        flip = rng.rand(N, N) < p
        np.fill_diagonal(flip, False)
        M = np.where(flip, 1 - M, M)
    else:
        raise ValueError(f'Unknown causal graph source: {source}')
    np.fill_diagonal(M, 1)
    return torch.tensor(M, dtype=torch.float32)


class CausalBias(nn.Module):
    """Produces the (N, N) additive bias for the variable-level attention, or None."""
    def __init__(self, args, adj=None):
        super().__init__()
        self.mode = getattr(args, 'causal_mode', 'none') or 'none'
        self.lam = float(getattr(args, 'causal_lambda', 1.0))
        N = args.ndim
        if self.mode == 'none':
            self.register_buffer('M', torch.zeros(0))
        elif self.mode == 'learned':
            self.M = nn.Parameter(torch.zeros(N, N))
        else:
            assert adj is not None, 'soft/hard causal bias needs a graph'
            self.register_buffer('M', build_M(adj, getattr(args, 'causal_graph', 'true'), seed=args.seed))

    def forward(self):
        if self.mode == 'none':
            return None
        if self.mode == 'learned':
            return self.M
        if self.mode == 'soft':
            return self.lam * self.M
        if self.mode == 'hard':
            return torch.where(self.M > 0, torch.zeros_like(self.M), torch.full_like(self.M, -1e9))
        raise ValueError(self.mode)
