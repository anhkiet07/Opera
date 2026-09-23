"""Synthetic irregular multivariate time series with a KNOWN causal graph (Phase 3).

Latent dynamics on a 48-"hour" window, step dt = 0.5 (96 steps), N = 10 variables:

    roots (no parents) : x_j(t+1) = 0.9 x_j(t) + 0.3 sin(w_j t + phi_j) + 0.1 eps
    others             : x_i(t+1) = 0.5 x_i(t) + sum_{j -> i} 0.9 tanh(x_j(t)) + 0.1 eps

Edges (j -> i):  0->1, 1->2, 0->3, 4->2, 5->6, 7->8; variable 9 is isolated noise.
"Y" = variable 2 (caused by 1 and 4, i.e. the chain X0->X1->Y and X4->Y from the plan).
Non-causal for Y: {5, 6, 7, 8, 9}.  Ancestors of Y: {0, 1, 4}.

Each variable is observed irregularly: every latent step is observed with probability p_obs[j]
(0.2 .. 0.4, variable specific), giving ~30 % density. Samples follow the (record_id, tt, vals, mask)
convention of the PhysioNet loader so the whole forecasting pipeline is reused unchanged.
"""
import os

import numpy as np
import torch

N_VARS = 10
EDGES = [(0, 1), (1, 2), (0, 3), (4, 2), (5, 6), (7, 8)]   # (cause, effect)
TARGET = 2
T_MAX, DT = 48.0, 0.5


def adjacency():
    A = np.zeros((N_VARS, N_VARS), dtype=np.float32)
    for j, i in EDGES:
        A[j, i] = 1.0
    return A


def simulate(n_samples, seed=0):
    rng = np.random.RandomState(seed)
    A = adjacency()
    steps = int(T_MAX / DT)
    t = np.arange(steps) * DT
    parents = [np.where(A[:, i] > 0)[0] for i in range(N_VARS)]
    roots = [i for i in range(N_VARS) if len(parents[i]) == 0]
    p_obs = rng.uniform(0.2, 0.4, size=N_VARS)
    records = []
    for s in range(n_samples):
        x = np.zeros((steps, N_VARS), dtype=np.float32)
        x[0] = rng.randn(N_VARS) * 0.5
        w = rng.uniform(0.2, 0.6, size=N_VARS)
        phi = rng.uniform(0, 2 * np.pi, size=N_VARS)
        for k in range(1, steps):
            eps = rng.randn(N_VARS) * 0.1
            for i in range(N_VARS):
                if i in roots:
                    x[k, i] = 0.9 * x[k - 1, i] + 0.3 * np.sin(w[i] * t[k] + phi[i]) + eps[i]
                else:
                    x[k, i] = 0.5 * x[k - 1, i] + 0.9 * np.tanh(x[k - 1, parents[i]]).sum() + eps[i]
        mask = (rng.rand(steps, N_VARS) < p_obs).astype(np.float32)
        mask[0] = 1.0                       # every sample starts observed (keeps tt[0] = 0 like PhysioNet)
        keep = mask.sum(1) > 0
        tt = torch.tensor(t[keep], dtype=torch.float32)
        vals = torch.tensor(x[keep] * mask[keep], dtype=torch.float32)
        m = torch.tensor(mask[keep], dtype=torch.float32)
        records.append((s, tt, vals, m))
    return records, A


class SyntheticCausal(object):
    """Cached generator; returns a list of (record_id, tt, vals, mask) on `device`."""
    def __init__(self, root, n_samples=4000, device=torch.device('cpu'), seed=0):
        os.makedirs(root, exist_ok=True)
        n_gen = min(n_samples, 4000)
        f = os.path.join(root, f'synthetic_n{n_gen}_seed{seed}.pt')
        if os.path.exists(f):
            data = torch.load(f, weights_only=False)
        else:
            records, A = simulate(n_gen, seed)
            data = {'records': records, 'adjacency': A}
            torch.save(data, f)
            np.save(os.path.join(root, 'adjacency.npy'), A)
        self.adjacency = data['adjacency']
        self.data = [(r, tt.to(device), v.to(device), m.to(device)) for r, tt, v, m in data['records']]

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)
