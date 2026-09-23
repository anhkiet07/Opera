"""Estimate a variable-level structural graph for PhysioNet from the TRAINING split only (Phase 3, real data).

Pipeline (the plan's "regular-grid reconstruction -> sparse lagged regression -> threshold"):
    irregular training records -> hourly bins (mean of observed values), min-max normalised with the training
    statistics -> forward-fill then fill remaining gaps with the per-variable training mean -> for every target i,
    lasso regression of x_i(t) on all x_j(t-1) (standardised) -> keep the top-k parents by |coef| above a
    threshold -> A[j, i] = 1.
Caveat (stated in the paper): the grid reconstruction introduces artificial values, and lagged regression finds
predictive temporal dependencies, not a causal DAG.

The split replicates data/parse.py exactly (same sklearn seed, same -n cap), so the graph never sees the
validation / test records of the run that uses it.

Usage: python estimate_graph.py [-n 1500] [--topk 3] [--min_coef 0.02] [--out data/physionet/estimated_graph_n1500.npy]
"""
import argparse
import os

import numpy as np
import torch
from sklearn import model_selection
from sklearn.linear_model import Lasso

from data import DATA_ROOT
from data.physionet import PhysioNet, get_data_min_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=int(1e8))
    ap.add_argument('--bin', type=float, default=1.0, help='grid step in hours')
    ap.add_argument('--topk', type=int, default=3)
    ap.add_argument('--min_coef', type=float, default=0.02)
    ap.add_argument('--alpha', type=float, default=0.002)
    ap.add_argument('--out', type=str, default=None)
    a = ap.parse_args()

    ds = PhysioNet(os.path.join(DATA_ROOT, 'physionet'), quantization=0.0, download=True, n_samples=a.n,
                   device=torch.device('cpu'))
    seen, test = model_selection.train_test_split(ds, train_size=0.8, random_state=42, shuffle=True)
    train, val = model_selection.train_test_split(seen, train_size=0.75, random_state=42, shuffle=False)
    dmin, dmax, _ = get_data_min_max(seen, torch.device('cpu'))
    dmin, dmax = dmin.numpy(), dmax.numpy()
    scale = np.where(dmax - dmin > 0, dmax - dmin, 1.0)
    N = len(PhysioNet.params)
    steps = int(48 / a.bin)

    # ---- grid reconstruction ----
    grids = []
    for _, tt, vals, mask in train:
        tt, vals, mask = tt.numpy(), vals.numpy(), mask.numpy()
        g = np.full((steps, N), np.nan, dtype=np.float32)
        cnt = np.zeros((steps, N))
        idx = np.minimum((tt / a.bin).astype(int), steps - 1)
        for k in range(len(tt)):
            m = mask[k] > 0
            g[idx[k], m] = np.where(np.isnan(g[idx[k], m]), 0, g[idx[k], m]) + vals[k, m]
            cnt[idx[k], m] += 1
        g = np.where(cnt > 0, g / np.maximum(cnt, 1), np.nan)
        g = (g - dmin) / scale
        for t in range(1, steps):                                   # forward fill
            g[t] = np.where(np.isnan(g[t]), g[t - 1], g[t])
        grids.append(g)
    G = np.stack(grids)                                             # (R, T, N)
    col_mean = np.nanmean(G.reshape(-1, N), axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    G = np.where(np.isnan(G), col_mean, G)

    # ---- sparse lagged regression ----
    X = G[:, :-1, :].reshape(-1, N)
    Y = G[:, 1:, :].reshape(-1, N)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    A = np.zeros((N, N), dtype=np.float32)
    coefs = np.zeros((N, N))
    for i in range(N):
        y = (Y[:, i] - Y[:, i].mean()) / (Y[:, i].std() + 1e-8)
        model = Lasso(alpha=a.alpha, max_iter=5000).fit(Xs, y)
        c = model.coef_.copy()
        c[i] = 0                                                    # exclude self-lag
        coefs[:, i] = c
        order = np.argsort(-np.abs(c))
        for j in order[:a.topk]:
            if abs(c[j]) >= a.min_coef:
                A[j, i] = 1.0
    out = a.out or os.path.join(DATA_ROOT, 'physionet', f'estimated_graph_n{min(a.n, 12000)}.npy')
    np.save(out, A)
    np.save(out.replace('.npy', '_coefs.npy'), coefs)
    names = PhysioNet.params
    print(f'{len(train)} training records, grid {steps}x{N}; {int(A.sum())} edges -> {out}')
    for i in range(N):
        par = [f'{names[j]}({coefs[j, i]:+.2f})' for j in range(N) if A[j, i] > 0]
        if par:
            print(f'  {names[i]:12s} <- ' + ', '.join(par))


if __name__ == '__main__':
    main()
