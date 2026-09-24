"""Irregular time-series classification datasets for OPERA: P12, P19, PAM (OPERA section 24).

Preprocessing and splits are the unmodified Raindrop pipeline already in this repo (`utils.get_data_split`,
`utils.tensorize_normalize_extract_feature_*`), including the minority-class upsampling used for the binary
tasks, so results stay comparable to the published protocol:

| dataset | samples | variables | classes | static | window | patch |
|---------|---------|-----------|---------|--------|--------|-------|
| P12     | 11,988  | 36        | 2 (in-hospital mortality) | 9 | 48 h | 6 |
| P19     | 38,803  | 34        | 2 (sepsis)                | 6 | 60 h | 3.75 |
| PAM     |  5,333  | 17        | 8 (activity)              | 0 | 60 u | 10 |

Tensors are held in memory and batched here, so this module exposes the same interface as the forecasting
registry: infinite generators of batch dicts. The observation-process machinery is task-agnostic because it
only edits `observed_data` / `observed_mask`, which have the same (B, M, L, N) layout as forecasting batches.
"""
import argparse
import math
import os

import numpy as np
import torch

from data import DATA_ROOT

CFG = {
    "P12": dict(d_static=9, variables_num=36, n_class=2, history=48, patch_size=6, split="/splits/phy12_split{k}.npy"),
    "P19": dict(d_static=6, variables_num=34, n_class=2, history=60, patch_size=3.75, split="/splits/phy19_split{k}_new.npy"),
    "PAM": dict(d_static=0, variables_num=17, n_class=8, history=60, patch_size=10, split="/splits/PAMAP2_split_{k}.npy"),
}


def _tensorize(name, args, split_idx, device):
    from utils import (get_data_split, getStats, getStats_static,
                       tensorize_normalize_extract_feature_patch,
                       tensorize_normalize_extract_feature_pam_patch)
    base = os.path.join(DATA_ROOT, name)
    Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(
        base, CFG[name]["split"].format(k=split_idx), dataset=name)
    if name == "PAM":
        mf, stdf = getStats(Ptrain)
        out = [tensorize_normalize_extract_feature_pam_patch(P, y, mf, stdf, args)
               for P, y in ((Ptrain, ytrain), (Pval, yval), (Ptest, ytest))]
    else:
        T, F = Ptrain[0]["arr"].shape
        D = len(Ptrain[0]["extended_static"])
        tr = np.zeros((len(Ptrain), T, F))
        st = np.zeros((len(Ptrain), D))
        for i in range(len(Ptrain)):
            tr[i] = Ptrain[i]["arr"]
            st[i] = Ptrain[i]["extended_static"]
        mf, stdf = getStats(tr)
        ms, ss = getStats_static(st, dataset=name)
        out = [tensorize_normalize_extract_feature_patch(P, y, mf, stdf, ms, ss, args)
               for P, y in ((Ptrain, ytrain), (Pval, yval), (Ptest, ytest))]
    labels = (ytrain, yval, ytest)
    return out, labels


class _Split:
    """Holds one split's tensors and yields batch dicts."""

    def __init__(self, data, y, device, d_static):
        vals, mask, static, tp, y_t, _maxlen = data
        self.x = vals.to(device)
        self.mask = mask.to(device)
        self.tp = tp.to(device)
        self.static = static.to(device) if d_static else None
        self.y = torch.as_tensor(np.asarray(y).ravel(), dtype=torch.long, device=device)
        self.n = self.x.shape[0]

    def batch(self, idx):
        return {
            "observed_data": self.x[idx],
            "observed_tp": self.tp[idx],
            "observed_mask": self.mask[idx],
            "static": self.static[idx] if self.static is not None else None,
            "labels": self.y[idx],
        }


def _train_stream(split, batch_size, balanced, rng):
    """Infinite generator over training batches, preserving the Raindrop balancing for binary tasks."""
    y = split.y.detach().cpu().numpy()
    half = batch_size // 2
    if balanced:
        idx0 = np.where(y == 0)[0]
        idx1 = np.concatenate([np.where(y == 1)[0]] * 3)
        n_batches = min(len(idx0) // half, len(idx1) // half)
    else:
        all_idx = np.arange(len(y))
        n_batches = len(all_idx) // batch_size
    while True:
        if balanced:
            rng.shuffle(idx0); rng.shuffle(idx1)
            for n in range(n_batches):
                sel = np.concatenate([idx0[n * half:(n + 1) * half], idx1[n * half:(n + 1) * half]])
                yield split.batch(torch.as_tensor(sel, device=split.x.device))
        else:
            rng.shuffle(all_idx)
            for n in range(n_batches):
                sel = all_idx[n * batch_size:(n + 1) * batch_size]
                yield split.batch(torch.as_tensor(sel, device=split.x.device))


def _eval_stream(split, batch_size):
    """Infinite generator cycling over a split in order (the trainer asks for a fixed number of batches)."""
    while True:
        for s in range(0, split.n, batch_size):
            sel = torch.arange(s, min(s + batch_size, split.n), device=split.x.device)
            yield split.batch(sel)


def load(cfg, device, seed=0):
    d = dict(cfg.get("data", {}))
    name = d.get("name")
    c = CFG[name]
    args = argparse.Namespace(
        dataset=name,
        history=c["history"],
        patch_size=float(d.get("patch_size", c["patch_size"])),
        stride=float(d.get("stride", d.get("patch_size", c["patch_size"]))),
        ndim=c["variables_num"],
        n_class=c["n_class"],
        d_static=c["d_static"],
        batch_size=int(d.get("batch_size", 32)),
        device=device,
        irr_emb=True,
    )
    args.npatch = int(math.ceil((args.history - args.patch_size) / args.stride)) + 1

    import utils
    utils.setup_seed(seed)
    split_idx = int(d.get("split_idx", 1))
    (train, val, test), (ytr, yva, yte) = _tensorize(name, args, split_idx, device)
    tr = _Split(train, ytr, device, c["d_static"])
    va = _Split(val, yva, device, c["d_static"])
    te = _Split(test, yte, device, c["d_static"])

    rng = np.random.RandomState(seed)
    balanced = name != "PAM"
    half = args.batch_size // 2
    y = tr.y.detach().cpu().numpy()
    n_train_batches = (min(len(np.where(y == 0)[0]) // half, (3 * len(np.where(y == 1)[0])) // half)
                       if balanced else tr.n // args.batch_size)

    data_obj = {
        "train_dataloader": _train_stream(tr, args.batch_size, balanced, rng),
        "val_dataloader": _eval_stream(va, args.batch_size),
        "test_dataloader": _eval_stream(te, args.batch_size),
        "n_train_batches": int(n_train_batches),
        "n_val_batches": int(math.ceil(va.n / args.batch_size)),
        "n_test_batches": int(math.ceil(te.n / args.batch_size)),
        "input_dim": args.ndim,
    }
    info = dict(
        name=name, task="classification", input_dim=args.ndim, n_class=c["n_class"], d_static=c["d_static"],
        history=c["history"], pred_window=0, patch_size=args.patch_size, stride=args.stride,
        npatch=args.npatch, maxlen=0, batch_size=args.batch_size, split_idx=split_idx,
        n_train=tr.n, n_val=va.n, n_test=te.n,
        n_train_batches=data_obj["n_train_batches"], n_val_batches=data_obj["n_val_batches"],
        n_test_batches=data_obj["n_test_batches"],
    )
    return data_obj, info


def next_batch(loader):
    return next(loader)
