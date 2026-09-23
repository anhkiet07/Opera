"""PhysioNet-2012 forecasting for OPERA (OPERA section 24).

The preprocessing is the existing, unmodified pipeline (`data/parse.py` -> `data/physionet.py`), i.e. the
t-PatchGNN protocol:

    history length     : 24 h (configurable)
    forecast horizon   : 48 h - history = 24 h
    variables          : 41 clinical variables
    sampling           : natively irregular; timestamps in hours, min-max normalised to [0, 1]
    normalisation      : per-variable min-max computed on the train+val ("seen") split
    split              : 60 / 20 / 20 by record, sklearn seed 42 (train_test_split 0.8 then 0.75)
    batching           : observations packed into (B, npatch, max_patch_len, N) with an explicit mask

Nothing here re-implements the preprocessing; this module only exposes it behind a stable interface.
"""
import argparse
import math

import torch


def load(cfg, device, seed=0):
    """Returns (data_obj, info). `data_obj` holds the three infinite dataloaders and batch counts."""
    d = cfg.get("data", {})
    args = argparse.Namespace(
        dataset="physionet",
        history=int(d.get("history", 24)),
        patch_size=float(d.get("patch_size", 6)),
        stride=float(d.get("stride", 6)),
        batch_size=int(d.get("batch_size", 32)),
        n=int(d.get("n_records", int(1e8))),
        quantization=float(d.get("quantization", 0.0)),
        pred_window=1,
        irr_emb=True,
        model=cfg.get("model", {}).get("backbone", "itransformer"),
        device=device,
        seed=seed,
    )
    args.npatch = int(math.ceil((args.history - args.patch_size) / args.stride)) + 1

    import utils
    from data.parse import parse_datasets
    utils.setup_seed(seed)
    data_obj = parse_datasets(args, patch_ts=True, max_ts=False)

    info = dict(
        name="physionet",
        input_dim=data_obj["input_dim"],
        history=args.history,
        pred_window=args.pred_window,          # parse_datasets sets this to 48 - history
        patch_size=args.patch_size,
        stride=args.stride,
        npatch=args.npatch,
        maxlen=getattr(args, "maxlen", 0),
        batch_size=args.batch_size,
        n_records=args.n,
        n_train_batches=data_obj["n_train_batches"],
        n_val_batches=data_obj["n_val_batches"],
        n_test_batches=data_obj["n_test_batches"],
    )
    info["pred_window"] = args.pred_window
    return data_obj, info


def next_batch(loader):
    import utils
    return utils.get_next_batch(loader)
