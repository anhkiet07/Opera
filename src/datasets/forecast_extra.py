"""Forecasting settings on the four datasets added for observation-process shift.

| name       | records          | vars | time unit | history -> horizon | patch | window stride | native observation process                    |
|------------|------------------|------|-----------|--------------------|-------|---------------|-----------------------------------------------|
| vitaldb    | 6,388 surgeries  | 11   | minute    | 60 -> 15 min       | 15    | 60 min        | 1-min monitor bins; NIBP cuff-cycled, ART only in ~60% of cases |
| noaa_isd   | 498 US stations  | 5    | hour      | 24 -> 6 h          | 6     | 168 h         | 10-min bins; ASOS hourly + specials, CRN 5-min, SYNOP 3-6 h |
| glucobench | 479 subjects     | 1    | minute    | 120 -> 60 min      | 30    | 180 min       | CGM every 5 min (univariate)                  |
| cgmacros   | 45 participants  | 4    | minute    | 180 -> 60 min      | 45    | 60 min        | Dexcom 5 min, Libre 15 min, Fitbit HR/METs 5-min means |

Records come from `load_records()` in data/<name>.py (cached under DATA_ROOT/<name>/processed/). The protocol
otherwise follows the existing forecasting settings exactly:

    split          : 60 / 20 / 20 by record (surgery / station / subject), sklearn seed 42, made BEFORE windowing,
                     so no record contributes windows to two splits
    windows        : [st, st + history + horizon) every `window_stride`, times re-based to the window start;
                     a window is kept if it has >= min_history_obs observed timestamps in the history and at
                     least one observation in the horizon
    normalisation  : per-variable min-max on the train+val windows; time divided by history + horizon
    batching       : data/physionet.patch_variable_time_collate_fn, i.e. the same patching and explicit mask

GlucoBench is univariate, so variable_dropout / variable_group_dropout leave it unchanged (one variable is always
kept): its `variables` and the variable half of `composed` are not informative there.
"""
import argparse
import math

import torch
from sklearn import model_selection
from torch.utils.data import DataLoader

CFG = {
    "vitaldb":    dict(history=60, pred_window=15, patch_size=15, stride=15, window_stride=60,
                       batch_size=32, min_history_obs=8),
    "noaa_isd":   dict(history=24, pred_window=6, patch_size=6, stride=6, window_stride=168,
                       batch_size=32, min_history_obs=8),
    "glucobench": dict(history=120, pred_window=60, patch_size=30, stride=30, window_stride=180,
                       batch_size=32, min_history_obs=8),
    "cgmacros":   dict(history=180, pred_window=60, patch_size=45, stride=45, window_stride=60,
                       batch_size=32, min_history_obs=8),
}


def _records(name):
    if name == "vitaldb":
        from data.vitaldb import load_records
    elif name == "noaa_isd":
        from data.noaa_isd import load_records
    elif name == "glucobench":
        from data.glucobench import load_records
    elif name == "cgmacros":
        from data.cgmacros import load_records
    return load_records()


def make_windows(records, history, pred_window, window_stride, min_history_obs):
    """Cut full-length records into forecasting windows with times re-based to the window start."""
    out = []
    span = history + pred_window
    for rid, tt, vals, mask in records:
        if len(tt) == 0:
            continue
        t0, t_end = float(tt.min()), float(tt.max())
        st = t0
        while st + history < t_end:
            idx = torch.where((tt >= st) & (tt < st + span))[0]
            w_tt = tt[idx] - st
            w_mask = mask[idx]
            hist = w_tt < history
            if int(hist.sum()) >= min_history_obs and w_mask[~hist].sum() > 0:
                out.append((f"{rid}_{int(round((st - t0) / window_stride))}", w_tt, vals[idx], w_mask))
            st += window_stride
    return out


def load(cfg, device, seed=0):
    """Returns (data_obj, info) for a setting in CFG, in the format of src/datasets/registry.load."""
    import utils
    from data.physionet import get_data_min_max, patch_variable_time_collate_fn

    d = dict(cfg.get("data", {}))
    name = d["name"]
    base = CFG[name]
    args = argparse.Namespace(
        dataset=name,
        history=float(d.get("history", base["history"])),
        pred_window=float(d.get("pred_window", base["pred_window"])),
        patch_size=float(d.get("patch_size", base["patch_size"])),
        stride=float(d.get("stride", base["stride"])),
        batch_size=int(d.get("batch_size", base["batch_size"])),
        n=int(d.get("n_records", int(1e8))),
        irr_emb=True,
        model=cfg.get("model", {}).get("backbone", "itransformer"),
        device=device,
        seed=seed,
        maxlen=0,
    )
    args.npatch = int(math.ceil((args.history - args.patch_size) / args.stride)) + 1
    window_stride = float(d.get("window_stride", base["window_stride"]))
    min_history_obs = int(d.get("min_history_obs", base["min_history_obs"]))

    utils.setup_seed(seed)
    records = _records(name)[: args.n]
    seen, test = model_selection.train_test_split(records, train_size=0.8, random_state=42, shuffle=True)
    train, val = model_selection.train_test_split(seen, train_size=0.75, random_state=42, shuffle=False)
    print(f"{name} records:", len(records), len(train), len(val), len(test))

    win = lambda recs: make_windows(recs, args.history, args.pred_window, window_stride, min_history_obs)
    train_w, val_w, test_w = win(train), win(val), win(test)
    print(f"{name} windows:", len(train_w) + len(val_w) + len(test_w), len(train_w), len(val_w), len(test_w))

    data_min, data_max, _ = get_data_min_max(train_w + val_w, torch.device("cpu"))
    # a variable never observed in train+val would give an infinite scale; pin it to [0, 1]
    absent = torch.isinf(data_min) | torch.isinf(data_max)
    data_min = torch.where(absent, torch.zeros_like(data_min), data_min).to(device)
    data_max = torch.where(absent, torch.ones_like(data_max), data_max).to(device)
    time_max = torch.tensor(args.history + args.pred_window)

    def loader(windows, data_type, shuffle):
        return DataLoader(windows, batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=lambda batch: patch_variable_time_collate_fn(
                              [(r, t.to(device), v.to(device), m.to(device)) for r, t, v, m in batch],
                              args, device, data_type=data_type,
                              data_min=data_min, data_max=data_max, time_max=time_max))

    train_dl, val_dl, test_dl = loader(train_w, "train", True), loader(val_w, "val", False), loader(test_w, "test", False)
    input_dim = train_w[0][2].size(-1)
    data_obj = {
        "train_dataloader": utils.inf_generator(train_dl),
        "val_dataloader": utils.inf_generator(val_dl),
        "test_dataloader": utils.inf_generator(test_dl),
        "input_dim": input_dim,
        "n_train_batches": len(train_dl),
        "n_val_batches": len(val_dl),
        "n_test_batches": len(test_dl),
        "data_max": data_max,
        "data_min": data_min,
        "time_max": time_max,
        "x_mean": None,
    }
    info = dict(
        name=name,
        input_dim=input_dim,
        history=args.history,
        pred_window=args.pred_window,
        patch_size=args.patch_size,
        stride=args.stride,
        npatch=args.npatch,
        maxlen=args.maxlen,
        batch_size=args.batch_size,
        n_records=args.n,
        n_train_batches=data_obj["n_train_batches"],
        n_val_batches=data_obj["n_val_batches"],
        n_test_batches=data_obj["n_test_batches"],
    )
    return data_obj, info


def next_batch(loader):
    import utils
    return utils.get_next_batch(loader)
