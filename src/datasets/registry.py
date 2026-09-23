"""Dataset registry for OPERA (OPERA section 24).

All three datasets go through the repository's existing, unmodified preprocessing (`data/parse.py`), so the
protocol is exactly the published one; this module only exposes it behind one interface and documents each
setup. Choose with `data.name` in the config.

| dataset   | records | vars | history -> horizon        | patch | notes                                    |
|-----------|---------|------|---------------------------|-------|------------------------------------------|
| physionet | 12,000  | 41   | 24 h -> 24 h              | 6 h   | ICU labs/vitals, ~88% missing            |
| activity  | 5,400   | 12   | 3000 ms -> 1000 ms        | 750   | 4 body sensors x 3 axes, ~75% missing    |
| ushcn     | 26,736  | 5    | 24 months -> 1 month      | 1.5   | climate stations, ~78% missing           |

The horizon is set by `data/parse.py` per dataset (physionet: 48 - history; activity: 4000 - history; ushcn:
the configured pred_window), which is why `pred_window` is read back from the namespace after loading.
"""
import argparse
import math

DEFAULTS = {
    "physionet": dict(history=24, patch_size=6, stride=6, batch_size=32, pred_window=1, quantization=0.0),
    "activity": dict(history=3000, patch_size=750, stride=750, batch_size=32, pred_window=1, quantization=0.0),
    "ushcn": dict(history=24, patch_size=1.5, stride=1.5, batch_size=128, pred_window=1, quantization=0.0),
}


def load(cfg, device, seed=0):
    """Returns (data_obj, info) for the dataset named in cfg['data']['name']."""
    d = dict(cfg.get("data", {}))
    name = d.get("name", "physionet")
    if name not in DEFAULTS:
        raise ValueError(f"unknown dataset '{name}'; known: {sorted(DEFAULTS)}")
    base = DEFAULTS[name]

    args = argparse.Namespace(
        dataset=name,
        history=type(base["history"])(d.get("history", base["history"])),
        patch_size=float(d.get("patch_size", base["patch_size"])),
        stride=float(d.get("stride", base["stride"])),
        batch_size=int(d.get("batch_size", base["batch_size"])),
        pred_window=int(d.get("pred_window", base["pred_window"])),
        quantization=float(d.get("quantization", base["quantization"])),
        n=int(d.get("n_records", int(1e8))),
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
        name=name,
        input_dim=data_obj["input_dim"],
        history=args.history,
        pred_window=args.pred_window,         # set by parse_datasets for physionet / activity
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
    return data_obj, info


def next_batch(loader):
    import utils
    return utils.get_next_batch(loader)
