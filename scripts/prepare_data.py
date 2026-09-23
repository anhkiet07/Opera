"""Build the on-disk cache for a dataset on CPU only, so it never competes with a running GPU job.

    python scripts/prepare_data.py --dataset activity
    python scripts/prepare_data.py --dataset ushcn

The Activity loader normally does its one-off parsing of the raw sensor file on whatever device the trainer
uses; forcing CPU here keeps the 8 GB GPU free for training. After this, training just loads the cache.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ["CUDA_VISIBLE_DEVICES"] = ""      # hard guarantee: no GPU allocation from this process


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["activity", "ushcn", "physionet"])
    a = ap.parse_args()

    import torch
    from data import DATA_ROOT
    dev = torch.device("cpu")
    t0 = time.time()

    if a.dataset == "activity":
        from data.activity import PersonActivity
        ds = PersonActivity(os.path.join(DATA_ROOT, "activity"), download=True, device=dev)
    elif a.dataset == "ushcn":
        from data.ushcn import USHCN
        ds = USHCN(os.path.join(DATA_ROOT, "ushcn"), device=dev)
    else:
        from data.physionet import PhysioNet
        ds = PhysioNet(os.path.join(DATA_ROOT, "physionet"), quantization=0.0, download=True, device=dev)

    rec = ds[0]
    print(f"{a.dataset}: {len(ds)} records, first record tt={tuple(rec[1].shape)} vals={tuple(rec[2].shape)}, "
          f"prepared in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
