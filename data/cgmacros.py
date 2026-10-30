"""Download CGMacros v1.0.0 from PhysioNet (open access, no credential).

    python data/cgmacros.py

Source: https://physionet.org/content/cgmacros/1.0.0/ (Gutierrez-Osuna et al.). 45 participants (healthy,
pre-diabetes, type 2 diabetes) wore two CGMs at once -- Abbott FreeStyle Libre Pro (15-min readings)
and Dexcom G6 Pro (5-min readings) -- alongside logged meals with macronutrients, Fitbit activity and
heart rate. The two devices measure the same glucose process on different schedules, which is a
natural observation-process pair. No type 1 participants.

Layout written under DATA_ROOT/cgmacros/raw/:
    CGMacros_dateshifted365.zip   main archive (~660 MB), verified against SHA256SUMS.txt
    <extracted>/                  per-participant CSVs
    DataDictionary*.csv / .pdf, LICENSE.txt, SHA256SUMS.txt

The main download is resumable (HTTP Range): rerun to finish a partial file.
"""
import argparse
import hashlib
import os
import sys
import zipfile

import pandas as pd
import requests

try:  # verify TLS against the OS trust store (needed behind TLS-inspecting proxies on Windows)
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import DATA_ROOT  # noqa: E402

BASE_URL = "https://physionet.org/files/cgmacros/1.0.0/"
ARCHIVE = "CGMacros_dateshifted365.zip"
FILES = [
    "SHA256SUMS.txt",
    "LICENSE.txt",
    "DataDictionary.pdf",
    "DataDictionary_Bio.csv",
    "DataDictionary_CGMacros-00X.csv",
    "DataDictionary_Gut_Health_Test.csv",
    "DataDictionary_Microbes.csv",
    ARCHIVE,
]


# The published CSVs linearly interpolate both CGMs onto a 1-min grid. The device readings are the knots:
# Libre every 15 min, Dexcom every 5 min, counted from the start of each contiguous run. On the knots the
# value is an integer (mg/dL) in >99% of rows versus 15-45% elsewhere, which is how the periods were checked.
CGM_PERIOD = {"Dexcom GL": 5, "Libre GL": 15}
VARIABLES = ["Dexcom GL", "Libre GL", "HR", "METs"]
WEARABLE_BIN = 5   # Fitbit HR / METs are per minute; averaged into 5-min bins to bound the token count


def _knots(minutes, values, period):
    """Boolean mask of the rows holding a real device reading (not an interpolated one)."""
    import numpy as np
    ok = ~np.isnan(values)
    keep = np.zeros(len(values), dtype=bool)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return keep
    m = minutes[idx]
    new_run = np.r_[True, (np.diff(idx) > 1) | (np.diff(m) > 1)]
    run_start = m[new_run][np.cumsum(new_run) - 1]
    keep[idx] = np.isclose((m - run_start) % period, 0)
    return keep


def _participant_record(path):
    import numpy as np
    import torch
    df = pd.read_csv(path)
    t = pd.to_datetime(df["Timestamp"])
    minutes = ((t - t.iloc[0]).dt.total_seconds() / 60.0).round().to_numpy()
    cols = {}
    for c, period in CGM_PERIOD.items():
        v = df[c].to_numpy(dtype=float)
        cols[c] = pd.Series(np.where(_knots(minutes, v, period), v, np.nan), index=minutes)
    wear_bin = (minutes // WEARABLE_BIN) * WEARABLE_BIN
    for c in ("HR", "METs"):   # not every participant has both wearable columns; missing -> unobserved
        v = df[c].to_numpy(dtype=float) if c in df else np.full(len(df), np.nan)
        cols[c] = pd.Series(v, index=wear_bin).groupby(level=0).mean()
    g = pd.concat({c: s.groupby(level=0).mean() for c, s in cols.items()}, axis=1)[VARIABLES]
    g = g[g.notna().any(axis=1)].sort_index()
    vals = g.to_numpy(dtype=np.float32)
    mask = ~np.isnan(vals)
    return (torch.tensor(g.index.to_numpy(dtype=np.float32)), torch.from_numpy(np.nan_to_num(vals)),
            torch.from_numpy(mask.astype(np.float32)))


def load_records(root=None):
    """Forecasting records, one per participant: list of (id, tt (T,), vals (T, 4), mask (T, 4)).

    Variables: Dexcom GL (5-min device readings), Libre GL (15-min device readings), HR and METs (5-min means).
    Time in minutes from the participant's first row. Built once from raw/CGMacros/ and cached in processed/.
    """
    import glob
    import torch
    root = root or os.path.join(DATA_ROOT, "cgmacros")
    cache = os.path.join(root, "processed", "records.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    files = sorted(glob.glob(os.path.join(root, "raw", "CGMacros", "CGMacros-*", "CGMacros-*.csv")))
    records = [(os.path.basename(f).split(".")[0], *_participant_record(f)) for f in files]
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save(records, cache)
    return records


def _fetch(session, url, out_path):
    have = os.path.getsize(out_path + ".part") if os.path.exists(out_path + ".part") else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with session.get(url, stream=True, timeout=300, headers=headers) as r:
        r.raise_for_status()
        mode = "ab" if r.status_code == 206 else "wb"
        with open(out_path + ".part", mode) as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(out_path + ".part", out_path)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=os.path.join(DATA_ROOT, "cgmacros"))
    args = p.parse_args()

    raw = os.path.join(args.root, "raw")
    os.makedirs(raw, exist_ok=True)
    session = requests.Session()

    for name in FILES:
        out = os.path.join(raw, name)
        if not os.path.exists(out):
            print(f"{name}: downloading")
            _fetch(session, BASE_URL + name, out)
        print(f"{name}: {os.path.getsize(out) / 1e6:.1f} MB")

    expected = {}
    with open(os.path.join(raw, "SHA256SUMS.txt")) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                expected[parts[1].lstrip("*")] = parts[0]
    bad = [n for n in FILES if n in expected and _sha256(os.path.join(raw, n)) != expected[n]]
    if bad:
        sys.exit(f"checksum mismatch: {bad} -- delete these files and rerun")
    print("checksums OK")

    with zipfile.ZipFile(os.path.join(raw, ARCHIVE)) as z:
        z.extractall(raw)
        print(f"extracted {len(z.namelist())} entries into {raw}")


if __name__ == "__main__":
    main()
