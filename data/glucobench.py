"""Download the GlucoBench collection of public CGM datasets (open licences, no credential).

    python data/glucobench.py

Source: raw_data.zip in https://github.com/IrinaStatsLab/GlucoBench (Sergazinov et al., ICLR 2024), which
bundles five public continuous-glucose-monitoring datasets already pulled from their original sources:

    Colas      208 subjects   CC BY       PLOS ONE
    Weinstock  200 subjects   CC BY-SA    JAEB public dataset
    Hall        57 subjects   CC BY       PLOS Biology supplement
    Dubosson     9 subjects   CC BY-SA    Zenodo 10.5281/zenodo.1421615
    Broll        5 subjects   GPL-2       iglu R package

All five are CGM at ~5-minute resolution, so the observation process is close to regular: shifts such
as a 15-minute device or sparse finger-stick readings have to be imposed on top of the dense trace.
Cite the original dataset papers as well as GlucoBench when using them.

Layout written under DATA_ROOT/glucobench/raw/:
    raw_data.zip     the archive as downloaded
    <extracted>/     its contents
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

REPO = "IrinaStatsLab/GlucoBench"
URL = f"https://raw.githubusercontent.com/{REPO}/main/raw_data.zip"


SOURCES = ["colas", "dubosson", "hall", "iglu", "weinstock"]   # iglu.csv is the Broll dataset


def load_records(root=None, sources=SOURCES):
    """Forecasting records, one per subject: list of (f"{source}_{id}", tt (T,), vals (T, 1), mask (T, 1)).

    Univariate glucose (mg/dL); time in minutes from the subject's first reading, rounded to the minute.
    Subject ids are prefixed with the source dataset because ids repeat across sources.
    Built once from raw/raw_data/ and cached in processed/.
    """
    import numpy as np
    import torch
    root = root or os.path.join(DATA_ROOT, "glucobench")
    cache = os.path.join(root, "processed", "records.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    records = []
    for src in sources:
        df = pd.read_csv(os.path.join(root, "raw", "raw_data", f"{src}.csv"), usecols=["id", "time", "gl"])
        df = df.dropna(subset=["id", "time", "gl"])
        df["time"] = pd.to_datetime(df["time"])
        for sid, g in df.groupby("id", sort=True):
            g = g.sort_values("time")
            minutes = ((g["time"] - g["time"].iloc[0]).dt.total_seconds() / 60.0).round()
            s = g.assign(minute=minutes).groupby("minute").gl.mean()
            tt = torch.tensor(s.index.to_numpy(dtype=np.float32))
            vals = torch.tensor(s.to_numpy(dtype=np.float32)).unsqueeze(-1)
            records.append((f"{src}_{sid}", tt, vals, torch.ones_like(vals)))
        print(f"glucobench: {src} done ({len(records)} subjects so far)", flush=True)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save(records, cache)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=os.path.join(DATA_ROOT, "glucobench"))
    args = p.parse_args()

    raw = os.path.join(args.root, "raw")
    os.makedirs(raw, exist_ok=True)
    zip_path = os.path.join(raw, "raw_data.zip")

    if not os.path.exists(zip_path):
        r = requests.get(URL, timeout=300)
        r.raise_for_status()
        with open(zip_path + ".part", "wb") as f:
            f.write(r.content)
        os.replace(zip_path + ".part", zip_path)
    with open(zip_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    print(f"{zip_path}: {os.path.getsize(zip_path) / 1e6:.1f} MB, sha256 {sha}")

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(raw)
        names = z.namelist()
    print(f"extracted {len(names)} entries into {raw}")


if __name__ == "__main__":
    main()
