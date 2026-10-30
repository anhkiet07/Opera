"""Download VitalDB intra-operative vital signs (open access, no credential).

    python data/vitaldb.py                      # all cases, default vital tracks
    python data/vitaldb.py --max-cases 200      # quick subset

Source: https://vitaldb.net (Lee et al., Sci Data 2022), served by the public API at api.vitaldb.net.

Why this dataset: the monitor records numeric vitals densely (~every 2 s), so the dense signal is a
genuine "rich view" and any recording policy (e.g. hand charting every 5 / 15 min, NIBP cuff cycles,
charting only when abnormal) can be applied on top of it as an observation process, while the target
stays dense.

Layout written under DATA_ROOT/vitaldb/raw/:
    cases.csv               clinical info, one row per case (caseid, age, sex, department, optype, ...)
    trks.csv                index of every track in VitalDB (caseid, tname, tid)
    tracks/<caseid>.csv.gz  long format: time (s from case start), track, value -- only the selected tracks

The download is resumable: cases whose file already exists are skipped.
"""
import argparse
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

try:  # verify TLS against the OS trust store (needed behind TLS-inspecting proxies on Windows)
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import DATA_ROOT  # noqa: E402

API = "https://api.vitaldb.net"

# Numeric vitals from the Solar8000 patient monitor. NIBP is cuff-cycled and therefore naturally sparse;
# ART is the invasive (continuous) counterpart, present in ~60% of cases.
DEFAULT_TRACKS = [
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2",
    "Solar8000/NIBP_SBP",
    "Solar8000/NIBP_DBP",
    "Solar8000/NIBP_MBP",
    "Solar8000/ART_SBP",
    "Solar8000/ART_DBP",
    "Solar8000/ART_MBP",
    "Solar8000/ETCO2",
    "Solar8000/RR_CO2",
    "Solar8000/BT",
]


# Physiologically plausible ranges; values outside are monitor artefacts (e.g. 0 when a sensor is off) and are
# treated as unobserved. Min-max normalisation is fit on the data, so leaving artefacts in would squash the scale.
PLAUSIBLE = {
    "Solar8000/HR": (20, 250),
    "Solar8000/PLETH_SPO2": (50, 100),
    "Solar8000/NIBP_SBP": (30, 300),
    "Solar8000/NIBP_DBP": (10, 200),
    "Solar8000/NIBP_MBP": (20, 250),
    "Solar8000/ART_SBP": (30, 300),
    "Solar8000/ART_DBP": (10, 200),
    "Solar8000/ART_MBP": (20, 250),
    "Solar8000/ETCO2": (5, 100),
    "Solar8000/RR_CO2": (2, 60),
    "Solar8000/BT": (25, 45),
}
BIN_SECONDS = 60   # monitor samples every ~2 s; averaged into 1-min bins, the time unit of the records
# The monitor re-emits the last cuff reading every 2 s until the next inflation, so a NIBP track looks dense.
# Only the first sample of each new value is a measurement; keeping the repeats would erase the sparse,
# cuff-cycled observation process that makes NIBP interesting. (Two identical consecutive readings merge.)
CARRY_FORWARD = {"Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP", "Solar8000/NIBP_MBP"}


def _case_record(path, tracks):
    import numpy as np
    import torch
    df = pd.read_csv(path)
    df = df[df.track.isin(tracks)].sort_values(["track", "time"])
    held = df.track.isin(CARRY_FORWARD)
    new_value = df.value.ne(df.groupby("track").value.shift())
    df = df[~held | new_value]
    lo = df.track.map(lambda t: PLAUSIBLE[t][0])
    hi = df.track.map(lambda t: PLAUSIBLE[t][1])
    df = df[(df.value >= lo) & (df.value <= hi)]
    if df.empty:
        return None
    df = df.assign(minute=(df.time // BIN_SECONDS).astype(int))
    g = df.groupby(["minute", "track"]).value.mean().unstack()
    g = g.reindex(columns=tracks)
    vals = g.to_numpy(dtype=np.float32)
    mask = ~np.isnan(vals)
    tt = g.index.to_numpy(dtype=np.float32)
    return (torch.from_numpy(tt), torch.from_numpy(np.nan_to_num(vals)), torch.from_numpy(mask.astype(np.float32)))


def load_records(root=None, tracks=DEFAULT_TRACKS):
    """Full-length forecasting records, one per surgical case: list of (caseid, tt (T,), vals (T, D), mask (T, D)).

    Time is in minutes from case start (1-min bins). Built once from raw/tracks/ and cached in processed/.
    """
    import torch
    root = root or os.path.join(DATA_ROOT, "vitaldb")
    cache = os.path.join(root, "processed", "records.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    track_dir = os.path.join(root, "raw", "tracks")
    files = sorted(os.listdir(track_dir), key=lambda f: int(f.split(".")[0]))
    records = []
    for i, f in enumerate(files, 1):
        rec = _case_record(os.path.join(track_dir, f), list(tracks))
        if rec is not None:
            records.append((f.split(".")[0], *rec))
        if i % 500 == 0:
            print(f"vitaldb: {i}/{len(files)} cases processed", flush=True)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save(records, cache)
    return records


def _get(session, url, retries=4):
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=120)
            r.raise_for_status()
            return r.text
        except requests.RequestException:
            if attempt == retries - 1:
                raise


def _download_case(session, caseid, case_trks, out_path):
    frames = []
    for tname, tid in case_trks:
        df = pd.read_csv(io.StringIO(_get(session, f"{API}/{tid}")))
        df.columns = ["time", "value"]
        df = df.dropna(subset=["value"])
        df.insert(1, "track", tname)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["time", "track", "value"])
    tmp = out_path + ".part"
    out.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, out_path)
    return caseid, len(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=os.path.join(DATA_ROOT, "vitaldb"))
    p.add_argument("--tracks", nargs="+", default=DEFAULT_TRACKS)
    p.add_argument("--max-cases", type=int, default=None, help="first N cases (by caseid) only")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    raw = os.path.join(args.root, "raw")
    track_dir = os.path.join(raw, "tracks")
    os.makedirs(track_dir, exist_ok=True)
    session = requests.Session()

    for name in ("cases", "trks"):
        path = os.path.join(raw, f"{name}.csv")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(_get(session, f"{API}/{name}").lstrip("﻿"))
            print(f"wrote {path}")

    trks = pd.read_csv(os.path.join(raw, "trks.csv"))
    trks = trks[trks.tname.isin(args.tracks)]
    caseids = sorted(trks.caseid.unique())
    if args.max_cases is not None:
        caseids = caseids[: args.max_cases]
    by_case = {c: list(zip(g.tname, g.tid)) for c, g in trks[trks.caseid.isin(caseids)].groupby("caseid")}

    todo = [c for c in caseids if not os.path.exists(os.path.join(track_dir, f"{c}.csv.gz"))]
    print(f"{len(caseids)} cases with selected tracks, {len(caseids) - len(todo)} already present, {len(todo)} to fetch")

    failed = []
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(_download_case, session, c, by_case[c], os.path.join(track_dir, f"{c}.csv.gz")): c
                for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
            except Exception as e:  # keep going; rerun the script to retry failures
                failed.append(futs[fut])
                print(f"case {futs[fut]} failed: {e}")
            if i % 100 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} cases done")

    print(f"done; {len(failed)} failed" + (f" (rerun to retry): {failed[:20]}" if failed else ""))


if __name__ == "__main__":
    main()
