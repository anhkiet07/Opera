"""Download NOAA Integrated Surface Database (ISD, "global-hourly") station reports (open access).

    python data/noaa_isd.py                                  # 500 random US stations, year 2023
    python data/noaa_isd.py --years 2022 2023 --max-stations 0   # every active US station

Source: NOAA NCEI ISD, read from the NOAA Open Data mirror on S3 (noaa-global-hourly-pds), which is
where www.ncei.noaa.gov/data/global-hourly/access/ redirects.

Why this dataset: the same weather process is reported on very different schedules. Automated
(ASOS) stations send routine METARs (REPORT_TYPE FM-15) plus event-driven specials (FM-16), staffed /
synoptic stations report every 1-3 h or 6 h (FM-12), so the observation process differs by station
type while short-horizon temperature is strongly determined by recent history.

Layout written under DATA_ROOT/noaa_isd/raw/:
    isd-history.csv                 NOAA station metadata (USAF, WBAN, name, state, ICAO, lat/lon, period)
    stations_selected.csv           the stations requested by this run
    <year>/<station>.csv.gz         one file per station-year, columns in KEEP_COLUMNS

Values are kept in ISD's raw encoding (e.g. TMP "+0117,1" = 11.7 degC with quality flag 1; 9999 = missing)
so no decoding decision is made silently at download time.
The download is resumable: files that already exist are skipped. Stations with no report in a year
return 404 and are listed in missing_<year>.txt.
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

HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
DATA_URL = "https://noaa-global-hourly-pds.s3.amazonaws.com/{year}/{station}.csv"

KEEP_COLUMNS = [
    "STATION", "DATE", "SOURCE", "REPORT_TYPE", "CALL_SIGN", "QUALITY_CONTROL",
    "TMP",   # air temperature
    "DEW",   # dew point
    "SLP",   # sea-level pressure
    "MA1",   # altimeter / station pressure
    "WND",   # wind direction, speed
    "VIS",   # visibility
    "CIG",   # ceiling height
    "AA1",   # liquid precipitation
]


# Forecasting variables decoded from the raw ISD strings: (name, column, field index, missing code, scale).
VARIABLES = [
    ("TMP", "TMP", 0, "+9999", 10.0),     # air temperature, degC
    ("DEW", "DEW", 0, "+9999", 10.0),     # dew point, degC
    ("ALT", "MA1", 0, "99999", 10.0),     # altimeter setting, hPa
    ("WSPD", "WND", 3, "9999", 10.0),     # wind speed, m/s
    ("VIS", "VIS", 0, "999999", 1000.0),  # visibility, km
]
# ISD quality codes marking a value as suspect or erroneous; those values are treated as unobserved.
BAD_QC = {"2", "3", "6", "7"}
BIN_MINUTES = 10   # reports are timestamped to the minute; averaged into 10-min bins, keeping hourly / 3-hourly
                   # schedules intact while bounding the number of distinct timestamps per window


def _decode(series, field, missing, scale):
    import numpy as np
    parts = series.fillna("").str.split(",", expand=True)
    if parts.shape[1] <= field:
        return np.full(len(series), np.nan)
    raw = parts[field]
    val = pd.to_numeric(raw.where(raw != missing), errors="coerce") / scale
    qc_col = field + 1 if parts.shape[1] > field + 1 else None
    if qc_col is not None:
        val = val.where(~parts[qc_col].isin(BAD_QC))
    return val.to_numpy(dtype=float)


def _station_record(path, year):
    import numpy as np
    import torch
    df = pd.read_csv(path, dtype=str)
    # SOD / SOM are daily / monthly summaries, not observations at their timestamp.
    df = df[~df.REPORT_TYPE.fillna("").str.strip().str.startswith("SO")]
    if df.empty:
        return None
    t = (pd.to_datetime(df.DATE) - pd.Timestamp(f"{year}-01-01")).dt.total_seconds() / 60.0
    out = pd.DataFrame({"bin": (t // BIN_MINUTES).astype(int).to_numpy()})
    for name, col, field, missing, scale in VARIABLES:
        out[name] = _decode(df[col], field, missing, scale) if col in df else np.nan
    g = out.groupby("bin").mean()[[v[0] for v in VARIABLES]]
    g = g[g.notna().any(axis=1)]
    if g.empty:
        return None
    vals = g.to_numpy(dtype=np.float32)
    mask = ~np.isnan(vals)
    tt = g.index.to_numpy(dtype=np.float32) * BIN_MINUTES / 60.0     # hours since 1 Jan
    return (torch.from_numpy(tt), torch.from_numpy(np.nan_to_num(vals)), torch.from_numpy(mask.astype(np.float32)))


def load_records(root=None, year=2023):
    """Full-year forecasting records, one per station: list of (station, tt (T,), vals (T, 5), mask (T, 5)).

    Variables: TMP, DEW, ALT, WSPD, VIS. Time in hours since 1 Jan of `year`, on 10-min bins.
    Built once from raw/<year>/ and cached in processed/.
    """
    import torch
    root = root or os.path.join(DATA_ROOT, "noaa_isd")
    cache = os.path.join(root, "processed", f"records_{year}.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    year_dir = os.path.join(root, "raw", str(year))
    files = sorted(f for f in os.listdir(year_dir) if f.endswith(".csv.gz"))
    records = []
    for i, f in enumerate(files, 1):
        rec = _station_record(os.path.join(year_dir, f), year)
        if rec is not None:
            records.append((f.split(".")[0], *rec))
        if i % 100 == 0:
            print(f"noaa_isd: {i}/{len(files)} stations processed", flush=True)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save(records, cache)
    return records


def _download(session, year, station, out_path):
    r = session.get(DATA_URL.format(year=year, station=station), timeout=300)
    if r.status_code in (403, 404):
        return station, None
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), dtype=str, low_memory=False)
    df = df[[c for c in KEEP_COLUMNS if c in df.columns]]
    tmp = out_path + ".part"
    df.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, out_path)
    return station, len(df)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=os.path.join(DATA_ROOT, "noaa_isd"))
    p.add_argument("--years", nargs="+", type=int, default=[2023])
    p.add_argument("--country", default="US")
    p.add_argument("--max-stations", type=int, default=500, help="random sample size; 0 = all eligible stations")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    raw = os.path.join(args.root, "raw")
    os.makedirs(raw, exist_ok=True)
    session = requests.Session()

    hist_path = os.path.join(raw, "isd-history.csv")
    if not os.path.exists(hist_path):
        r = session.get(HISTORY_URL, timeout=300)
        r.raise_for_status()
        with open(hist_path, "w", encoding="utf-8", newline="") as f:
            f.write(r.text)
        print(f"wrote {hist_path}")

    # Stations whose recorded period covers every requested year, with a known state (excludes buoys/ships).
    hist = pd.read_csv(hist_path, dtype=str)
    first, last = min(args.years), max(args.years)
    st = hist[(hist.CTRY == args.country) & hist.STATE.notna()
              & (hist.BEGIN <= f"{first}0101") & (hist.END >= f"{last}1231")].copy()
    st["station"] = st.USAF + st.WBAN
    st = st.drop_duplicates("station")
    if args.max_stations and len(st) > args.max_stations:
        st = st.sample(args.max_stations, random_state=args.seed)
    st = st.sort_values("station")
    st.to_csv(os.path.join(raw, "stations_selected.csv"), index=False)
    print(f"{len(st)} stations selected")

    for year in args.years:
        year_dir = os.path.join(raw, str(year))
        os.makedirs(year_dir, exist_ok=True)
        todo = [s for s in st.station if not os.path.exists(os.path.join(year_dir, f"{s}.csv.gz"))]
        print(f"[{year}] {len(st) - len(todo)} present, {len(todo)} to fetch")
        missing, failed = [], []
        with ThreadPoolExecutor(args.workers) as pool:
            futs = {pool.submit(_download, session, year, s, os.path.join(year_dir, f"{s}.csv.gz")): s for s in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    if fut.result()[1] is None:
                        missing.append(futs[fut])
                except Exception as e:  # keep going; rerun the script to retry failures
                    failed.append(futs[fut])
                    print(f"{futs[fut]} failed: {e}")
                if i % 50 == 0 or i == len(todo):
                    print(f"[{year}] {i}/{len(todo)} stations done")
        if missing:
            with open(os.path.join(raw, f"missing_{year}.txt"), "a") as f:
                f.write("\n".join(missing) + "\n")
        print(f"[{year}] done; {len(missing)} not in archive, {len(failed)} failed")


if __name__ == "__main__":
    main()
