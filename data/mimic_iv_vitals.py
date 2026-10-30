"""Download the MIMIC-IV v3.1 tables needed for a vitals-only ICU dataset (credentialed).

    set PHYSIONET_USER=<username>
    set PHYSIONET_PASS=<password>
    python data/mimic_iv_vitals.py

Requires a PhysioNet account with credentialed access and the signed MIMIC-IV data use agreement:
https://physionet.org/content/mimiciv/3.1/

Separate from data/mimic.py, which builds the older MIMIC forecasting benchmark; this file only fetches
raw tables.

Why this dataset: vital signs are charted on different schedules across eras (anchor_year_group in
patients.csv.gz, e.g. 2008-2010 vs 2017-2019) and ICU types (first_careunit in icustays.csv.gz), which
gives natural observation-process splits. Note that patients also differ between eras/units, so part of
any shift is a shift in X, not only in the observation process.

Files written under DATA_ROOT/mimic_iv/raw/ keeping PhysioNet's module layout:
    hosp/patients.csv.gz        anchor_year_group, age, sex
    hosp/admissions.csv.gz      admission times, outcome
    icu/icustays.csv.gz         ICU stays, first/last care unit
    icu/d_items.csv.gz          item dictionary (to select vital-sign itemids)
    icu/chartevents.csv.gz      all charted events, incl. vitals (~3.3 GB compressed)
Labs (hosp/labevents) are deliberately not downloaded: they are ordered when something looks abnormal,
so their observation process depends on the target.

Downloads are resumable (HTTP Range): rerun to finish a partial file.
"""
import argparse
import os
import sys

import requests

try:  # verify TLS against the OS trust store (needed behind TLS-inspecting proxies on Windows)
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import DATA_ROOT  # noqa: E402

BASE_URL = "https://physionet.org/files/mimiciv/{version}/"
FILES = [
    "hosp/patients.csv.gz",
    "hosp/admissions.csv.gz",
    "icu/icustays.csv.gz",
    "icu/d_items.csv.gz",
    "icu/chartevents.csv.gz",
]


def _fetch(session, url, out_path):
    have = os.path.getsize(out_path + ".part") if os.path.exists(out_path + ".part") else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with session.get(url, stream=True, timeout=300, headers=headers) as r:
        if r.status_code in (401, 403):
            sys.exit(f"{r.status_code} for {url}: check PHYSIONET_USER/PHYSIONET_PASS and that the "
                     "account has signed the MIMIC-IV data use agreement.")
        r.raise_for_status()
        mode = "ab" if r.status_code == 206 else "wb"
        with open(out_path + ".part", mode) as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(out_path + ".part", out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=os.path.join(DATA_ROOT, "mimic_iv"))
    p.add_argument("--version", default="3.1")
    args = p.parse_args()

    user, password = os.environ.get("PHYSIONET_USER"), os.environ.get("PHYSIONET_PASS")
    if not user or not password:
        sys.exit("Set PHYSIONET_USER and PHYSIONET_PASS (PhysioNet credentialed account).")

    raw = os.path.join(args.root, "raw")
    session = requests.Session()
    session.auth = (user, password)

    for name in FILES:
        out = os.path.join(raw, *name.split("/"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out):
            print(f"{name}: present")
            continue
        print(f"{name}: downloading")
        _fetch(session, BASE_URL.format(version=args.version) + name, out)
        print(f"{name}: {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
