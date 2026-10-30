"""Download the full eICU Collaborative Research Database v2.0 vital-sign tables (credentialed).

    set PHYSIONET_USER=<username>
    set PHYSIONET_PASS=<password>
    python data/eicu_crd.py

Requires a PhysioNet account with credentialed access and the signed eICU-CRD data use agreement:
https://physionet.org/content/eicu-crd/2.0/

Why this dataset: >200 hospitals, some recording vitals automatically every few minutes from the
monitor and others charting them by hand roughly hourly, so the observation schedule really differs
between sites. Split train/test by hospital (hospitalid in patient.csv.gz).

Files written under DATA_ROOT/eicu_crd/raw/ (sizes are the compressed sizes on PhysioNet):
    patient.csv.gz          stays, hospitalid, unit type, outcome
    hospital.csv.gz         hospital region / size / teaching status
    vitalPeriodic.csv.gz    monitor vitals at 5-min resolution (~1.6 GB)
    vitalAperiodic.csv.gz   non-invasive BP and other aperiodic vitals
Add --nurse-charting to also fetch nurseCharting.csv.gz (hand-charted vitals, ~1.5 GB).
Labs are deliberately not downloaded: they are ordered when something looks abnormal, so their
observation process depends on the target.

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

BASE_URL = "https://physionet.org/files/eicu-crd/2.0/"
FILES = ["patient.csv.gz", "hospital.csv.gz", "vitalPeriodic.csv.gz", "vitalAperiodic.csv.gz"]


def _fetch(session, url, out_path):
    have = os.path.getsize(out_path + ".part") if os.path.exists(out_path + ".part") else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with session.get(url, stream=True, timeout=300, headers=headers) as r:
        if r.status_code in (401, 403):
            sys.exit(f"{r.status_code} for {url}: check PHYSIONET_USER/PHYSIONET_PASS and that the "
                     "account has signed the eICU-CRD data use agreement.")
        r.raise_for_status()
        mode = "ab" if r.status_code == 206 else "wb"
        with open(out_path + ".part", mode) as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(out_path + ".part", out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=os.path.join(DATA_ROOT, "eicu_crd"))
    p.add_argument("--nurse-charting", action="store_true")
    args = p.parse_args()

    user, password = os.environ.get("PHYSIONET_USER"), os.environ.get("PHYSIONET_PASS")
    if not user or not password:
        sys.exit("Set PHYSIONET_USER and PHYSIONET_PASS (PhysioNet credentialed account).")

    raw = os.path.join(args.root, "raw")
    os.makedirs(raw, exist_ok=True)
    session = requests.Session()
    session.auth = (user, password)

    files = FILES + (["nurseCharting.csv.gz"] if args.nurse_charting else [])
    for name in files:
        out = os.path.join(raw, name)
        if os.path.exists(out):
            print(f"{name}: present")
            continue
        print(f"{name}: downloading")
        _fetch(session, BASE_URL + name, out)
        print(f"{name}: {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
