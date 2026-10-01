"""Build the eICU-demo mortality dataset in the Raindrop record format.

    python scripts/prepare_eicu_demo.py

Task: in-hospital mortality from the first 60 hours of periodic vital signs.

Layout follows P19 exactly -- 60 hourly bins, `arr` of shape (60, F) holding values with 0 meaning
"not observed", `time` in hours, and a separate static vector -- so the existing classification loader,
normalisation and patching apply unchanged and eICU is comparable with the other classification settings.

Split is by hospital REGION, not by hospital: the demo has 2,520 stays spread over 186 hospitals (~13 each),
so holding out a single hospital would leave about one death in the test set. Holding out the largest region
maximises the test set while still making the test hospitals genuinely unseen.

POWER WARNING, recorded here because it governs how the result may be read: the held-out region has roughly
800 stays at an ~8% mortality rate, i.e. ~60 positive events. The AUROC standard error at that event count is
around +/-0.04, while the arm differences this benchmark resolves are 1-3 points. This setting can show a
large effect; it cannot adjudicate between opera and opera++. The full eICU-CRD (credentialed) is what that
comparison needs.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import DATA_ROOT  # noqa: E402

T_BINS = 60                  # hourly bins, matching P19's 60-step window
BIN_MIN = 60                 # minutes per bin
# Periodic vitals, ordered by coverage. The spread in coverage is deliberate: it is a real
# observation process, which is what this benchmark is about.
CHANNELS = ["heartrate", "sao2", "respiration", "systemicsystolic",
            "systemicdiastolic", "systemicmean", "temperature", "cvp"]
HELD_OUT_REGION = "Midwest"  # largest region -> largest test set
VAL_FRAC = 1.0 / 9.0
N_SPLITS = 5


def _age(v):
    if pd.isna(v):
        return 0.0
    s = str(v).strip()
    if s.startswith(">"):
        return 90.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def main():
    src = os.path.join(DATA_ROOT, "eicu_demo")
    out = os.path.join(DATA_ROOT, "eicu_demo", "processed_data")
    os.makedirs(out, exist_ok=True)

    pat = pd.read_csv(os.path.join(src, "patient.csv.gz"), low_memory=False)
    hosp = pd.read_csv(os.path.join(src, "hospital.csv.gz"))
    pat = pat.merge(hosp[["hospitalid", "region"]], on="hospitalid", how="left")

    pat = pat[pat.hospitaldischargestatus.notna()].copy()
    pat["label"] = (pat.hospitaldischargestatus == "Expired").astype(int)
    pat["age_n"] = pat.age.map(_age)
    pat["gender_n"] = (pat.gender == "Male").astype(float)
    pat["weight_n"] = pd.to_numeric(pat.admissionweight, errors="coerce").fillna(0.0)
    print(f"stays with a label: {len(pat)}  mortality {pat.label.mean() * 100:.2f}%")

    vit = pd.read_csv(os.path.join(src, "vitalPeriodic.csv.gz"), low_memory=False,
                      usecols=["patientunitstayid", "observationoffset"] + CHANNELS)
    vit = vit[(vit.observationoffset >= 0) & (vit.observationoffset < T_BINS * BIN_MIN)]
    vit["bin"] = (vit.observationoffset // BIN_MIN).astype(int)
    agg = vit.groupby(["patientunitstayid", "bin"])[CHANNELS].mean()
    print(f"binned vitals: {len(agg)} (stay, hour) cells over {agg.index.get_level_values(0).nunique()} stays")

    records, labels, regions = [], [], []
    for sid, row in pat.set_index("patientunitstayid").iterrows():
        arr = np.zeros((T_BINS, len(CHANNELS)), dtype=np.float64)
        if sid in agg.index.get_level_values(0):
            sub = agg.loc[sid]
            idx = sub.index.to_numpy()
            vals = np.nan_to_num(sub.to_numpy(dtype=np.float64), nan=0.0)
            arr[idx] = vals
        if not arr.any():
            continue                                  # no periodic vitals at all
        records.append({
            "id": f"eicu_{int(sid)}",
            "length": int((arr != 0).any(axis=1).sum()),
            "extended_static": (row.age_n, row.gender_n, row.weight_n),
            "arr": arr,
            "time": np.arange(T_BINS, dtype=np.float64).reshape(-1, 1),
        })
        labels.append(int(row.label))
        regions.append(row.region if isinstance(row.region, str) else "Unknown")

    labels = np.asarray(labels, dtype=float)
    regions = np.asarray(regions)
    print(f"usable records: {len(records)}  mortality {labels.mean() * 100:.2f}%")
    print(f"observed fraction of (hour x channel) cells: "
          f"{np.mean([(r['arr'] != 0).mean() for r in records]) * 100:.1f}%")

    np.save(os.path.join(out, "PTdict_list.npy"), np.array(records, dtype=object), allow_pickle=True)
    # arr_outcomes: loader reads the LAST column as the label
    np.save(os.path.join(out, "arr_outcomes.npy"), labels.reshape(-1, 1), allow_pickle=True)

    test_idx = np.where(regions == HELD_OUT_REGION)[0]
    pool = np.where(regions != HELD_OUT_REGION)[0]
    print(f"\nheld-out region {HELD_OUT_REGION}: {len(test_idx)} stays, "
          f"{int(labels[test_idx].sum())} deaths ({labels[test_idx].mean() * 100:.2f}%)")
    print(f"training pool (other regions): {len(pool)} stays, "
          f"{int(labels[pool].sum())} deaths ({labels[pool].mean() * 100:.2f}%)")

    sp_dir = os.path.join(DATA_ROOT, "eicu_demo", "splits")
    os.makedirs(sp_dir, exist_ok=True)
    for k in range(1, N_SPLITS + 1):
        rng = np.random.RandomState(2000 + k)
        perm = rng.permutation(pool)
        n_val = int(round(len(perm) * VAL_FRAC))
        idx_val, idx_train = perm[:n_val], perm[n_val:]
        path = os.path.join(sp_dir, f"eicu_region_split{k}.npy")
        np.save(path, np.array([idx_train, idx_val, test_idx], dtype=object), allow_pickle=True)
        print(f"  split{k}: train {len(idx_train)}  val {len(idx_val)}  test {len(test_idx)}")


if __name__ == "__main__":
    main()
