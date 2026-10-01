"""Write the Sepsis-2019 hospital-shift split files for the P19_hospB setting.

    python scripts/make_sepsis_hospital_split.py

Train and validate on hospital system A, test on hospital system B. This is a native covariate shift --
two different hospital systems, no synthetic corruption -- and it needs no new data: the Raindrop P19
release preserves each record's source path in its `id` field (``Data_physionet19/trainingA\\p000001.psv``),
so the hospital label is recoverable from the arrays already in the repo.

Deriving the split this way rather than re-preprocessing the raw .psv files is deliberate: P19_hospB then
uses byte-identical records, normalisation and patching to the existing P19 cells, so a difference between
them measures hospital shift and not a second preprocessing pipeline (CLAUDE.md section 24).

Test is always the whole of system B. Only the train/validation partition within system A varies with the
split index, matching the 8:1 train:val ratio of the Raindrop splits.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import DATA_ROOT  # noqa: E402

N_SPLITS = 5
VAL_FRAC = 1.0 / 9.0          # 8:1 train:val within system A


def hospital_of(record_id):
    s = str(record_id)
    if "trainingA" in s:
        return "A"
    if "trainingB" in s:
        return "B"
    return "?"


def main():
    base = os.path.join(DATA_ROOT, "P19")
    recs = np.load(os.path.join(base, "processed_data", "PT_dict_list_6.npy"), allow_pickle=True)
    y = np.load(os.path.join(base, "processed_data", "arr_outcomes_6.npy"), allow_pickle=True)[:, -1]
    hosp = np.array([hospital_of(r["id"]) for r in recs])
    if (hosp == "?").any():
        sys.exit(f"{(hosp == '?').sum()} records have no recoverable hospital label")

    idx_a = np.where(hosp == "A")[0]
    idx_b = np.where(hosp == "B")[0]
    print(f"system A: {len(idx_a)} records, sepsis {y[idx_a].astype(float).mean() * 100:.2f}%")
    print(f"system B: {len(idx_b)} records, sepsis {y[idx_b].astype(float).mean() * 100:.2f}%  (held out)")

    out_dir = os.path.join(base, "splits")
    os.makedirs(out_dir, exist_ok=True)
    for k in range(1, N_SPLITS + 1):
        rng = np.random.RandomState(1000 + k)
        perm = rng.permutation(idx_a)
        n_val = int(round(len(perm) * VAL_FRAC))
        idx_val, idx_train = perm[:n_val], perm[n_val:]
        path = os.path.join(out_dir, f"phy19_hospB_split{k}.npy")
        # same container shape as the Raindrop split files: (idx_train, idx_val, idx_test)
        np.save(path, np.array([idx_train, idx_val, idx_b], dtype=object), allow_pickle=True)
        print(f"  split{k}: train {len(idx_train)}  val {len(idx_val)}  test {len(idx_b)}  -> {path}")


if __name__ == "__main__":
    main()
