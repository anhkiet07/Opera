"""LLM-elicited physiological prior graph for the 41 PhysioNet-2012 variables (Phase 3, real data).

PROVENANCE: this adjacency was written by an LLM (Claude) from textbook physiology, not by a clinician and
not estimated from data. Treat it as a *noisy external prior*: the Phase 3 controls (shuffled / random /
noisy graphs) measure how much a wrong or partially wrong prior costs.

Semantics: EDGES lists (cause, effect). A[j, i] = 1 for j -> i. The attention bias built from it lets the
token of `effect` attend preferentially to its `cause`s. Bidirectional pairs (measurement of the same
quantity, or tight coupling) are listed in both directions. Sparse on purpose (~55 of 1640 possible edges).
"""
import numpy as np

from data.physionet import PhysioNet

VARS = list(PhysioNet.params)
IDX = {v: i for i, v in enumerate(VARS)}

EDGES = [
    # ---- ventilation / gas exchange ----
    ('MechVent', 'FiO2'), ('MechVent', 'PaCO2'), ('MechVent', 'RespRate'), ('MechVent', 'PaO2'),
    ('FiO2', 'PaO2'), ('FiO2', 'SaO2'), ('PaO2', 'SaO2'),
    ('RespRate', 'PaCO2'),
    # ---- acid-base ----
    ('PaCO2', 'pH'), ('HCO3', 'pH'), ('Lactate', 'pH'), ('Lactate', 'HCO3'),
    ('pH', 'K'),                                   # acidosis shifts potassium out of cells
    # ---- haemodynamics (MAP is derived from systolic / diastolic; invasive and cuff readings measure the same thing) ----
    ('SysABP', 'MAP'), ('DiasABP', 'MAP'), ('NISysABP', 'NIMAP'), ('NIDiasABP', 'NIMAP'),
    ('SysABP', 'NISysABP'), ('NISysABP', 'SysABP'), ('DiasABP', 'NIDiasABP'), ('NIDiasABP', 'DiasABP'),
    ('MAP', 'NIMAP'), ('NIMAP', 'MAP'),
    ('MAP', 'HR'),                                 # baroreflex: hypotension -> tachycardia
    ('MAP', 'Urine'), ('MAP', 'Lactate'),          # perfusion -> renal output, tissue hypoxia
    ('Temp', 'HR'), ('Temp', 'RespRate'),          # fever -> tachycardia / tachypnoea
    ('SaO2', 'HR'),                                # hypoxaemia -> tachycardia
    # ---- renal ----
    ('Urine', 'Creatinine'), ('Urine', 'BUN'),
    ('Creatinine', 'BUN'), ('BUN', 'Creatinine'),
    ('Creatinine', 'K'), ('Creatinine', 'Mg'),     # failing kidneys retain K, Mg
    ('Age', 'Creatinine'), ('Gender', 'Creatinine'), ('Weight', 'Creatinine'),   # muscle mass / GFR
    ('Na', 'Urine'),                               # osmotic diuresis / fluid state (weak)
    # ---- liver / cholestasis ----
    ('ALT', 'AST'), ('AST', 'ALT'), ('ALT', 'Bilirubin'), ('AST', 'Bilirubin'), ('ALP', 'Bilirubin'),
    ('Bilirubin', 'Albumin'),                      # chronic hepatic dysfunction -> low albumin (weak)
    # ---- infection / inflammation ----
    ('WBC', 'Temp'), ('Temp', 'WBC'), ('WBC', 'Lactate'),
    ('WBC', 'Platelets'),                          # sepsis -> consumptive thrombocytopenia (weak)
    # ---- cardiac injury ----
    ('TroponinI', 'TroponinT'), ('TroponinT', 'TroponinI'),
    ('TroponinI', 'HR'), ('MAP', 'TroponinI'),     # ischaemia <-> haemodynamics (weak)
    # ---- neuro / sedation ----
    ('GCS', 'MechVent'), ('MechVent', 'GCS'),      # low GCS -> intubation; sedation on the ventilator -> low GCS
    # ---- metabolic ----
    ('Glucose', 'Na'),                             # hyperglycaemia -> pseudohyponatraemia / osmotic shifts
    ('Gender', 'HCT'), ('Age', 'GCS'),
]


def adjacency():
    A = np.zeros((len(VARS), len(VARS)), dtype=np.float32)
    for c, e in EDGES:
        A[IDX[c], IDX[e]] = 1.0
    return A


def describe():
    A = adjacency()
    n_in = A.sum(0)
    print(f'{len(EDGES)} edges over {len(VARS)} variables; '
          f'{int((n_in > 0).sum())} variables have >= 1 cause, isolated: '
          f'{[v for v in VARS if A[IDX[v]].sum() == 0 and A[:, IDX[v]].sum() == 0]}')


if __name__ == '__main__':
    describe()
