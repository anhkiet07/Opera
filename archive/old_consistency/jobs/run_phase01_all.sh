#!/bin/bash
# Master queue: Phase 0 + Phase 1 forecasting (PhysioNet), then classification (P12).
cd "$(dirname "$0")/.." || exit 1
EPOCH=${EPOCH:-30} PATIENCE=${PATIENCE:-8} SEEDS=${SEEDS:-"1 2 3"} bash jobs/phase01_physionet.sh
EPOCH=${EPOCH:-30} PATIENCE=${PATIENCE:-8} N_FOLDS=${N_FOLDS:-1} bash jobs/phase01_p12.sh
.venv/Scripts/python summarize_results.py
echo "MASTER DONE"
