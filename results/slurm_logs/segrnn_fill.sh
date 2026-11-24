#!/bin/bash
# SegRNN on the 8 forecasting settings it was missing; vanilla before opera (teacher), seeds 0-2.
cd /home/vmo/Opera
S=physionet,physionet_h36,activity,activity_h2000,ushcn_1mo,ushcn_12mo,cgmacros,noaa_isd
P=.venv/bin/python
$P scripts/run_benchmark.py --stage A --backbones segrnn --settings $S --no_csv --gpu_wait_mb 2500
$P scripts/run_benchmark.py --stage C --backbones segrnn --settings $S --no_csv --gpu_wait_mb 2500
$P scripts/run_benchmark.py --stage B --backbones segrnn --settings $S --seeds 1,2 --no_csv --gpu_wait_mb 2500
$P scripts/run_benchmark.py --stage D --backbones segrnn --settings $S --seeds 1,2 --gpu_wait_mb 2500
