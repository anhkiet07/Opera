#!/bin/bash
# Usage: sbatch -J <name> scripts/slurm/bench.sh <setting> <stage> [<stage> ...]
#SBATCH -p gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH -o logs/%x_%j.out
set -u
cd ~/opera/Opera
export QUITE_DATA_ROOT=/scratch/kiet/opera_data
export CC=/scratch/kiet/envs/build-cc/bin/x86_64-conda-linux-gnu-gcc
export CPATH=/scratch/kiet/envs/build-cc/include/python3.14
PY=~/quite_test/.venv/bin/python
SETTING=$1; shift
for STAGE in "$@"; do
  $PY scripts/run_benchmark.py --stage "$STAGE" --settings "$SETTING" --no_wandb
done
