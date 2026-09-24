#!/usr/bin/env bash
# Regenerate the benchmark CSVs whenever a new run record lands. CPU and file IO only, so it never contends
# with the training job for the GPU. Exits once the sweep log reports stage C finished.
PY=".venv/Scripts/python.exe"
[ -x "$PY" ] || PY=".venv/bin/python"
LOG="results/bench_run.log"
prev=-1
while true; do
  n=$(ls -1 results/raw/*/results.json 2>/dev/null | grep -vc '/_' || true)
  if [ "$n" != "$prev" ]; then
    "$PY" scripts/benchmark_csv.py > results/csv_refresh.log 2>&1
    echo "$(date '+%H:%M:%S')  runs=$n  csv refreshed"
    prev=$n
  fi
  grep -q "=== STAGE C DONE" "$LOG" 2>/dev/null && { echo "$(date '+%H:%M:%S')  sweep finished, final refresh"; "$PY" scripts/benchmark_csv.py > results/csv_refresh.log 2>&1; exit 0; }
  sleep 60
done
