#!/usr/bin/env bash
# Every run the two campaigns produced, in one table and one pair of figures.
# The control arms are passed as `<policy>-control` so viz_learning draws them
# dashed against the learning curve they are the baseline for.
set -eu
# RUNS is a campaign's run stores; PYTHON must have matplotlib. To replot the
# KEPT record instead of a live campaign, use ../regenerate.sh -- it reads the
# gzipped logs beside it and needs no run stores at all.
RUNS=${RUNS:?set RUNS to the runs/ directory of a campaign}
OUTDIR=${OUTDIR:-$PWD/viz}
PY=${PYTHON:-python3}
ARGS=()
for wl in nyx vpic lammps warpx; do
  for pair in "learn:learn" "learn-ratio:learn-ratio" \
              "dynamic:learn-control" "dynamic-ratio:learn-ratio-control"; do
    cfg=${pair%%:*}; pol=${pair##*:}
    if [ -f "$RUNS/${wl}_${cfg}/selection.csv" ]; then
      ARGS+=(--run "$wl:$pol:$RUNS/${wl}_${cfg}")
    fi
  done
done
exec "$PY" "$(dirname "${BASH_SOURCE[0]}")/../../plot/viz_learning.py" \
     trend --out "$OUTDIR" "${ARGS[@]}"
