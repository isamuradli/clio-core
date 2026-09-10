#!/usr/bin/env bash
# Every figure in this directory, rebuilt from this directory. No run stores,
# no scratch, no dumps -- the gzipped selection logs and their meta.json are
# the whole input.
#
#   ./regenerate.sh [OUTDIR]        (default: ./regenerated)
#
# PYTHON must have matplotlib; the system python3 here does not.
set -eu
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:-$HERE/regenerated}
VIZ=$HERE/../plot/viz_learning.py
PY=${PYTHON:-python3}
"$PY" -c "import matplotlib" 2>/dev/null || {
  echo "matplotlib missing from $PY -- set PYTHON=<interpreter that has it>" >&2
  exit 1; }
mkdir -p "$OUT"

# --- the cross-workload pair, all sixteen runs at once ---------------------
ARGS=()
for wl in nyx vpic lammps warpx; do
  for pair in learn:learn learn-ratio:learn-ratio \
              dynamic:learn-control dynamic-ratio:learn-ratio-control; do
    ARGS+=(--run "$wl:${pair##*:}:$HERE/$wl/${pair%%:*}.selection.csv.gz")
  done
done
"$PY" "$VIZ" trend --out "$OUT" "${ARGS[@]}"

# --- per-chunk, one figure per learning run, then the single-field cuts -----
for wl in nyx vpic lammps warpx; do
  for cfg in learn learn-ratio; do
    "$PY" "$VIZ" perchunk --out "$OUT/$wl" \
        --run "${wl}_${cfg}:$HERE/$wl/${cfg}.selection.csv.gz"
  done
done
# The variables the placed figures single out: one representative field per
# workload, plus nyx xmom, whose curve differs from nyx density.
for cfg in learn learn-ratio; do
  for pick in nyx:density vpic:ex lammps:velocity warpx:rho nyx:xmom; do
    wl=${pick%%:*} fld=${pick##*:}
    [ "$fld" = xmom ] && [ "$cfg" = learn-ratio ] && continue
    "$PY" "$VIZ" perchunk --out "$OUT/$wl" \
        --run "${wl}_${cfg}:$HERE/$wl/${cfg}.selection.csv.gz" --field "$fld"
  done
done
echo "regenerated into $OUT"
