#!/usr/bin/env bash
# Phase 3: LAMMPS and WarpX again, both policies and both controls, with the
# two invocation errors phases 1 and 2 made corrected.
#
# WarpX --bin. run_config.sh defaults WARPX_BIN to the first binary under
# ~/src/warpx/build-clio/bin, and that build links the SYSTEM HDF5 1.10, which
# has no VOL plugin API at all. WarpX then runs to completion, writes a perfect
# 25 GB of native openPMD-HDF5, and stages ZERO chunks -- the connector was
# never loaded, so nothing was ever intercepted. build-h5114 is the build
# against the 1.14.4 prefix the connector needs. (The runner does catch this:
# it reports NO FIELD CHUNKS STAGED and exits 1. It just cannot pick the right
# binary on its own.)
#
# LAMMPS no --f32. The K=31 campaign this arm is compared against dumped
# FLOAT64 -- 3,704,832,000 payload bytes over 1206 chunks, exactly twice what
# --f32 produces. Half the bytes is also half the chunks, and a learning curve
# over 603 chunks is not the same experiment as one over 1206.
set -u
PB=/home/cc/clio-core/paper-benchmark
OUT=${OUT:?}
RUNS=$OUT/runs
LOGS=$OUT/logs
WARPX_H5114=$HOME/src/warpx/build-h5114/bin/warpx.3d.NOMPI.CUDA.SP.PSP.OPMD.EB.QED
mkdir -p "$RUNS" "$LOGS"

[ -x "$WARPX_H5114" ] || { echo "no HDF5-1.14 WarpX at $WARPX_H5114" >&2; exit 1; }

prune() {
  local store=$1
  rm -rf "$store/run" "$store/chi_bdev.dat"
  rm -f  "$store"/cte_tier.dat* 2>/dev/null
}
note() { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOGS/campaign.log"; }

run_one() {
  local wl=$1 cfg=$2; shift 2
  local tag="${wl}_${cfg}"
  local t0=$(date +%s)
  note "START $tag (fix)"
  ( cd "$PB" && "$@" ) > "$LOGS/$tag.log" 2>&1
  local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  local sel="$RUNS/$tag/selection.csv"
  local n=0; [ -f "$sel" ] && n=$(( $(wc -l < "$sel") - 1 ))
  note "DONE  $tag rc=$rc ${dt}s chunks=$n"
  prune "$RUNS/$tag"
}

for cfg in learn learn-ratio dynamic dynamic-ratio; do
  run_one lammps "$cfg" ./lammps/run_config.sh "$cfg" \
      --box 40 --steps 2000 --gap 10 --require-device \
      --chunk 4194304 --bw 5e6 --eb 1e-3 \
      --results "$RUNS" --tag "lammps_$cfg"
done

for cfg in learn learn-ratio dynamic dynamic-ratio; do
  run_one warpx "$cfg" ./warpx/run_config.sh "$cfg" \
      --bin "$WARPX_H5114" \
      --steps 2000 --interval 10 --chunk 1048576 --stage-h2d \
      --bw 5e6 --eb 1e-3 \
      --results "$RUNS" --tag "warpx_$cfg"
done

note "FIX CAMPAIGN COMPLETE"
