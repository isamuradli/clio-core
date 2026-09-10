#!/usr/bin/env bash
# The learning-mode campaign: every workload under BOTH cost policies.
#
#   learn        online SGD, balanced cost   (w_ct = w_dt = w_io = 1)
#   learn-ratio  online SGD, ratio-only cost (w_ct = w_dt = 0, w_io = 1)
#
# Exploration is OFF in both -- that is what makes this the learning arm and
# not the exploration arm: the model trains on the ONE action it actually
# picked, chunk after chunk, with no measured alternatives to learn from. The
# question is whether that alone moves it.
#
# Every physics/chunk/error-bound parameter is copied from the K=31 campaign's
# meta.json so the two arms are directly comparable; only the config differs.
set -u
PB=/home/cc/clio-core/paper-benchmark
OUT=${OUT:?}
RUNS=$OUT/runs
LOGS=$OUT/logs
mkdir -p "$RUNS" "$LOGS"

# Keep the CSVs and the logs, drop the tier/bdev images and WarpX's 24 GB of
# native HDF5. Eight runs would otherwise be ~250 GB of bytes nothing reads.
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
  note "START $tag"
  ( cd "$PB" && "$@" ) > "$LOGS/$tag.log" 2>&1
  local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  local sel="$RUNS/$tag/selection.csv"
  local n=0; [ -f "$sel" ] && n=$(( $(wc -l < "$sel") - 1 ))
  note "DONE  $tag rc=$rc ${dt}s chunks=$n"
  prune "$RUNS/$tag"
}

for cfg in learn learn-ratio; do
  run_one nyx "$cfg" env CLIO_NEUROPRESS_STAGE_H2D=1 ./nyx/run_config.sh "$cfg" \
      --fields "$PB/nyx/fields" --bw 5e6 --eb 1e-3 --chunk 4194304 \
      --results "$RUNS" --tag "nyx_$cfg"
done

VPIC_FIELDS=${VPIC_FIELDS:?}
for cfg in learn learn-ratio; do
  run_one vpic "$cfg" env CLIO_NEUROPRESS_STAGE_H2D=1 ./vpic/run_config.sh "$cfg" \
      --fields "$VPIC_FIELDS" --bw 5e6 --eb 1e-3 --chunk 4194304 \
      --results "$RUNS" --tag "vpic_$cfg"
done

for cfg in learn learn-ratio; do
  run_one lammps "$cfg" ./lammps/run_config.sh "$cfg" \
      --box 40 --steps 2000 --gap 10 --f32 --require-device \
      --chunk 4194304 --bw 5e6 --eb 1e-3 \
      --results "$RUNS" --tag "lammps_$cfg"
done

for cfg in learn learn-ratio; do
  run_one warpx "$cfg" ./warpx/run_config.sh "$cfg" \
      --steps 2000 --interval 10 --chunk 1048576 --stage-h2d \
      --bw 5e6 --eb 1e-3 \
      --results "$RUNS" --tag "warpx_$cfg"
done

note "CAMPAIGN COMPLETE"
