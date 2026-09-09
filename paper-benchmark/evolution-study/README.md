# The 1,000-timestep evolution study

**This directory is empty pending a new campaign.** Its contents — 26
configurations' worth of summaries, per-block CSVs and sweep scripts — were
cleared in `6ce4226f` because the runs behind them are being replaced at larger
scale, and a stale measurement beside a new figure is worse than none.

Everything is in history:

```bash
git show 256e7c2c --stat                                   # the commit that added it
git checkout 6ce4226f~1 -- paper-benchmark/evolution-study/ # all 78 files back
```

## What it held, and what replaces it

Each workload's default was selected by measuring how fast its data actually
evolves — because a selector's whole job is to notice that data changed, and a
simulation that reaches steady state early says nothing about whether it can.
26 configurations across four workloads, every one run 1,000 timesteps and
sampled every 10, scored by `../evolution.py` and ranked by
`../evolution_rank.py`:

```
<workload>/<config>.json            mean/median/p10/last_quarter, pct_active,
                                    pct_cells_same, and the per-interval series
<workload>/<config>.blocks.csv.gz   one row per (step_from, step_to, field,
                                    block) -- the only per-field breakdown
<workload>/run*.sh                  the sweep that produced them
warpx/FE_*.txt                      WarpX FieldEnergy, the evidence that
                                    do_moving_window=0 is a resonant cavity
nyx-20gb/                           a separate 20 GB record, and the only
                                    in-situ selection/explore/blobs CSVs
```

The conclusions did **not** go with the data. Each workload's README keeps its
"Default Evolving Benchmark Configuration" section — the parameters, the
upstream reference for each, the values tested and the outcome numbers — and
those sections are self-contained. What is gone is the raw evidence beneath
them.

The dumps were never here to begin with: about 171 GB across the four sweeps,
deleted by each `run*.sh` after measuring. Re-running a sweep regenerates them.

## Running the replacement

`../README.md` carries the campaign parameters as local-GPU invocations, and
records which file each of the paper's Figures 3-5 was computed from, so a
regenerated figure can be checked against the numbers the published one
reported.

Two things to clear first:

- **`h5dump` must be installed** (`hdf5-tools`). `../evolution.py --source
  openpmd` and `../analysis/validate/warpx_gen_fields.sh` both shell out to it
  and neither fails gracefully without it.
- **Nyx in situ stores zero blobs and exits 0.** The binary lost its Clio hook
  in a rebuild; `nyx/patches/` carries the raw-field-dump and single-precision
  patches but not that one, which is why it did not survive. The replay route
  is unaffected.
