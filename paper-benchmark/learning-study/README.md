# The learning-mode campaign

Does NeuroPress's online SGD actually move the model as a run proceeds?

16 runs: four workloads x {balanced, ratio-only cost} x {learning on, off}.
Exploration is OFF in all sixteen -- that is what makes this the learning arm
and not the exploration arm. The model trains on the ONE action it actually
picked, chunk after chunk, with no measured alternatives to learn from.

    <workload>/learn.*          online SGD, balanced cost (w_ct = w_dt = w_io = 1)
    <workload>/learn-ratio.*    online SGD, ratio-only cost (w_ct = w_dt = 0)
    <workload>/dynamic.*        CONTROL: same weights, SGD off
    <workload>/dynamic-ratio.*  CONTROL: same weights, SGD off

**The controls are the point.** A cost-prediction error that falls over a run
proves nothing on its own: the data gets easier or harder as the physics
develops, and that motion would otherwise be credited to learning. Each
control is the same policy on the same input with only the SGD removed.

## Files

```
learning_error.png        cost-model error over the run, learning vs control
learning_ratio.png        achieved compression ratio over the run
learning_table.txt        the numbers both figures are read against
learning_summary.json     the same, machine-readable
<workload>/perchunk_<run>.png          one chunk per x-position, every SGD
                                       gradient marked
<workload>/perchunk_<run>_<field>.png  the same for one variable
<workload>/<config>.selection.csv.gz   the per-chunk log the figures come from
<workload>/<config>.meta.json          that run's parameters, self-describing
regenerate.sh                          rebuilds every figure above from them
logs/campaign.log                      one timeline for all sixteen runs
logs/<workload>_<config>.log           that run's console output
drivers/*.sh                           the scripts that produced the campaign
```

**`logs/campaign.log` records the failures too**, and deliberately. It is the
whole timeline, so it shows the four LAMMPS runs that produced 603 chunks
instead of 1206 and the four WarpX runs that produced `rc=1 chunks=0`, then
shows the same eight redone under `(fix)` once the two invocation errors
below were found. Only the corrected runs are kept as data; the per-run
`.log` files are the corrected ones, because phase 3 wrote over them. A
timeline that quietly began at the third attempt would be the wrong record.

Tier images, bdev files and WarpX's 25 GB of native openPMD-HDF5 were deleted
after each run -- about 250 GB across the sixteen. What is here regenerates
every figure.

## The SGD gate is RECOMPUTED, not read

`selection.csv` has no `sgd` column, but every input to the gate is in it, so
`../plot/viz_learning.py` transcribes the runtime's own arithmetic
(`compressor_runtime.cc:1486-1500`):

```
cost(ct, dt, r) = w_ct*max(1,ct) + w_dt*max(1,dt) + w_io*bytes/(min(r,cap)*bw)
predicted       = cost(pred_ct_ms,   pred_dt_ms, pred_ratio)
actual          = cost(actual_ct_ms, pred_dt_ms, actual_ratio)
error_pct       = |actual - predicted| / actual      SGD fires above 0.30
```

Note `pred_dt_ms` on both lines. Decompression time is not measured at write
time, so the runtime scores the PREDICTED value on both sides and the term
cancels out of the difference; substituting anything else would give a curve
no training decision was ever made from. The `pred_*` columns are written
from the same `CompressionStats` the gate scores, selected by the same
predicate (`logged_pred`, compressor_runtime.cc:1351).

It could NOT be corroborated against the runtime's own log: the SGD line is
`HLOG(kDebug, ...)` and `CTP_LOG_LEVEL` defaults to `kInfo` at COMPILE time,
so those messages are compiled out of this build. `CTP_LOG_LEVEL=debug` in the
environment cannot bring back what was never emitted.

## The CSV

`<config>.selection.csv.gz` is `CLIO_NEUROPRESS_SELECTION_LOG`, one row per
chunk in the order the run produced them, header written by
`neuropress_telemetry.cc:210`:

| column | meaning |
|---|---|
| `seq` | row number |
| `blob` | `<frame>/<field>/chunk_<n>` — the field parser handles all three spellings (`plt00007/fab0000_comp00_density`, `step00010/E_x`, `position/step_0`) |
| `chunk_bytes` | the chunk the decision was made for |
| `entropy`, `mad`, `second_deriv` | the three statistics the model consumes, and its ONLY inputs besides the size |
| `wire_lib`, `lib_name`, `algo_idx` | the codec picked; `algo_idx` is NeuroPress's own 0-7 index |
| `quantize`, `shuffle`, `preset` | the rest of the action. `shuffle` is the WIDTH (0/2/4/8), not a flag |
| `pred_ratio`, `pred_ct_ms`, `pred_dt_ms`, `pred_psnr` | what the model said |
| `actual_ratio`, `actual_ct_ms`, `actual_psnr` | what happened. `actual_psnr` is -1 when quality measurement is off |
| `checksum` | FNV-1a over the chunk, so a repeat is identifiable |
| `role` | `primary` everywhere here; exploration runs also write `adopted` |
| `select_us` | selection latency, -1 when the chunk never reached the model |
| `reused` | THREE-valued: 0 forward pass, 1 ranking reused, -1 never reached the model |

There is no `sgd` column — see the section above for how the gate is
recovered from the columns that are here.

`blobs.csv` is not kept: it is the tier's view of the same chunks, and every
column the figures need is in the selection log.

## Reading the figures

`learning_error.png` -- solid is learning, dashed is the same policy with SGD
off. In 7 of 8 arms the solid line sits below its own control.

`perchunk_*.png`, third panel -- the cumulative gradient count. A KNEE is
convergence: `vpic/perchunk_vpic_learn-ratio_ex.png` takes 25 gradients, all
but one before chunk 110, and the predicted line visibly settles onto the
actual one. A STRAIGHT RAMP is the opposite: `nyx/perchunk_nyx_learn-ratio_
density.png` takes 343 gradients over 816 chunks at a constant rate and never
converges, because the actual ratio runs to 1000-5000x while the prediction is
pinned at the 100x cap. It is being corrected towards a number it cannot
represent.

## Three things that make a flat curve mean nothing

Each is reported per run in `learning_table.txt`.

- **capped** Under ratio-only weights the cost is `bytes/(min(ratio,cap)*bw)`
  and nothing else, so once the predicted AND actual ratio both clear the 100x
  cap the two costs are the same number by arithmetic. `error_pct` is then
  exactly 0 and SGD cannot fire at any threshold.
- **reused** The `reused` column is THREE-valued. 0 = a forward pass ran.
  1 = the previous ranking was reused, so the error is inherited rather than
  earned. -1 = the chunk never entered `NeuroPressRankChunk` at all, so no
  gradient could come from it under any weights. On Nyx, VPIC and LAMMPS that
  last population is 18-62% of chunks; the table repeats the error over the
  inference-only subset for this reason.
- **field interleaving** Consecutive chunks of a run belong to different
  variables, so a run-order delta measures which variable came next. The
  "prediction moved after a gradient" statistic groups by field first.

## Parameters

Copied from the K=31 campaign's `meta.json` so the two arms compare directly:
eb 1e-3, bw 5e6, 2000 timesteps each. Nyx and VPIC replay their field dumps at
a 4 MiB chunk; LAMMPS simulates box 40 / gap 10 in situ at 4 MiB; WarpX
simulates 64x64x512 / diag every 10 in situ at 1 MiB.

Two invocation errors were found and the affected runs redone. WarpX defaults
`WARPX_BIN` to `~/src/warpx/build-clio`, which links the system HDF5 1.10 and
has no VOL plugin API: WarpX completed 2000 steps, wrote a perfect 25 GB of
native openPMD, and staged ZERO chunks. `build-h5114` is the build against the
1.14.4 prefix the connector needs. And LAMMPS must NOT be given `--f32` here:
the K=31 campaign dumped float64, 1206 chunks, exactly twice what `--f32`
gives.

## Regenerating

```bash
PYTHON=<interpreter with matplotlib> ./regenerate.sh [OUTDIR]
```

rebuilds all 19 figures plus the table and the JSON from this directory alone.
Verified: every one comes back BYTE-IDENTICAL, gzip and all. The only inputs
are the `.selection.csv.gz` / `.meta.json` pairs here; the only things outside
are `../plot/viz_learning.py` and matplotlib (the system `python3` on this
machine does not have it, hence `PYTHON=`).

The two commands `regenerate.sh` drives, if you want one figure rather than
all of them:

```bash
../plot/viz_learning.py trend --out DIR \
    --run vpic:learn:vpic/learn.selection.csv.gz \
    --run vpic:learn-control:vpic/dynamic.selection.csv.gz ...
../plot/viz_learning.py perchunk --out DIR \
    --run vpic_learn-ratio:vpic/learn-ratio.selection.csv.gz --field ex
```

Both accept either a live run store (a directory holding `selection.csv` and
`meta.json`) or a log named directly, gzipped or not, with the matching
`<config>.meta.json` beside it -- that is where the cost weights come from.

What is NOT here is the campaign itself: re-running the sixteen runs needs the
workload dumps (about 50 GB for Nyx and VPIC) and the three driver scripts,
which stayed in the session scratch. The parameters above and the two
invocation corrections are enough to reconstruct the invocations.
