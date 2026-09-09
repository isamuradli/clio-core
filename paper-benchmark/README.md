# paper-benchmark

Four production simulation workloads — **Nyx**, **VPIC**, **WarpX**, **LAMMPS** —
run through Clio's compressor with NeuroPress selecting a codec per chunk. This
directory produces every number and figure in the paper's evaluation, plus the
evidence behind the benchmark's own configuration choices.

Start here for orientation. [`BENCHMARK.md`](BENCHMARK.md) is the reference: it
explains the cost models, the error-bound semantics, device residency, and the
findings that change how results should be read. It is organised by topic, not
by script, so use the index below to find the tool and that document to
understand it.

```bash
./run_benchmark.sh --smoke          # 16 cells, exercises every path, minutes
./run_benchmark.sh --dry-run        # print the matrix without running it
./run_benchmark.sh --profile mid    # ~5-10 min per run
./run_paper.sh                      # the paper matrix: 6 configs x 4 workloads
```

Requires an NVIDIA GPU. `numpy pandas scipy scikit-learn matplotlib` for the
analysis and plotting scripts; `h5dump` (poppler's `hdf5-tools`) for anything
reading WarpX's openPMD output.

---

## Which script do I run?

### Run a workload

| script | what it does |
|---|---|
| `run_benchmark.sh` | the whole matrix — four workloads, both cost models, lossless and lossy |
| `run_paper.sh` | the paper's six-cell matrix per workload |
| `<workload>/run_config.sh` | **one** named configuration of one workload |
| `nyx/run_config_insitu.sh`, `vpic/run_config_insitu.sh` | the same, but GPU-resident: the simulation hands the compressor a device pointer instead of replaying dumps |
| `<workload>/gen_fields.sh` | write field dumps to disk, for the replay route and for the evolution metric |
| `<workload>/run_sweep.sh` | one workload across several configurations |

WarpX has no replay route: a stock, unmodified WarpX writes through Clio's HDF5
VOL, so `warpx/run_config.sh` always runs the simulation.

### Compare configurations

| script | what it does |
|---|---|
| `compare_wallclock.sh` | NeuroPress against fixed codecs and cuSZ/cuSZp3/ndzip, end to end |
| `compare_perchunk_oracle.sh` | how close the selector gets to a per-chunk oracle |
| `wallclock_table.py` | the summary table for `compare_wallclock.sh` |
| `perchunk_oracle_tables.py` | the tables for `compare_perchunk_oracle.sh` |

### Aggregate and check a run

| script | what it does |
|---|---|
| `collect.py` | a whole sweep → `summary.csv` |
| `metrics.py` | one cell → `metrics.json` |
| `audit_run.py` | cross-check one run: does the tier hold what the log claims? |

### Ask why a chunk compressed the way it did

| script | what it does |
|---|---|
| `analysis/analyze_exploration.py` | **the complete analysis pipeline** over an exploration log — mechanism, modelling, cross-workload comparison, figures. One command; the flags only turn work *off*. See [`analysis/README.md`](analysis/README.md). |
| `analysis/plot_ratio_explained.py` | why one chunk gets the ratio it gets |
| `analysis/plot_three_properties.py` | what each of the three data properties does to compressibility |
| `analysis/redundancy_probe.py` | temporal redundancy the run leaves unexploited |

### Measure whether the data evolves

| script | what it does |
|---|---|
| `evolution.py` | **the metric.** Per-block change `E` between consecutive dumps, and the share of cells bit-identical to the previous dump. Reads all four workloads' dump formats (`--source f32 / openpmd / raw`). |
| `evolution_pass.sh` | runs the metric *and* the begin/middle/end figure for a workload, then deletes the dumps |
| `evolution_rank.py` | ranks a workload's configurations from their `evolution.json` — re-scores without re-running anything |
| `analysis/plot_block_evolution.py` | every block's trajectory, the early-vs-late settle test, and cell-level reusability |

### Look at the data

| script | what it does |
|---|---|
| `plot/figure_evolution.py` | one field at the first, middle and last frame, **one shared colour scale** across the three so a panel that looks empty *is* empty |
| `plot/figure_lossy.py` | the same three frames, original against decompressed, plus the error map |
| `plot/viz_fields.py` | a full montage and GIF of an f32 dump sequence, with blast-wave diagnostics |
| `plot/viz_selection.py {actions,bound,chunks}` | a run chunk by chunk: what the model saw, what it picked, and whether the error bound did anything |
| `plot/viz_openpmd.py`, `plot/viz_atoms.py` | the same for WarpX's openPMD fields and LAMMPS's atom state |

`plot/figure_evolution.py` refuses to write a blank plate: a slice that is
identically zero while the volume is not means the plane or the shape is wrong,
not that the data is static. Pass `--shape` for a non-cubic grid — WarpX's
64×64×512 has exactly 128³ cells, so the cube-root test *passes* and silently
reshapes a slab into a cube.

---

## Running the large workloads on one local GPU

These are the campaign runs the evolution study's figures are regenerated from.
They were previously driven by Slurm jobs on Delta; the parameters below are
those jobs' own, with the cluster paths localised. **Note the 8 MiB chunk** —
the campaign uses `--chunk 8388608`, not the 4 MiB default.

```bash
# --- Nyx, 30 GB campaign cell -------------------------------------------
./nyx/gen_fields.sh --ncell 128 --steps 6400 --plot-int 10 --out "$FIELDS"
CLIO_NEUROPRESS_STAGE_H2D=1 ./nyx/run_config.sh explore-balance \
    --fields "$FIELDS" --bw 5e6 --eb 0.05 \
    --explore-k 31 --explore-thresh -1 --chunk 8388608 \
    --results "$RESULTS" --tag nyx_lossy_balance

# --- Nyx, K=31 exploration ----------------------------------------------
CLIO_NEUROPRESS_STAGE_H2D=1 ./nyx/run_config.sh explore-balance \
    --fields "$FIELDS" --bw 5e6 --eb 1e-3 \
    --explore-k 31 --chunk 8388608 --results "$RESULTS" --tag nyx_k31

# --- VPIC, 30 GB campaign cell (in situ, GPU-resident) ------------------
./vpic/run_config_insitu.sh explore-balance \
    --ncell 126 --steps 6000 --int 25 \
    --chunk 8388608 --bw 5e6 --eb 0.05 \
    --explore-k 31 --explore-thresh -1 --results "$RESULTS" --tag vpic_lossy_balance

# --- VPIC at the default evolving configuration, 1,000 steps ------------
# 200 steps is NOT enough: the run is still in the noise phase and the first
# blob compresses 1.006x. At 1,000 the Weibel instability has grown -- `cby`
# amplitude goes from +/-0.045 to +/-0.169, with 0 of 24 dumps bit-identical.
./vpic/run_config_insitu.sh explore-balance \
    --ncell 126 --steps 1000 --int 25 --chunk 8388608 \
    --bw 5e6 --eb 1e-3 --explore-k 3 --results "$RESULTS" --tag vpic_evolving

# --- LAMMPS -------------------------------------------------------------
./lammps/run_config.sh explore-balance \
    --box "$BOX" --steps "$STEPS" --gap "$GAP" --f32 --require-device \
    --chunk 8388608 --bw 5e6 --eb 1e-3 --explore-k 31 \
    --results "$RESULTS" --tag lammps
```

`--require-device` on LAMMPS is a correctness flag, not a performance one:
without it the driver gathers each chunk into host memory, where NeuroPress's
CUDA-only shuffle and quantizer refuse, and the blobs are silently lost.

Three things that bite on a single local GPU:

- **Nyx must scale its grid with the step count.** A CFL timestep halves when
  the grid doubles and the Sedov shock advances a roughly fixed number of
  *cells* per step, so 2,000 steps at 256³ ends at the same physical state as
  1,000 at 128³. Running 2,000 at 128³ instead drives the front into the domain
  boundary and spends the second half of the run measuring a blast with nowhere
  to go.
- **Nyx replay needs `CLIO_NEUROPRESS_STAGE_H2D=1`.** Replay reads `.f32` into
  host shm, and `CLIO_NEUROPRESS_REQUIRE_DEVICE` refuses it. Staging copies the
  chunk up and runs the CUDA kernels on it; the in-situ route needs no such flag
  because AMReX hands over a device pointer directly.
- **WarpX's metric needs `h5dump`** (`hdf5-tools`). `evolution.py --source
  openpmd` and `analysis/validate/warpx_gen_fields.sh` both shell out to it, and
  neither fails gracefully when it is absent.

---

## Where the paper's figures come from

The evaluation's heterogeneity section rests on three figures, each computed
from one measurement file rather than regenerated at paper time. Those files
have been **cleared from `evolution-study/`** ahead of a new campaign; restore
them with `git checkout 6ce4226f~1 -- paper-benchmark/evolution-study/`. The
simulation dumps behind them were never kept at all — about 171 GB across the
four sweeps, deleted by each `run*.sh` after measuring.

The table is the record of what the *published* figures were computed from, so
a regenerated figure can be checked against the numbers the old one reported.

| figure | source file (in history, not the tree) | the numbers it produced |
|---|---|---|
| **Fig. 3** — activity is spatially localized | `evolution-study/nyx/e10_cfl08.blocks.csv.gz` | group `pct_cells_same` by block index, density field, last frame pair → outermost 85.3%, central 26.4% |
| **Fig. 4** — compression varies within one dump | `evolution-study/nyx-20gb/insitu.blobs.csv.gz` | 233× spread at step 1559; median within-dump spread 61× over 300 dumps; 216 of 300 dumps assigned more than one codec |
| **Fig. 5** — the same measurement on four workloads | `evolution-study/{nyx/e10_cfl08, warpx/baseline, vpic/baseline, lammps/melt_hot_nb}.json` | the `interval_means` and `interval_pct_cells_same` series in each summary |

Note that Fig. 5 uses each workload's **`baseline`** configuration for VPIC and
WarpX, not the default the evolution study selected. For VPIC the difference is
load-bearing: `baseline` runs `clean_div_e_interval = 0`, under which
`div_e_err`, `div_b_err`, `rhob` and `rhof` are never recomputed and are dumped
unchanged every frame — a quarter of the payload. Its flat 26.9% cell-level
redundancy is partly those four variables. Under the study's own default
(`clean_div = 10`) the same measurement gives 8.33%.

---

## Configuration: what makes each workload evolve

A compression selector's job is to notice when data changes, so a workload that
reaches steady state early says nothing about it. Each workload's README carries
a **"Default Evolving Benchmark Configuration"** section giving the parameters,
the upstream reference for each, the values tested, and the outcome:

| workload | section | selected |
|---|---|---|
| Nyx | [`nyx/README.md`](nyx/README.md) | `nyx.cfl = 0.8`, `exp_energy` at the deck's 1.0 |
| VPIC | [`vpic/README.md`](vpic/README.md) | `VPIC_CLEAN_DIV_INT = 10`, upstream thermal velocities |
| WarpX | [`warpx/README.md`](warpx/README.md) | `laser1.e_max = 32e12` (a0 = 8), moving window **on** |
| LAMMPS | [`lammps/README.md`](lammps/README.md) | `--temp 6.0 --skin 0.8 --every 5` |

These are defaults in the runner scripts, not flags you must remember — running
any `run_config.sh` with no physics arguments gives the studied configuration.
`BENCHMARK.md` §6 summarises all four, and each README's "Parameters tested"
table carries its own outcome numbers, so the reasoning is self-contained.

The 26-configuration sweep those choices were made from has been cleared from
`evolution-study/` pending a new campaign at larger scale. It is still in git —
`git show 256e7c2c` — and `git checkout -- paper-benchmark/evolution-study/`
restores it, including the WarpX configuration disqualified on physics rather
than on score.
