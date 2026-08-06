# Issue #526 perf-eval pipelines — Ares setup guide

This directory holds the two Jarvis regression pipelines for Issue #526:

- [single_node.yaml](single_node.yaml) — one node, a 4×6 (I/O size × thread
  count) sweep over four storage stacks = **24 rows**.
- [distributed.yaml](distributed.yaml) — four nodes, a 1→2→4 client-node scaling
  sweep over three stacks = **3 rows**.

Both run their real work inside a pre-built Apptainer container (the SIF). This
guide takes you from a **fresh Ares account that has only Spack and a clio-core
checkout** to a green `jarvis ppl submit`. No conda, and no assumptions beyond
those two things.

---

## Quick reference — full sequence

The whole setup, if you accept every default. Each step is explained in detail
below; start there if anything fails or you need to deviate.

```bash
# 0. prerequisites: on Ares login node, spack on PATH, $CLIO_REPO and $JARVIS_CD set
export CLIO_REPO="$HOME/clio-core"
export JARVIS_CD="$HOME/jarvis-cd"

# 1. jarvis venv
python3 -m venv "$HOME/jarvis-venv"
"$HOME/jarvis-venv/bin/pip" install -e "$JARVIS_CD"
export PATH="$HOME/jarvis-venv/bin:$PATH"

# 2. init jarvis + build the SIF + register the clio repo
jarvis init
bash "$JARVIS_CD/docker/build_perf_eval_image.sh"
jarvis repo add "$CLIO_REPO/jarvis_clio_core"

# 3. submit — apptainer is installed automatically on the first run
jarvis ppl submit "$CLIO_REPO/jarvis_clio_core/pipelines/ares/single_node.yaml"
jarvis ppl submit "$CLIO_REPO/jarvis_clio_core/pipelines/ares/distributed.yaml"
```

> **Apptainer is not in this list, deliberately.** Both YAMLs declare it under
> `config.host_pkgs`, and Jarvis installs it on the first submit that finds it
> missing. Setting it up by hand used to be steps 2 and 3 — a `spack install`
> plus a `spack view` on a shared filesystem, wired into the job through a
> `PERF_EVAL_VIEW` override. All of that is gone. See
> [Apptainer, and how it gets there](#apptainer-and-how-it-gets-there).

---

## The mental model (read this first)

A submit succeeds when three things are true:

1. **`jarvis` is on `PATH`** — the launcher itself; the generated sbatch script
   calls bare `jarvis ppl run`. Provided by a **dedicated Python venv**
   (`pip install -e <jarvis-cd>`). This is the one binary you still place by
   hand, because it is the thing that would otherwise place itself.
2. **`apptainer` is installed, or installable** — it launches the per-node
   containers. The YAMLs declare it as a host prerequisite and Jarvis
   provisions it; you only need `spack` reachable. See the next section.
3. **The SIF is staged** — `iowarp-perf-eval.sif`, under
   `<jarvis shared_dir>/containers/`, so the YAMLs resolve it by basename.

The pipelines never `spack load` or `conda activate` in the job script — those
install shell functions that misbehave under `set -euo pipefail`. What the
pre_cmds do is a plain `PATH` assignment, driven by three env vars you can
override at submit time:

| Env var | Default | What it points at |
|---|---|---|
| `JARVIS_VENV` | `$HOME/jarvis-venv` | The jarvis venv (step 1) |
| `CLIO_REPO` | `$HOME/clio-core` | This clio-core checkout (used for `jarvis repo add`) |
| `SPACK_ROOT` | `$HOME/spack` | Spack install root, used to resolve and install apptainer |

If you accept every default below, you never set any of these. Override them
**before** `jarvis ppl submit` (they are read when the job runs, not when it is
queued).

---

## Apptainer, and how it gets there

Each YAML carries this:

```yaml
config:
  host_pkgs:
    - install_method: spack
      install_query: apptainer~suid
```

Before the pipeline does anything else, Jarvis:

1. **probes** with `spack location -i apptainer~suid` (which matches only specs
   that are genuinely installed — `spack find` also matches specs merely known
   to spack);
2. **installs** with `spack install apptainer~suid` if that came back empty,
   streaming the build to the job log — it builds Go and a FUSE stack from
   source and takes the better part of an hour the first time;
3. **activates** by folding what `spack load apptainer~suid` sets into its own
   process environment, so the `apptainer instance start` / `build` calls
   resolve the binary.

A run that finds apptainer already installed skips straight to (3). A failed
install is not retried within a process, so a 24-combination sweep does not
re-enter the same doomed build 24 times.

**`~suid` is mandatory** and is part of the spec for that reason. The builtin
recipe defaults to `+suid`, which builds a `starter-suid` that must be
root-owned — unbuildable unprivileged, and fatal to the rootless `--fakeroot`
path these pipelines need for the CTE and JuiceFS FUSE mounts.

**`SPACK_ROOT` has to reach the job.** Jarvis sources
`$SPACK_ROOT/share/spack/setup-env.sh` to make `spack` callable; the shell
function your rc file installs is invisible to the non-interactive bash it
probes with. `sbatch` exports the submitting environment by default, so if
`spack` works in the shell you submit from, this is already true — check the
`spack_root=` field in the job log's `toolchain:` line.

> **Shared-filesystem requirement.** The Spack install tree, the venv, and the
> SIF must all live on a filesystem the compute nodes can see; on Ares both
> `$HOME` (NFS) and `/mnt/common` qualify. This is what lets a single install on
> the head node serve all four nodes of `distributed.yaml`. If your Spack
> install root is on node-local disk, move it.

### Doing it by hand

You never need to, but the install is an ordinary spack install:

```bash
spack install apptainer~suid
```

If that fails with `error obtaining VCS status: exit status 128`, you are on the
login node and hit a Go VCS-stamp bug present in apptainer ≤ 1.4.4. Either build
on a compute node (`srun -p debug -N1 -n1 --pty bash`), or set
`GOFLAGS=-buildvcs=false` in the environment you submit from — the pipeline's
install inherits it. The bug is fixed in 1.5.0; if your spack's builtin repo is
too old to offer it, `spack repo update builtin`.

---

## Prerequisites

- You are on an **Ares login node**.
- **Spack is installed and on `PATH`** (`spack --version` works — i.e. you have
  sourced `share/spack/setup-env.sh`), and `SPACK_ROOT` is exported. Jarvis
  needs it to install apptainer; see
  [Apptainer, and how it gets there](#apptainer-and-how-it-gets-there).
- **clio-core is checked out** (this repo). Its path is your `CLIO_REPO`
  (default `$HOME/clio-core`).
- **jarvis-cd is checked out at `dev`** — it carries the container deploy,
  `container_fakeroot`, `run_timeout`, `host_pkgs`, the Slurm `scheduler:`
  block, and the SIF build script these pipelines depend on. Call its path
  `$JARVIS_CD` below.
- **Docker is available** on the machine where you build the SIF (step 3). The
  SIF build is a `docker build` followed by an `apptainer build
  docker-daemon://…`.

---

## Step 1 — Create the jarvis venv (the `jarvis` on `PATH`)

A dedicated venv keeps jarvis off conda and makes the launcher environment
reproducible:

```bash
python3 -m venv "$HOME/jarvis-venv"
"$HOME/jarvis-venv/bin/pip" install -e "$JARVIS_CD"
export PATH="$HOME/jarvis-venv/bin:$PATH"
command -v jarvis                         # -> $HOME/jarvis-venv/bin/jarvis
```

`pip install -e` pulls jarvis's Python deps; any gap surfaces here (cheap),
before you ever queue a job.

---

## Step 2 — Initialize Jarvis

This writes `~/.ppi-jarvis/*.yaml`, which defines `shared_dir` — where the SIF
build stages the image and where the pipelines look for it:

```bash
jarvis init                               # accepts defaults; shared_dir = ~/.ppi-jarvis/shared
```

`~/.ppi-jarvis/shared` is on NFS home, so it is node-visible and fine. To put it
on `/mnt/common` instead:
`jarvis init ~/.ppi-jarvis/config ~/.ppi-jarvis/private /mnt/common/$USER/jarvis-shared`.

---

## Step 3 — Build the SIF

With `jarvis` (venv) on `PATH` from step 1, build the image. This does a `docker
build` then converts it to a SIF and stages it at
`<shared_dir>/containers/iowarp-perf-eval.sif`:

```bash
bash "$JARVIS_CD/docker/build_perf_eval_image.sh"
```

- It reads `shared_dir` from `~/.ppi-jarvis/*.yaml`, so step 2 must be done
  first.
- The `apptainer build` half needs apptainer on `PATH` **here**, in your
  interactive shell. This is the one place the `host_pkgs` activation does not
  reach, because the script is not run by Jarvis. If the pipelines have already
  installed it, `spack load apptainer~suid` is enough; otherwise run
  `spack install apptainer~suid` first.
- It bakes in clio-core at `CLIO_REF` (default `dev`) — the resolved commit SHA
  is echoed; record it, that is exactly what is inside the SIF.
- Re-run it daily so the sweep tracks the latest IOWarp. Useful overrides:
  `CLIO_REF=<branch|sha>`, `IOWARP_SPEC="iowarp@dev +fuse"`, `SKIP_DOCKER_BUILD=1`
  (reuse an existing local docker image).

---

## Step 4 — Register the clio-core package repo

The pipelines' pre_cmds run `jarvis repo add "$CLIO_REPO/jarvis_clio_core"`
idempotently, but doing it once now confirms the clio packages
(`clio_runtime`, `clio_cte`, `clio_cte_libfuse`) import cleanly:

```bash
jarvis repo add "$CLIO_REPO/jarvis_clio_core"
jarvis repo list                          # jarvis_clio_core should appear
```

---

## Step 5 — Submit

Hostfiles are **generated automatically** by the scheduler from
`$SLURM_JOB_NODELIST` at job start — you do **not** create
`~/.jarvis-perf-eval/*_hostfile.txt` by hand.

```bash
jarvis ppl submit "$CLIO_REPO/jarvis_clio_core/pipelines/ares/single_node.yaml"
# then, when that is green:
jarvis ppl submit "$CLIO_REPO/jarvis_clio_core/pipelines/ares/distributed.yaml"
```

`single_node.yaml` requests 1 node on `debug`; `distributed.yaml` requests 4. If
you accepted every default above, submit with no extra env. Otherwise export
your overrides first, e.g.:

```bash
JARVIS_VENV=$HOME/jarvis-venv CLIO_REPO=$HOME/src/clio-core SPACK_ROOT=$HOME/spack \
  jarvis ppl submit "$CLIO_REPO/jarvis_clio_core/pipelines/ares/single_node.yaml"
```

The **first** submit is the slow one if apptainer is not installed yet: Jarvis
runs `spack install apptainer~suid` before the pipeline starts, and that build
takes the better part of an hour. It streams into the job log, so you can watch
it. Every later submit finds the spec installed and skips straight past it.

---

## Verifying a run

1. In the `.out` log, the toolchain echo must point at **your venv and your
   spack**:
   ```
   toolchain: jarvis=/home/<you>/jarvis-venv/bin/jarvis spack_root=/home/<you>/spack apptainer=/home/<you>/spack/opt/spack/.../bin/apptainer
   ```
   A miniconda `jarvis` path means the venv did **not** take effect. An empty or
   wrong `spack_root=` means `SPACK_ROOT` did not reach the job, and the
   `host_pkgs` install will fail with *"spack is not on PATH and SPACK_ROOT is
   unset"*. `apptainer=<not yet installed…>` is expected and fine on a first
   run — Jarvis installs it a moment later.
2. The post_cmd prints `ALL 24 GREEN` (single_node) / `ALL 3 GREEN`
   (distributed).
3. **Check the numbers, not just the color.** Open the results CSV
   (`$HOME/single_node_results/results.csv` or `$HOME/distributed_results/results.csv`)
   and confirm the throughput columns are populated — a green row with a blank
   `*_max_mibs` column is a failure. Sane single-node reference: `nfs_ior` 4k read
   scales roughly 380 → 5000+ MiB/s across 1 → 32 ranks.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| *1 required host package(s) … could not be provisioned* | The `host_pkgs` install ran and apptainer is still undetectable. Run `spack install apptainer~suid` by hand to see the build failure in full — the rows below cover the usual ones. |
| *spack is not on PATH and SPACK_ROOT is unset* | `SPACK_ROOT` did not reach the job. `sbatch` exports the submitting environment by default, so export it in the shell you submit from; confirm with the `spack_root=` field in the job log's `toolchain:` line. |
| `error obtaining VCS status: exit status 128` during the apptainer build | A Go VCS-stamp bug in apptainer ≤ 1.4.4, triggered on the login node. Export `GOFLAGS=-buildvcs=false` in the shell you submit from — the `host_pkgs` install inherits it. Or install by hand from a compute node, where a clean `/tmp` stage tree never trips it. Fixed in 1.5.0. |
| `apptainer@1.5.0` is an unknown version | Stale Spack packages repo: `spack repo update builtin`. |
| `go@1.25.7` won't concretize (needed by 1.5.0) | Pin a fallback in the pipeline's `install_query`: `apptainer@1.4.4~suid` (needs `go@1.23.6`) or `apptainer@1.3.6~suid` (needs `go@1.20`). |
| `--fakeroot`: *newuidmap must be owned by the root user* | Something has put a non-setuid `newuidmap` ahead of the system's setuid-root `/usr/bin/newuidmap` on `PATH` — classically spack's `shadow`, which apptainer pulls in via `+libsubid`. The pipelines only add apptainer's own prefix to `PATH`, never its dependency closure, so this should not happen; check for a stray `spack load` or view in your environment. |
| `--fakeroot`: *Target N is owned by a different user … gid mismatch* | Slurm gave the job the project GID. The pipelines already re-exec under `exec sg "$USER"` in pre_cmd #1; for a **hand-run** smoke, wrap it yourself: `sg "$USER" -c 'apptainer instance start --fakeroot <SIF> t'`. |
| `ERROR: SIF missing: …/iowarp-perf-eval.sif` | The SIF was never staged — run **step 3** (`build_perf_eval_image.sh`). |
| `ERROR: jarvis not on PATH` in the job log | Your `JARVIS_VENV` override does not match where you built the venv, or the venv is on node-local disk. Fix the override or move it onto a shared FS. |
