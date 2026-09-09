# External codecs: build setup

Builds the three GPU compressors NeuroPress **cannot select** — cuSZ, cuSZp v3
and ndzip — and turns them on in the Clio build. They are what
`../compare_wallclock.sh` runs as its `static-cusz`, `static-cuszp` and
`static-ndzip` arms.

Only setup lives here. The campaign that once drove these codecs through their
own Slurm phases has been removed; `../compare_wallclock.sh` covers the same
comparison from one GPU.

## Why these codecs are baselines, not candidates

`kNeuroPressTrainedGpuBaseIds = {13,14,15,16,17,18,23,24}`
(`neuropress_bridge.cc`) filters cuSZ/cuSZp/ndzip out of the candidate set
before ranking, and the model has no output slot for them: its action space is
8 nvcomp algorithms x quantize x shuffle = 32. Upstream is the same — its own
trace campaign records ndzip as the per-chunk optimum on ~15% of chunks and
NeuroPress selecting it 0.0% of the time. So these arms measure what the
**action space** gives up, which is a separate question from whether the
**selector** is good.

## One-time setup

```bash
./install_codecs.sh --arch 80        # builds cuSZp v3 + ndzip, patches included
cp env.sh site.sh                    # then edit site.sh with your paths (gitignored)
```

Then configure clio-core with the codecs on `CMAKE_PREFIX_PATH` — see the
message `install_codecs.sh` prints. **Verify `CLIO_CTP_ENABLE_{CUSZ,CUSZP,NDZIP}`
are all `ON`.** If a codec is missing the arm still runs: `WireIdForName` falls
back to zstd and produces a plausible but wrong result, so check the flags:

```bash
grep -E "CLIO_CTP_ENABLE_(CUSZ|CUSZP|NDZIP)" build/CMakeCache.txt
```

`cmake` must be >= 3.28 — clio-core uses `set_tests_properties(DIRECTORY ...)`,
and an older cmake fails configure with hundreds of "Can not find test to add
properties to".

## Two caveats before ranking anything against these arms

- **cuSZ and cuSZp do not hold the error bound.** Measured across this
  benchmark, they missed `eb=1e-3` on 151-276 of 300 chunks wherever it was
  checked. A `compare_wallclock.sh` run at `--eb 1e-3 --check-bound` reproduces
  it: both report `BOUND FAILED` while NeuroPress and the nvCOMP arms pass. The
  overshoot is small — worst case 47 ppm, inclusive-bound rounding rather than
  real accuracy loss — but it means a ratio comparison against them is not
  like-for-like unless the bound is verified.
- **Two cost models give opposite answers.** Ratio-only weights
  (`W_CT=0 W_DT=0 W_IO=1`) favour high-ratio slow codecs; time-inclusive
  (`W_CT=1`) favour fast ones. NeuroPress must be *run* under the weights it is
  then graded on, or it is being scored on an objective it was never given.
  cuSZ's viability in particular is entirely a bandwidth question.
