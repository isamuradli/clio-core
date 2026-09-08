#ifndef _GNU_SOURCE
#define _GNU_SOURCE  // expose O_DIRECT from <fcntl.h> (raw-arm cache-bypass parity)
#endif
/*
 * THREE-WAY Gray-Scott I/O benchmark (the paper figure the advisor asked for):
 * ONE shared computation, THREE storage paths, wall-clock compared:
 *
 *   - raw   : no CLIO. GPU compute -> D2H to pinned host -> pwrite + fsync to disk.
 *   - sync  : CLIO, fused submit-AND-WAIT snapshot (GPU blocks on each PutBlob).
 *   - async : CLIO, fire-all snapshot + drain-at-end (server I/O overlaps GPU compute).
 *
 * SAME COMPUTATION across all three by construction: every arm runs the identical
 * `GsStepKernel` (one thread per cell) through the identical timed `RunSim` loop;
 * only the per-snapshot "sink" differs. So the comparison isolates the I/O + storage
 * backend, not the math.
 *
 * ROUTING AROUND THE iowarp ~16-large-backend ceiling (ADVISOR-REPORT §6a; the limit
 * is by COUNT not bytes, and dataset REUSE hangs — see refire probe): each arm reaches
 * the ~2 GB target with FEWER, BIGGER snapshots (<=~12 distinct datasets, each large),
 * keeping the proven fresh-dataset-per-snapshot recipe. The sync and async arms run in
 * SEPARATE processes (one arm per TEST_CASE invocation), so neither exceeds the ceiling.
 *
 * FAIR STORAGE: the CLIO arms can target a kFile bdev (O_DIRECT to a real file, cache-
 * bypassing) to match the raw arm's disk, or a kRam bdev for a RAM baseline — selected
 * by GSBENCH_BDEV. The raw arm always writes real files + fsync.
 *
 * All knobs are ENV VARS so scaling to 2 GB / switching to disk needs NO recompile:
 *   GSBENCH_N            grid dim (NxN float32)              default 512
 *   GSBENCH_CHUNKS       chunks per snapshot dataset         default 4
 *   GSBENCH_SNAPS        number of snapshots (<= ~12!)       default 4
 *   GSBENCH_STEPS_PER    sim steps between snapshots         default 8
 *   GSBENCH_BDEV         ram | pinned | file  (CLIO arms)    default ram
 *   GSBENCH_BDEV_CAP_MB  bdev capacity (MB)                  default 512
 *   GSBENCH_BDEV_PATH    kFile path (CLIO arms)              default ./gsbench_bdev.dat
 *   GSBENCH_DISK_DIR     raw-arm output dir                  default ./gsbench_raw_out
 *
 * Each arm prints ONE machine-parseable RESULT line (ms, MB, MB/s, checksum). A wrapper
 * runs all three processes and builds the relative table; the shared checksum proves all
 * three computed identical bytes. Cases HIDDEN ([.]); CLIO arms RUN AT num_threads=1.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <clio_runtime/singletons.h>
#include <clio_ctp/util/gpu_api.h>

#include <clio_cte/kvhdf5/layout.h>           // Layout
#include <clio_cte/kvhdf5/gpu_cte_dataset.h>
#include <clio_cte/kvhdf5/tag_path.h>         // CanonicalTag

#include <cuda_runtime.h>
#include <cooperative_groups.h>   // Option A: grid.sync() across a resident cooperative grid

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if !CTP_IS_DEVICE_PASS
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/types.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <catch2/catch_test_macros.hpp>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#ifndef O_DIRECT
#define O_DIRECT 040000  // Linux x86/arm value; fallback if _GNU_SOURCE didn't expose it
#endif
#if GSBENCH_HAVE_HDF5
#include <hdf5.h>
#endif
#endif

using kvhdf5::byte_t;

namespace {
struct GsParams { float Du, Dv, F, k, dt; };
}  // namespace

// ---- shared computation: identical for every arm ---------------------------

// One Gray-Scott step, one thread per cell, periodic BCs. Pure CUDA.
__global__ void GsStepKernel(const float* u, const float* v, float* un, float* vn,
                             GsParams p, unsigned N) {
    unsigned cells = N * N;
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= cells) return;
    unsigned x = gid % N, y = gid / N;
    unsigned xm = (x == 0) ? (N - 1) : (x - 1);
    unsigned xp = (x == N - 1) ? 0u : (x + 1);
    unsigned ym = (y == 0) ? (N - 1) : (y - 1);
    unsigned yp = (y == N - 1) ? 0u : (y + 1);
    float uc = u[gid], vc = v[gid];
    float lap_u = u[y*N+xm] + u[y*N+xp] + u[ym*N+x] + u[yp*N+x] - 4.f*uc;
    float lap_v = v[y*N+xm] + v[y*N+xp] + v[ym*N+x] + v[yp*N+x] - 4.f*vc;
    float uvv = uc * vc * vc;
    un[gid] = uc + p.dt * (p.Du * lap_u - uvv + p.F * (1.f - uc));
    vn[gid] = vc + p.dt * (p.Dv * lap_v + uvv - (p.F + p.k) * vc);
}

// ---- CLIO snapshot kernels (device-facing handle) --------------------------

// Bulk stage a snapshot into its registered per-chunk device backends with a FULL grid
// (gridDim.y = chunk, gridDim.x = blocks/chunk), saturating HBM. This is split OUT of the
// submit kernels: the old design fused the copy into the single <<<1,256>>> CLIO-producer
// block, so 156 MB moved at ~1 GB/s (~160 ms/snapshot) and dominated BOTH arms — dwarfing
// the actual compute and masking any async advantage. `src` is the flat masked grid; chunk
// c's bytes are src[c*size .. c*size+size). Word-wise (float-grid sizes are 4-byte
// multiples). Being its OWN completed kernel gives kernel-boundary ordering, so the staged
// writes are visible to the subsequent submit kernel and the server's readback — no
// __threadfence_system needed (that fence was only for the old same-kernel copy+enqueue).
__global__ void TwCopyKernel(kvhdf5::GpuDatasetHandle h, const byte_t* src) {
    uint32_t c = blockIdx.y;
    uint64_t n = h.Size(c);
    const uint32_t* s = reinterpret_cast<const uint32_t*>(src + uint64_t(c) * n);
    uint32_t* d = reinterpret_cast<uint32_t*>(h.Data(c));
    uint64_t words = n >> 2;
    uint64_t gid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    for (uint64_t i = gid; i < words; i += stride) d[i] = s[i];
}

// SYNC submit: per chunk, fused Write(c) = submit-AND-WAIT (GPU blocks). Data already
// staged by TwCopyKernel. GRID-STRIDE over chunks: block b submits chunks b, b+gridDim,
// ... (each block's thread-0 enqueues via its own per-block IpcManager — the framework's
// documented one-block-per-chunk pattern). gridDim.x == 1 reproduces the old single-block
// serial loop byte-for-byte; gridDim.x == Count() gives one CUDA block per chunk. This is
// the "number of GPU blocks" axis. __syncthreads is per-block, so unequal per-block trip
// counts (Count() not divisible by gridDim.x) are safe.
template <bool kProbing>
__global__ __launch_bounds__(256) void TwSnapSyncKernel(kvhdf5::GpuDatasetHandle h) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        h.Write<kProbing>(c);
        __syncthreads();
    }
}

// ASYNC FIRE only: fire every chunk's PutBlob, no wait. Data already staged by
// TwCopyKernel. Drained later so the puts run on the server WHILE the subsequent sim
// steps run on the GPU. Grid-stride over chunks (see TwSnapSyncKernel).
template <bool kProbing>
__global__ __launch_bounds__(256) void TwSnapFireKernel(kvhdf5::GpuDatasetHandle h) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        h.WriteAsync<kProbing>(c);
        __syncthreads();
    }
}

// Explicitly instantiate both submit-kernel variants.
template __global__ void TwSnapSyncKernel<false>(kvhdf5::GpuDatasetHandle);
template __global__ void TwSnapSyncKernel<true>(kvhdf5::GpuDatasetHandle);
template __global__ void TwSnapFireKernel<false>(kvhdf5::GpuDatasetHandle);
template __global__ void TwSnapFireKernel<true>(kvhdf5::GpuDatasetHandle);

// ---- READER: GPU-initiated read-back + PDF -------------------------------------------- Th...
constexpr unsigned kHistBins = 256;

// GPU-initiated GetBlob for every chunk, grid-stride. Thread-0 submits AND waits (the
// handle's internal producer guard); __syncthreads keeps the block together per chunk.
__global__ __launch_bounds__(256) void TwReadKernel(kvhdf5::GpuDatasetHandle h) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        h.Read(c);        // thread-0 only (internal guard)
        __syncthreads();
    }
}

// ASYNC read: device-stamp this snapshot's tag (SetTag covers the GET slot too), then FIRE t...
__global__ __launch_bounds__(256) void TwReadFireKernel(kvhdf5::GpuDatasetHandle h,
                                                        const clio::cte::core::TagId* tags,
                                                        unsigned si) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        h.SetTag(c, tags[si]);
        __syncthreads();
        h.ReadAsync(c);   // thread-0 only (internal guard); no wait
        __syncthreads();
    }
}

// Drain a fired snapshot's outstanding gets (the read-side TwDrainKernel).
__global__ __launch_bounds__(32) void TwReadDrainKernel(kvhdf5::GpuDatasetHandle h) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        h.ReadWait(c);
        __syncthreads();
    }
}

// Bin one chunk of floats into the global histogram.
__global__ __launch_bounds__(256) void HistKernel(const float* __restrict__ data, uint64_t n,
                                                  unsigned long long* __restrict__ hist) {
    __shared__ unsigned long long smem[kHistBins];
    for (unsigned i = threadIdx.x; i < kHistBins; i += blockDim.x) smem[i] = 0ULL;
    __syncthreads();
    const uint64_t gid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    for (uint64_t i = gid; i < n; i += stride) {
        int b = int(data[i] * float(kHistBins));
        b = (b < 0) ? 0 : ((b >= int(kHistBins)) ? int(kHistBins) - 1 : b);
        atomicAdd(&smem[b], 1ULL);
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < kHistBins; i += blockDim.x)
        if (smem[i]) atomicAdd(&hist[i], smem[i]);
}

// Drain a fired snapshot's outstanding puts. Grid-stride over chunks (see TwSnapSyncKernel).
__global__ __launch_bounds__(32) void TwDrainKernel(kvhdf5::GpuDatasetHandle h) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        h.WriteWait(c);
        __syncthreads();
    }
}

// REUSE arm's fire kernel: like TwSnapFireKernel (async fire, no wait), but first device-sta...
__global__ __launch_bounds__(256) void ReuseFireKernel(
    kvhdf5::GpuDatasetHandle h, const clio::cte::core::TagId* tag_table, uint32_t snap) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        h.SetTag(c, tag_table[snap]);
        h.WriteAsync</*Probing=*/false>(c);
        __syncthreads();
    }
}

// ---- OPTION A: the PERSISTENT (resident, cooperative) producer kernel -------- ONE...
namespace cg = cooperative_groups;
// Async == true: fire the snapshot's Puts and defer draining to the next reuse / the tail (the...
template <bool Async>
__global__ void GsPersistentKernel(
    float* u0, float* v0, float* u1, float* v1,
    const kvhdf5::GpuDatasetHandle* groups, unsigned ngroups,
    const clio::cte::core::TagId* tag_table, const uint32_t* mask,
    GsParams p, unsigned N, unsigned num_snaps, unsigned steps_per) {
    CLIO_GPU_INIT(groups[0].info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    cg::grid_group grid = cg::this_grid();
    const unsigned cells = N * N;
    const unsigned gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned gstride = gridDim.x * blockDim.x;

    float* uc = u0; float* vc = v0; float* un = u1; float* vn = v1;
    for (unsigned s = 0; s < num_snaps; ++s) {
        // COMPUTE steps_per Gray-Scott steps in-kernel; grid.sync() is the grid-wide barrier between...
        for (unsigned t = 0; t < steps_per; ++t) {
            for (unsigned gid = gtid; gid < cells; gid += gstride) {
                unsigned x = gid % N, y = gid / N;
                unsigned xm = (x == 0) ? (N - 1) : (x - 1);
                unsigned xp = (x == N - 1) ? 0u : (x + 1);
                unsigned ym = (y == 0) ? (N - 1) : (y - 1);
                unsigned yp = (y == N - 1) ? 0u : (y + 1);
                float ucv = uc[gid], vcv = vc[gid];
                float lap_u = uc[y*N+xm] + uc[y*N+xp] + uc[ym*N+x] + uc[yp*N+x] - 4.f*ucv;
                float lap_v = vc[y*N+xm] + vc[y*N+xp] + vc[ym*N+x] + vc[yp*N+x] - 4.f*vcv;
                float uvv = ucv * vcv * vcv;
                un[gid] = ucv + p.dt * (p.Du * lap_u - uvv + p.F * (1.f - ucv));
                vn[gid] = vcv + p.dt * (p.Dv * lap_v + uvv - (p.F + p.k) * vcv);
            }
            grid.sync();
            float* tu = uc; uc = un; un = tu;
            float* tv = vc; vc = vn; vn = tv;
        }

        // SUBMIT: vc now holds snapshot s.
        const kvhdf5::GpuDatasetHandle h = groups[s % ngroups];
        const unsigned chunk_cells = cells / h.Count();
        const uint32_t* vsrc = reinterpret_cast<const uint32_t*>(vc);
        for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
            // ACQUIRE this chunk's buffer back from snapshot s-G. Thread-0 only, so the
            // __syncthreads() below is what actually protects the fill.
            if (s >= ngroups) h.WriteWait(c);
            __syncthreads();   // <-- LOAD-BEARING: no thread may fill until the drain is done
            uint32_t* dst = reinterpret_cast<uint32_t*>(h.Data(c));
            for (unsigned i = threadIdx.x; i < chunk_cells; i += blockDim.x) {
                unsigned gid = c * chunk_cells + i;
                uint32_t val = vsrc[gid];
                if (mask) val ^= mask[gid];
                dst[i] = val;
            }
            __threadfence_system();
            __syncthreads();
            h.SetTag(c, tag_table[s]);
            if (Async) h.WriteAsync</*Probing=*/false>(c);   // fire, drain later (thread-0)
            else       h.Write</*Probing=*/false>(c);        // fire-AND-WAIT (thread-0)
            __syncthreads();
        }
        grid.sync();   // every block's fire done before the next snapshot's compute reuses vc
    }
    // TAIL: drain every group still in flight (already complete under Async==false).
    const unsigned tail_groups = (num_snaps < ngroups) ? num_snaps : ngroups;
    for (unsigned gi = 0; gi < tail_groups; ++gi) {
        const kvhdf5::GpuDatasetHandle h = groups[gi];
        for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) h.WriteWait(c);
    }
}
template __global__ void GsPersistentKernel<true>(
    float*, float*, float*, float*, const kvhdf5::GpuDatasetHandle*, unsigned,
    const clio::cte::core::TagId*, const uint32_t*, GsParams, unsigned, unsigned, unsigned);
template __global__ void GsPersistentKernel<false>(
    float*, float*, float*, float*, const kvhdf5::GpuDatasetHandle*, unsigned,
    const clio::cte::core::TagId*, const uint32_t*, GsParams, unsigned, unsigned, unsigned);

// ---- COMPUTE-ONLY variant of the persistent kernel (register-decomposition probe) ------ S...
__global__ void GsPersistentComputeOnlyKernel(
    float* u0, float* v0, float* u1, float* v1,
    const kvhdf5::GpuDatasetHandle* groups, unsigned ngroups,
    const clio::cte::core::TagId* tag_table, const uint32_t* mask,
    GsParams p, unsigned N, unsigned num_snaps, unsigned steps_per) {
    (void)groups; (void)ngroups; (void)tag_table; (void)mask;
    cg::grid_group grid = cg::this_grid();
    const unsigned cells = N * N;
    const unsigned gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned gstride = gridDim.x * blockDim.x;

    float* uc = u0; float* vc = v0; float* un = u1; float* vn = v1;
    for (unsigned s = 0; s < num_snaps; ++s) {
        // COMPUTE steps_per Gray-Scott steps in-kernel; identical to GsPersistentKernel's
        // compute phase. No SUBMIT section here — that is the whole point of this probe.
        for (unsigned t = 0; t < steps_per; ++t) {
            for (unsigned gid = gtid; gid < cells; gid += gstride) {
                unsigned x = gid % N, y = gid / N;
                unsigned xm = (x == 0) ? (N - 1) : (x - 1);
                unsigned xp = (x == N - 1) ? 0u : (x + 1);
                unsigned ym = (y == 0) ? (N - 1) : (y - 1);
                unsigned yp = (y == N - 1) ? 0u : (y + 1);
                float ucv = uc[gid], vcv = vc[gid];
                float lap_u = uc[y*N+xm] + uc[y*N+xp] + uc[ym*N+x] + uc[yp*N+x] - 4.f*ucv;
                float lap_v = vc[y*N+xm] + vc[y*N+xp] + vc[ym*N+x] + vc[yp*N+x] - 4.f*vcv;
                float uvv = ucv * vcv * vcv;
                un[gid] = ucv + p.dt * (p.Du * lap_u - uvv + p.F * (1.f - ucv));
                vn[gid] = vcv + p.dt * (p.Dv * lap_v + uvv - (p.F + p.k) * vcv);
            }
            grid.sync();
            float* tu = uc; uc = un; un = tu;
            float* tv = vc; vc = vn; vn = tv;
        }
    }
}

// ---- POOLED arm: fill + fire fused into ONE kernel -------------------------- The sync/asy...
struct TwPooledFill {
    const byte_t* src;   // masked snapshot grid; chunk c is src[c*n .. c*n+n)
    // BLOCK-WIDE (every thread), word-wise, block-strided — the same copy TwCopyKernel does
    // for one chunk, minus the extra gridDim.x blocks. Uniform across the block.
    __device__ void operator()(uint32_t c, byte_t* dst, uint64_t n) const {
        const uint32_t* s = reinterpret_cast<const uint32_t*>(src + uint64_t(c) * n);
        uint32_t* d = reinterpret_cast<uint32_t*>(dst);
        const uint64_t words = n >> 2;
        for (uint64_t i = threadIdx.x; i < words; i += blockDim.x) d[i] = s[i];
    }
};

// NOTE (2026-07-20, measured via cudaFuncGetAttributes + cudaOccupancyMaxActiveBlocksPerMult...
template <bool kProbing>
__global__ __launch_bounds__(256) void TwSnapPooledKernel(kvhdf5::GpuDatasetHandle h,
                                                          const byte_t* src) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    h.WritePipelined<kProbing, /*TailDrain=*/false>(TwPooledFill{src});
}
template __global__ void TwSnapPooledKernel<false>(kvhdf5::GpuDatasetHandle, const byte_t*);
template __global__ void TwSnapPooledKernel<true>(kvhdf5::GpuDatasetHandle, const byte_t*);

// XOR the snapshot with a fixed random mask (word-wise) into a scratch buffer, so the
// PERSISTED bytes are high-entropy / incompressible — otherwise the Gray-Scott field is
// mostly zeros and a compressing filesystem (e.g. btrfs zstd) makes the "disk I/O" nearly
// free, voiding any disk comparison. Deterministic (fixed mask) => byte-identical across
// arms => checksums still match.
__global__ void MaskKernel(uint32_t* dst, const uint32_t* src, const uint32_t* mask,
                           unsigned words) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < words) dst[i] = src[i] ^ mask[i];
}

// ---- Submit-path hop breakdown (GSBENCH_SUBMIT_PROBE=1) --------------------- Splits one G...

// Device sees the host's flag write, stamps immediately. Bounds the offset from below.
__global__ void ProbePingA(volatile unsigned* flag, unsigned long long* out) {
    while (*flag == 0u) {}
    *out = clio::run::gpu::ProbeNowNs();
}

// Device stamps, then publishes to pinned memory the host is spinning on.
__global__ void ProbePingB(volatile unsigned long long* out) {
    unsigned long long t = clio::run::gpu::ProbeNowNs();
    *out = t;
}

// Characterizes the two device clocks against each other.
__global__ void ProbeClockCalKernel(unsigned long long* out) {
    const unsigned long long t0 = clio::run::gpu::ProbeNowNs();
    const unsigned long long c0 = clio::run::gpu::ProbeCycles();
    unsigned long long prev = t0, mn = ~0ull, t = t0;
    while (t - t0 < 20000000ull) {          // 20 ms
        t = clio::run::gpu::ProbeNowNs();
        const unsigned long long d = t - prev;
        if (d > 0 && d < mn) mn = d;
        prev = t;
    }
    const unsigned long long c1 = clio::run::gpu::ProbeCycles();
    out[0] = mn;
    out[1] = t - t0;
    out[2] = c1 - c0;
}

#if !CTP_IS_DEVICE_PASS

namespace {

constexpr GsParams kGs{0.16f, 0.08f, 0.055f, 0.062f, 1.0f};

// ---- env-var config --------------------------------------------------------

unsigned EnvU(const char* k, unsigned dflt) {
    const char* v = std::getenv(k);
    if (!v || !*v) return dflt;
    long x = std::strtol(v, nullptr, 10);
    return x > 0 ? static_cast<unsigned>(x) : dflt;
}
// Like EnvU but 0 is a MEANINGFUL value (EnvU coerces <=0 to the default).
unsigned EnvU0(const char* k, unsigned dflt) {
    const char* v = std::getenv(k);
    if (!v || !*v) return dflt;
    long x = std::strtol(v, nullptr, 10);
    return x >= 0 ? static_cast<unsigned>(x) : dflt;
}
std::string EnvS(const char* k, const char* dflt) {
    const char* v = std::getenv(k);
    return (v && *v) ? std::string(v) : std::string(dflt);
}

struct Cfg {
    unsigned N       = EnvU("GSBENCH_N", 512);
    unsigned chunks  = EnvU("GSBENCH_CHUNKS", 4);
    unsigned snaps   = EnvU("GSBENCH_SNAPS", 4);
    unsigned steps_per = EnvU("GSBENCH_STEPS_PER", 8);
    std::string bdev = EnvS("GSBENCH_BDEV", "ram");
    unsigned cap_mb  = EnvU("GSBENCH_BDEV_CAP_MB", 512);
    std::string bdev_path = EnvS("GSBENCH_BDEV_PATH", "./gsbench_bdev.dat");
    std::string disk_dir  = EnvS("GSBENCH_DISK_DIR", "./gsbench_raw_out");
    // Raw arm O_DIRECT (cache-bypass) for the disk comparison; set 0 for buffered writes
    // when the raw target is a RAM tier (tmpfs, which rejects O_DIRECT) — the fair
    // software-path comparison vs CLIO's kRam bdev.
    unsigned raw_odirect  = EnvU("GSBENCH_RAW_ODIRECT", 1);
    // Raw arm fsync per snapshot (also gates the hdf5 arms' per-snapshot H5Fflush+fdatasync): DU...
    unsigned raw_fsync    = EnvU0("GSBENCH_RAW_FSYNC", 1);
    // DURABILITY PARITY FOR THE CLIO ARMS.
    unsigned clio_fsync = EnvU0("GSBENCH_CLIO_FSYNC", 1);
    // PER-SNAPSHOT durability for the CLIO arms (checkpoint semantics): drain THIS snapshot's Pu...
    unsigned clio_persnap = EnvU0("GSBENCH_CLIO_PERSNAP", 0);
    // cudaHostRegister the hostclio arm's shm staging buffer.
    unsigned hostclio_pin = EnvU0("GSBENCH_HOSTCLIO_PIN", 1);
    // XOR each snapshot with a random mask so persisted bytes are incompressible (else the
    // mostly-zero Gray-Scott field compresses away on btrfs zstd, voiding disk comparisons).
    unsigned incompressible = EnvU("GSBENCH_INCOMPRESSIBLE", 1);
    // Raw writer structure: 0 = background thread (I/O overlaps compute); 1 = inline
    // synchronous (GPU idle during the write, matching host-CLIO / sync-CLIO). Use inline
    // for a storage-path comparison free of this box's GPU-concurrent-I/O throttle.
    unsigned raw_inline   = EnvU("GSBENCH_RAW_INLINE", 0);
    // Place the CLIO snapshot DATA backend in pinned host memory (kPinnedHost)
    // instead of on-GPU (kDeviceMem). The in-process bdev server's device->host
    // readback does not overlap the producer's compute (its first per-run D2H
    // stalls for the whole compute window), so with kDeviceMem the async drain
    // serializes after compute and barely beats sync. Pinned data removes the
    // server D2H, letting disk writes pipeline under compute: at steps_per=192,
    // N=6400 async goes ~836 -> ~955 MB/s. Trade-off: the producer kernel writes
    // mapped host over PCIe (slower submit) and it regresses the disk-bound /
    // low-compute regime, so it is off by default.
    unsigned data_pinned  = EnvU("GSBENCH_DATA_PINNED", 0);
    // READER benchmark (GSBENCH_READ=1, default off so the writer numbers are untouched): after ...
    unsigned read_pdf     = EnvU0("GSBENCH_READ", 0);
    // GSBENCH_READ_ASYNC=1: use the PIPELINED reader (2 reused read buffers, snapshot s+1's Gets...
    unsigned read_async   = EnvU0("GSBENCH_READ_ASYNC", 0);
    // Number of CUDA thread blocks that submit PutBlob tasks (grid dim of the fire/sync/drain ke...
    unsigned submit_blocks = EnvU("GSBENCH_SUBMIT_BLOCKS", 1);
    // POOLED arm only: M, the number of RESIDENT per-chunk data buffers a snapshot's dataset holds.
    unsigned pool = EnvU0("GSBENCH_POOL", 0);
    // Force the CLIO kernels' device code resident BEFORE the timed region (cudaFuncGetAttributes).
    unsigned prewarm = EnvU0("GSBENCH_PREWARM", 1);

    // ---- hdf5 arm ---------------------------------------------------------- Output dir for th...
    std::string hdf5_dir = EnvS("GSBENCH_HDF5_DIR", "./gsbench_hdf5_out");
    // Leave EVERY tuning knob alone (stock HDF5: 1 MB chunk cache, late alloc, default fill).
    unsigned hdf5_stock = EnvU0("GSBENCH_HDF5_STOCK", 0);
    // Per-dataset chunk cache (H5Pset_chunk_cache), in MB.
    unsigned hdf5_rdcc_mb = EnvU0("GSBENCH_HDF5_RDCC_MB", 0);
    // H5Pset_alloc_time(H5D_ALLOC_TIME_EARLY): allocate the dataset's file space up front
    // instead of chunk-by-chunk during the write.
    unsigned hdf5_early_alloc = EnvU0("GSBENCH_HDF5_EARLY_ALLOC", 1);
    // Use H5Dwrite_chunk() (direct chunk write: skips the cache AND the filter pipeline) instead...
    unsigned hdf5_direct_chunk = EnvU0("GSBENCH_HDF5_DIRECT_CHUNK", 0);
    // The "typical non-expert user" baseline (arm `hdf5_naive`): a default HDF5 setup with NO op...
    unsigned hdf5_naive = EnvU0("GSBENCH_HDF5_NAIVE", 0);
    // Layout for the naive arm: 0 = CONTIGUOUS (no H5Pset_chunk — the most default thing a naive...
    unsigned hdf5_naive_chunked = EnvU0("GSBENCH_HDF5_NAIVE_CHUNKED", 0);
    // Create all `snaps` datasets BEFORE the timed region rather than one per snapshot inside it.
    unsigned hdf5_precreate = EnvU0("GSBENCH_HDF5_PRECREATE", 0);
    // File driver: "sec2" (default; unbuffered pwrite, the same kernel path the raw arm takes) o...
    std::string hdf5_vfd = EnvS("GSBENCH_HDF5_VFD", "sec2");
    // Pinned host buffers in the threaded `hdf5` arm's writer pool (raw uses 3).
    unsigned hdf5_nbuf = EnvU("GSBENCH_HDF5_NBUF", 3);
    // Pinned (cudaMallocHost) vs pageable (malloc) staging buffers for the hdf5 arm.
    unsigned hdf5_pinned = EnvU0("GSBENCH_HDF5_PINNED", 1);
    // H5Pset_meta_block_size: how much file space HDF5 grabs at a time for metadata.
    unsigned hdf5_meta_block_kb = EnvU0("GSBENCH_HDF5_META_BLOCK_KB", 2048);
    // PAGE-FAULT PARITY between the file-backed arms (raw/hdf5) and the CLIO arms.
    unsigned prefault = EnvU0("GSBENCH_PREFAULT", 1);
    // Effective submit grid: clamp the knob to chunks (more blocks than chunks idle).
    unsigned submit_grid() const {
        return submit_blocks < chunks ? submit_blocks : chunks;
    }

    unsigned steps() const { return snaps * steps_per; }
    uint64_t cells() const { return uint64_t(N) * N; }
    uint64_t grid_bytes() const { return cells() * sizeof(float); }
    // total bytes persisted by the whole run (one v-grid per snapshot).
    uint64_t total_bytes() const { return grid_bytes() * snaps; }
};

// ---- shared sim scaffolding ------------------------------------------------

struct Grids { float *u_curr, *u_next, *v_curr, *v_next; };
Grids MakeGrids(unsigned N) {
    uint64_t cells = uint64_t(N) * N, bytes = cells * sizeof(float);
    Grids g{};
    REQUIRE(cudaMalloc(&g.u_curr, bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&g.u_next, bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&g.v_curr, bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&g.v_next, bytes) == cudaSuccess);
    std::vector<float> u0(cells, 1.0f), v0(cells, 0.0f);
    unsigned lo = N/2 - 3, hi = N/2 + 3;
    for (unsigned y = lo; y < hi; ++y)
        for (unsigned x = lo; x < hi; ++x) v0[uint64_t(y)*N + x] = 1.0f;
    ctp::GpuApi::Memcpy(g.u_curr, u0.data(), bytes);
    ctp::GpuApi::Memcpy(g.v_curr, v0.data(), bytes);
    return g;
}
void FreeGrids(Grids& g) {
    cudaFree(g.u_curr); cudaFree(g.u_next); cudaFree(g.v_curr); cudaFree(g.v_next);
}

// Makes each snapshot INCOMPRESSIBLE: Apply(v) XORs the grid with a fixed random mask into
// a scratch device buffer and returns it, so the persisted bytes are high-entropy (the
// mostly-zero Gray-Scott field would otherwise compress away on btrfs zstd). Mask is fixed
// (seed) => byte-identical across arms => checksums still match. If disabled, Apply is a
// pass-through. One shared scratch is safe: the MaskKernel and each arm's subsequent
// copy/D2H are serialized on the default stream, so scratch is consumed before the next
// Apply overwrites it.
struct Masker {
    bool on_;
    unsigned cells_;
    uint32_t* d_mask_ = nullptr;
    uint32_t* d_scratch_ = nullptr;
    Masker(unsigned N, bool on) : on_(on), cells_(N * N) {
        if (!on_) return;
        uint64_t bytes = uint64_t(cells_) * sizeof(uint32_t);
        REQUIRE(cudaMalloc(&d_mask_, bytes) == cudaSuccess);
        REQUIRE(cudaMalloc(&d_scratch_, bytes) == cudaSuccess);
        std::vector<uint32_t> mask(cells_);
        std::mt19937 rng(0xC0FFEEu);            // fixed => identical across arms
        for (auto& x : mask) x = rng();
        ctp::GpuApi::Memcpy(d_mask_, mask.data(), bytes);
    }
    ~Masker() { if (on_) { cudaFree(d_mask_); cudaFree(d_scratch_); } }
    // Returns an incompressible view of v (scratch), or v itself if disabled.
    float* Apply(float* v) {
        if (!on_) return v;
        unsigned t = 256, b = (cells_ + t - 1) / t;
        MaskKernel<<<b, t>>>(d_scratch_, reinterpret_cast<uint32_t*>(v), d_mask_, cells_);
        return reinterpret_cast<float*>(d_scratch_);
    }
    // Raw device mask pointer for arms that XOR inline (the persistent kernel), or nullptr when
    // incompressibility is off. Same fixed-seed mask Apply() uses, so persisted bytes match.
    const uint32_t* MaskPtr() const { return on_ ? d_mask_ : nullptr; }
};

// FNV-1a over a host byte buffer (cross-arm "identical computation" proof).
uint64_t Fnv1a(const void* data, size_t n, uint64_t h = 1469598103934665603ull) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

// GPU-timeline phase tracer (GSBENCH_TRACE=1), used by the sync & async CLIO arms to
// answer the advisor's question: is async's I/O actually OVERLAPPING compute, or is
// compute so cheap there's nothing to overlap? It splits each snapshot interval on the
// GPU's own timeline into:
//   compute_ms  — the steps_per GsStepKernels leading up to the snapshot,
//   submit_ms   — the snapshot's mask + PutBlob-task emplace kernel (async: fire only;
//                 sync: fire-AND-device-wait, so sync's submit bucket carries the wait),
//   drain_ms    — (async only) the single TAIL TwDrainKernel spin-wait at the very end.
// So the hypothesis reads directly off the buckets: async should have near-zero submit_ms
// and a drain_ms that is SMALL vs total compute if I/O kept up (good overlap), or LARGE if
// the server fell behind (backlog => async collapses toward sync). sync has no drain; its
// per-snapshot wait is inside submit_ms.
//
// Mechanics: everything runs on the default stream, so cudaEvents recorded at the phase
// boundaries measure GPU-serialized work in order. We ONLY cudaEventRecord in the hot loop
// (async, ~1us, no stall) and read every cudaEventElapsedTime AFTER the closing Synchronize
// — a mid-loop cudaEventSynchronize would serialize the stream and destroy the overlap we
// are measuring. NOTE: submit_ms includes the per-snapshot MaskKernel (identical work in
// both arms, so it cancels in the sync-vs-async comparison).
struct PhaseTrace {
    bool on_;
    unsigned snaps_;
    bool has_drain_;
    std::vector<cudaEvent_t> cs_, ce_, se_;   // compute-start, compute-end(=submit-start), submit-end
    cudaEvent_t ds_ = nullptr, de_ = nullptr; // drain-start, drain-end (async only)
    PhaseTrace(bool on, unsigned snaps, bool async)
        : on_(on), snaps_(snaps), has_drain_(async) {
        if (!on_) return;
        cs_.resize(snaps); ce_.resize(snaps); se_.resize(snaps);
        for (unsigned i = 0; i < snaps; ++i) {
            cudaEventCreate(&cs_[i]); cudaEventCreate(&ce_[i]); cudaEventCreate(&se_[i]);
        }
        if (has_drain_) { cudaEventCreate(&ds_); cudaEventCreate(&de_); }
    }
    ~PhaseTrace() {
        if (!on_) return;
        for (auto e : cs_) cudaEventDestroy(e);
        for (auto e : ce_) cudaEventDestroy(e);
        for (auto e : se_) cudaEventDestroy(e);
        if (has_drain_) { cudaEventDestroy(ds_); cudaEventDestroy(de_); }
    }
    void CompStart(unsigned si)  { if (on_) cudaEventRecord(cs_[si]); }
    void CompEnd(unsigned si)    { if (on_) cudaEventRecord(ce_[si]); }
    void SubmitEnd(unsigned si)  { if (on_) cudaEventRecord(se_[si]); }
    void DrainStart()            { if (on_ && has_drain_) cudaEventRecord(ds_); }
    void DrainEnd()              { if (on_ && has_drain_) cudaEventRecord(de_); }
    // Read the deltas + print the per-snapshot series and totals. MUST be called after the
    // caller's RunSim returns (i.e. after the closing Synchronize) so every event is done.
    void Report(const char* arm) {
        if (!on_) return;
        double tot_c = 0, tot_s = 0;
        std::fprintf(stderr, "GSBENCH_TRACE arm=%s per-snapshot (GPU-timeline ms):\n", arm);
        for (unsigned i = 0; i < snaps_; ++i) {
            float c = 0, s = 0;
            cudaEventElapsedTime(&c, cs_[i], ce_[i]);
            cudaEventElapsedTime(&s, ce_[i], se_[i]);
            tot_c += c; tot_s += s;
            std::fprintf(stderr, "GSBENCH_TRACE   snap=%2u compute_ms=%8.3f submit_ms=%8.3f\n",
                         i, c, s);
        }
        float drain = 0;
        if (has_drain_) cudaEventElapsedTime(&drain, ds_, de_);
        std::fprintf(stderr,
            "GSBENCH_TRACE arm=%s TOTALS compute_ms=%.3f submit_ms=%.3f drain_ms=%.3f "
            "(sum=%.3f)\n",
            arm, tot_c, tot_s, double(drain), tot_c + tot_s + double(drain));
    }
};

// Direct measurement of the PRODUCER-BLOCKING device-to-host copy (GSBENCH_D2H_TRACE=1), i.e.
struct D2HTrace {
    bool on_;
    uint64_t bytes_;                       // bytes moved per copy (constant across snapshots)
    std::vector<cudaEvent_t> a_, b_;       // copy-start / copy-end, one pair per snapshot
    std::vector<double> block_ms_;         // host-observed block, one per snapshot
    unsigned i_ = 0;                       // next slot

    D2HTrace(bool on, unsigned snaps, uint64_t bytes) : on_(on), bytes_(bytes) {
        if (!on_) return;
        a_.resize(snaps); b_.resize(snaps); block_ms_.assign(snaps, 0.0);
        for (unsigned i = 0; i < snaps; ++i) {
            cudaEventCreate(&a_[i]); cudaEventCreate(&b_[i]);
        }
    }
    ~D2HTrace() {
        if (!on_) return;
        for (auto e : a_) cudaEventDestroy(e);
        for (auto e : b_) cudaEventDestroy(e);
    }

    // Drop-in for `cudaMemcpy(dst, src, bytes_, cudaMemcpyDeviceToHost)`. Untraced when off.
    void Copy(void* dst, const void* src) {
        if (!on_ || i_ >= a_.size()) {
            cudaMemcpy(dst, src, bytes_, cudaMemcpyDeviceToHost);
            return;
        }
        const unsigned i = i_++;
        auto h0 = std::chrono::steady_clock::now();
        cudaEventRecord(a_[i]);
        cudaMemcpy(dst, src, bytes_, cudaMemcpyDeviceToHost);
        cudaEventRecord(b_[i]);
        block_ms_[i] = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - h0).count();
    }

    // MUST be called after RunSim returns (i.e. after the closing Synchronize).
    void Report(const char* arm) {
        if (!on_ || i_ == 0) return;
        double tot_copy = 0, tot_block = 0;
        std::fprintf(stderr, "GSBENCH_D2H arm=%s per-snapshot:\n", arm);
        for (unsigned i = 0; i < i_; ++i) {
            float c = 0;
            cudaEventElapsedTime(&c, a_[i], b_[i]);
            tot_copy += c; tot_block += block_ms_[i];
            std::fprintf(stderr,
                "GSBENCH_D2H   snap=%2u copy_ms=%8.3f block_ms=%8.3f GBps=%6.2f\n",
                i, c, block_ms_[i],
                double(bytes_) / (double(c) / 1e3) / 1e9);
        }
        const double mean_copy = tot_copy / double(i_);
        std::fprintf(stderr,
            "GSBENCH_D2H arm=%s copies=%u bytes_per_copy=%llu total_MB=%.1f "
            "copy_ms=%.3f block_ms=%.3f mean_copy_ms=%.3f GBps=%.2f\n",
            arm, i_, (unsigned long long)bytes_,
            double(bytes_) * double(i_) / (1024.0 * 1024.0),
            tot_copy, tot_block, mean_copy,
            double(bytes_) / (mean_copy / 1e3) / 1e9);
    }
};

// host_ns ~= device_globaltimer_ns + offset.
struct ProbeClockOffset {
    long long est_ns = 0;   // midpoint of the bracket
    long long err_ns = 0;   // half-width: the honest error bar on hops 3 and 8
    long long lo_ns = 0;    // raw bounds, kept so the analysis can see the inversion
    long long hi_ns = 0;
    long long at_host_ns = 0;  // host clock when this anchor was taken (for the lerp)

    static ProbeClockOffset Measure(unsigned reps = 200) {
        ProbeClockOffset o;
        unsigned* flag = nullptr;
        unsigned long long* stamp = nullptr;
        cudaHostAlloc(&flag, sizeof(unsigned), cudaHostAllocMapped);
        cudaHostAlloc(&stamp, sizeof(unsigned long long), cudaHostAllocMapped);

        long long lo = LLONG_MIN;   // max over A: offset >= h - d
        long long hi = LLONG_MAX;   // min over B: offset <= h - d
        for (unsigned i = 0; i < reps; ++i) {
            // --- A: host -> device
            *flag = 0; *stamp = 0;
            ProbePingA<<<1, 1>>>(flag, stamp);
            // Let the kernel reach its spin before releasing it, so the kernel-launch
            // latency is not charged to the one-way trip we are bounding.
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            long long h = (long long)clio::run::gpu::SubmitProbe::NowNs();
            *flag = 1;
            cudaDeviceSynchronize();
            long long d = (long long)*stamp;
            if (d > 0) lo = std::max(lo, h - d);

            // --- B: device -> host
            *stamp = 0;
            ProbePingB<<<1, 1>>>(stamp);
            unsigned long long seen = 0;
            while ((seen = *(volatile unsigned long long*)stamp) == 0ull) {}
            long long h2 = (long long)clio::run::gpu::SubmitProbe::NowNs();
            cudaDeviceSynchronize();
            hi = std::min(hi, h2 - (long long)seen);
        }
        cudaFreeHost(flag);
        cudaFreeHost(stamp);
        o.lo_ns = lo;
        o.hi_ns = hi;
        o.est_ns = (lo + hi) / 2;
        o.err_ns = (hi - lo) / 2;
        o.at_host_ns = (long long)clio::run::gpu::SubmitProbe::NowNs();
        return o;
    }
};

struct SubmitProbeHarness {
    bool on_ = false;
    unsigned cap_ = 0;
    clio::run::gpu::SubmitProbeRec* d_recs_ = nullptr;
    unsigned* d_counter_ = nullptr;
    ProbeClockOffset off_;             // anchor 1: taken at arm time
    ProbeClockOffset off_end_;         // anchor 2: taken at dump time (drift correction)
    unsigned long long tick_ns_ = 0;   // %globaltimer resolution (the fine-hop floor)
    double sm_ghz_ = 0.0;              // SM clock, for the cycles->ns cross-check

    // Arms both halves and hands the device half to the kernel via gpu_info. MUST run
    // before the datasets are built — they snapshot gpu_info by value into every handle.
    SubmitProbeHarness(bool on, unsigned cap, clio::run::IpcManagerGpuInfo* gpu_info)
        : on_(on), cap_(cap) {
        if (!on_) return;
        cudaMalloc(&d_recs_, size_t(cap_) * sizeof(clio::run::gpu::SubmitProbeRec));
        cudaMemset(d_recs_, 0, size_t(cap_) * sizeof(clio::run::gpu::SubmitProbeRec));
        cudaMalloc(&d_counter_, sizeof(unsigned));
        cudaMemset(d_counter_, 0, sizeof(unsigned));
        gpu_info->probe_.recs = d_recs_;
        gpu_info->probe_.counter = d_counter_;
        gpu_info->probe_.cap = cap_;
        clio::run::gpu::SubmitProbe::Get().Enable(cap_);
        off_ = ProbeClockOffset::Measure();

        unsigned long long* cal = nullptr;
        cudaMalloc(&cal, 3 * sizeof(unsigned long long));
        ProbeClockCalKernel<<<1, 1>>>(cal);
        cudaDeviceSynchronize();
        unsigned long long h[3] = {0, 0, 0};
        cudaMemcpy(h, cal, sizeof(h), cudaMemcpyDeviceToHost);
        cudaFree(cal);
        tick_ns_ = h[0];
        sm_ghz_ = h[1] ? double(h[2]) / double(h[1]) : 0.0;
        std::fprintf(stderr,
            "GSBENCH_PROBE clock_offset_ns=%lld err_ns=%lld (host = device + offset) "
            "globaltimer_tick_ns=%llu sm_ghz=%.3f\n",
            off_.est_ns, off_.err_ns, tick_ns_, sm_ghz_);
    }

    void Dump(const char* arm, const char* dir) {
        if (!on_) return;
        // Second clock anchor, taken as soon after the timed region as possible: the device<->host o...
        off_end_ = ProbeClockOffset::Measure();
        unsigned n = 0;
        cudaMemcpy(&n, d_counter_, sizeof(unsigned), cudaMemcpyDeviceToHost);
        if (n > cap_) {
            std::fprintf(stderr,
                "GSBENCH_PROBE WARNING arm=%s: %u submits but capacity %u — "
                "%u records DROPPED; raise the cap before trusting this run\n",
                arm, n, cap_, n - cap_);
            n = cap_;
        }
        std::vector<clio::run::gpu::SubmitProbeRec> recs(n);
        cudaMemcpy(recs.data(), d_recs_,
                   size_t(n) * sizeof(clio::run::gpu::SubmitProbeRec),
                   cudaMemcpyDeviceToHost);

        char path[512];
        std::snprintf(path, sizeof(path), "%s/probe_dev_%s.csv", dir, arm);
        if (FILE* f = std::fopen(path, "w")) {
            std::fprintf(f, "task_ptr,seq,d_enter,d_pushed,d_wait_begin,d_wait_end,"
                            "c_enter,c_fields,c_prefence,c_postfence,c_pushed,"
                            "c_wait_begin,c_wait_end\n");
            for (unsigned i = 0; i < n; ++i) {
                const auto& r = recs[i];
                std::fprintf(f,
                             "%llu,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
                             r.task_ptr, i, r.d_enter, r.d_pushed, r.d_wait_begin,
                             r.d_wait_end, r.c_enter, r.c_fields, r.c_prefence,
                             r.c_postfence, r.c_pushed, r.c_wait_begin, r.c_wait_end);
            }
            std::fclose(f);
        }
        std::snprintf(path, sizeof(path), "%s/probe_host_%s.csv", dir, arm);
        clio::run::gpu::SubmitProbe::Get().Dump(path);

        std::snprintf(path, sizeof(path), "%s/probe_meta_%s.csv", dir, arm);
        if (FILE* f = std::fopen(path, "w")) {
            std::fprintf(f, "arm,dev_records,host_records,clock_offset_ns,clock_err_ns,"
                            "clock_at_ns,clock_offset_end_ns,clock_err_end_ns,"
                            "clock_at_end_ns,globaltimer_tick_ns,sm_ghz\n");
            std::fprintf(f, "%s,%u,%u,%lld,%lld,%lld,%lld,%lld,%lld,%llu,%.4f\n", arm, n,
                         clio::run::gpu::SubmitProbe::Get().Count(), off_.est_ns,
                         off_.err_ns, off_.at_host_ns, off_end_.est_ns, off_end_.err_ns,
                         off_end_.at_host_ns, tick_ns_, sm_ghz_);
            std::fclose(f);
        }
        std::fprintf(stderr,
            "GSBENCH_PROBE arm=%s dev_records=%u host_records=%u -> %s/probe_*_%s.csv\n",
            arm, n, clio::run::gpu::SubmitProbe::Get().Count(), dir, arm);
    }

    ~SubmitProbeHarness() {
        if (!on_) return;
        cudaFree(d_recs_);
        cudaFree(d_counter_);
    }
};

// The ONE timed sim loop shared by every arm. `snap(si, v_curr)` persists snapshot si;
// `finalize()` runs inside the timed region (async drain). No per-step Synchronize so
// the stream can overlap; one Synchronize closes the region. Returns wall-clock ms and
// accumulates a checksum of every snapshot's v-grid (host-read) for cross-arm equality.
// `trace` (optional) records GPU-timeline phase events; nullptr = off.
// ---- Peak device-memory measurement (GSBENCH_MEM=1, default off) -----------
// A background thread polls cudaMemGetInfo during the timed loop and records the minimum
// free bytes seen. The reported peak is (baseline_free - min_free): device memory the arm
// brought resident ON TOP OF whatever was already allocated when the server came up. The
// baseline is captured at the END of BenchEnv's ctor (server up, bdev allocated) but BEFORE
// the arm allocates its compute grids and client buffer groups, so:
//   * other GPU apps' steady usage and the fixed bdev cancel out of the delta;
//   * what remains is THIS arm's device working set = compute grids (identical across arms)
//     + client buffer groups + any in-flight device staging.
// kRam/kFile bdevs and pinned-host data live in HOST memory, so pinned/persistent arms show
// a LOW device peak by design (their footprint is host-pinned, not device) -- that asymmetry
// is exactly the point of the measurement, not a bug.
std::atomic<bool> g_mem_run{false};
size_t g_mem_baseline_free = 0;   // set in BenchEnv ctor, before arm allocations
size_t g_mem_min_free = SIZE_MAX; // min free seen during the timed loop
double g_last_peak_device_mb = -1.0;
double g_last_peak_host_mb = -1.0;

// Deterministic I/O staging-buffer footprint: the bytes each arm holds resident for moving s...
uint64_t g_io_buf_bytes = 0;

// The ACTUAL data backend the arm ran on.
int g_actual_pinned = -1;
unsigned ReportedPinned(const Cfg& cfg) {
    return (g_actual_pinned >= 0) ? unsigned(g_actual_pinned) : cfg.data_pinned;
}

// Read one "Key: <n> kB" field from /proc/self/status (host RSS accounting). Returns kB, or 0.
long ReadProcStatusKB(const char* key) {
    FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    long kb = 0;
    const size_t klen = std::strlen(key);
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, key, klen) == 0) { kb = std::strtol(line + klen, nullptr, 10); break; }
    }
    std::fclose(f);
    return kb;
}
// Captured at program load (static init), i.e.
long g_host_baseline_kb = ReadProcStatusKB("VmRSS:");

struct MemSampler {
    std::thread th;
    void Start() {
        if (!EnvU0("GSBENCH_MEM", 0)) return;
        g_mem_min_free = SIZE_MAX;
        g_mem_run.store(true, std::memory_order_relaxed);
        th = std::thread([] {
            while (g_mem_run.load(std::memory_order_relaxed)) {
                size_t f = 0, t = 0;
                if (cudaMemGetInfo(&f, &t) == cudaSuccess && f < g_mem_min_free)
                    g_mem_min_free = f;
                std::this_thread::sleep_for(std::chrono::microseconds(500));
            }
        });
    }
    void Stop() {
        if (!th.joinable()) return;
        g_mem_run.store(false, std::memory_order_relaxed);
        th.join();
        // baseline_free >= min_free normally; clamp so noise/other-app frees can't go negative.
        const double peak_bytes = (g_mem_baseline_free > g_mem_min_free)
                                      ? double(g_mem_baseline_free - g_mem_min_free) : 0.0;
        g_last_peak_device_mb = peak_bytes / (1024.0 * 1024.0);
        // Peak HOST RSS above the program-load baseline (VmHWM = kernel high-water-mark, so no
        // sampling needed). Valid for EVERY arm -- gives the HDF5 arms a real number.
        const long hwm_kb = ReadProcStatusKB("VmHWM:");
        g_last_peak_host_mb = (hwm_kb > g_host_baseline_kb)
                                  ? double(hwm_kb - g_host_baseline_kb) / 1024.0 : 0.0;
    }
};

template <class SnapFn, class FinalizeFn>
double RunSim(const Cfg& cfg, Grids& g, SnapFn snap, FinalizeFn finalize,
              uint64_t* checksum_out, PhaseTrace* trace = nullptr) {
    unsigned N = cfg.N;
    uint64_t cells = cfg.cells();
    unsigned threads = 256, blocks = unsigned((cells + threads - 1) / threads);
    ctp::GpuApi::Synchronize();  // settle the seed before timing
    MemSampler mem;
    mem.Start();                 // no-op unless GSBENCH_MEM=1
    auto t0 = std::chrono::steady_clock::now();
    unsigned si = 0;
    for (unsigned step = 1; step <= cfg.steps(); ++step) {
        // First step of a snapshot interval => mark where this snapshot's compute begins.
        if (trace && (step - 1) % cfg.steps_per == 0) trace->CompStart(si);
        GsStepKernel<<<blocks, threads>>>(g.u_curr, g.v_curr, g.u_next, g.v_next,
                                          kGs, N);
        std::swap(g.u_curr, g.u_next);
        std::swap(g.v_curr, g.v_next);
        if (step % cfg.steps_per != 0) continue;
        if (trace) trace->CompEnd(si);   // compute done; snap() is the submit phase
        snap(si, g.v_curr);
        if (trace) trace->SubmitEnd(si);
        ++si;
    }
    if (trace) trace->DrainStart();
    finalize();                          // async: tail-drain all outstanding puts
    if (trace) trace->DrainEnd();
    ctp::GpuApi::Synchronize();
    auto t1 = std::chrono::steady_clock::now();
    mem.Stop();          // records g_last_peak_device_mb (no-op unless GSBENCH_MEM=1)
    (void)checksum_out;  // checksum computed by arms post-run from persisted bytes
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

void PrintResult(const char* arm, const Cfg& cfg, double ms, uint64_t checksum) {
    double mb = double(cfg.total_bytes()) / (1024.0 * 1024.0);
    // durable= is the flush state THIS arm ran under, so a result line can never be mistaken for...
    const bool clio_arm = (std::strncmp(arm, "raw", 3) != 0 &&
                           std::strncmp(arm, "hdf5", 4) != 0 &&
                           std::strcmp(arm, "async_VOL") != 0);
    const unsigned durable = clio_arm ? (cfg.clio_fsync && cfg.bdev == "file")
                                      : cfg.raw_fsync;
    std::fprintf(stderr,
        "GSBENCH_RESULT arm=%s N=%u chunks=%u blocks=%u snaps=%u steps=%u bdev=%s "
        "pinned=%u durable=%u MB=%.1f ms=%.2f MBps=%.1f io_buf_mb=%.1f checksum=%llu\n",
        arm, cfg.N, cfg.chunks, cfg.submit_grid(), cfg.snaps, cfg.steps(),
        cfg.bdev.c_str(), ReportedPinned(cfg), durable,
        mb, ms, mb / (ms / 1000.0),
        double(g_io_buf_bytes) / (1024.0 * 1024.0), (unsigned long long)checksum);
    // Peak device-memory line (only when GSBENCH_MEM=1 populated it). Separate line so the
    // GSBENCH_RESULT format is unchanged for existing parsers; associate by the arm= field.
    if (g_last_peak_device_mb >= 0.0 || g_last_peak_host_mb >= 0.0) {
        std::fprintf(stderr,
            "GSBENCH_MEM arm=%s N=%u chunks=%u snaps=%u bdev=%s pinned=%u "
            "peak_device_mb=%.1f peak_host_mb=%.1f\n",
            arm, cfg.N, cfg.chunks, cfg.snaps, cfg.bdev.c_str(), ReportedPinned(cfg),
            g_last_peak_device_mb, g_last_peak_host_mb);
    }
}

// ---- CLIO env bring-up (configurable bdev) ---------------------------------

struct BenchEnv {
    clio::cte::core::TagId probe_tag;
    BenchEnv(const Cfg& cfg) {
        using namespace std::chrono_literals;
        namespace bdev = clio::run::bdev;
        std::fprintf(stderr, "[bench] bringing up server (bdev=%s cap=%uMB)\n",
                     cfg.bdev.c_str(), cfg.cap_mb);
        if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer))
            throw std::runtime_error("CLIO_INIT(kServer) failed");
        if (!clio::cte::core::CLIO_CTE_CLIENT_INIT())
            throw std::runtime_error("CLIO_CTE_CLIENT_INIT failed");
        auto* cte = CLIO_CTE_CLIENT;
        cte->Init(clio::cte::core::kCtePoolId);
        clio::cte::core::CreateParams params;
        auto ct = cte->AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                   clio::cte::core::kCtePoolName,
                                   clio::cte::core::kCtePoolId, params);
        ct.Wait();
        if (ct->GetReturnCode() != 0) throw std::runtime_error("CTE create failed");
        std::this_thread::sleep_for(50ms);

        const clio::run::u64 cap = clio::run::u64(cfg.cap_mb) << 20;
        const bool is_file = (cfg.bdev == "file");
        // "pinned" is kRam's page-locked sibling: same in-memory bdev, but its pages are cudaMallocH...
        const bdev::BdevType type = is_file     ? bdev::BdevType::kFile
                                    : (cfg.bdev == "pinned") ? bdev::BdevType::kPinned
                                                             : bdev::BdevType::kRam;
        // For kFile the bdev name IS the on-disk file path (O_DIRECT). For kRam/kPinned it
        // is just an identifier.
        const std::string name = is_file ? cfg.bdev_path : std::string("gsbench_ram");
        clio::run::PoolId bdev_pool_id(960, 0);
        bdev::Client bclient(bdev_pool_id);
        auto bc = bclient.AsyncCreate(clio::run::PoolQuery::Dynamic(), name, bdev_pool_id,
                                      type, cap);
        bc.Wait();
        if (bc->GetReturnCode() != 0) throw std::runtime_error("bdev create failed");
        std::this_thread::sleep_for(50ms);
        auto rt = cte->AsyncRegisterTarget(name, type, cap, clio::run::PoolQuery::Local(),
                                           bdev_pool_id);
        rt.Wait();
        if (rt->GetReturnCode() != 0) throw std::runtime_error("RegisterTarget failed");
        std::this_thread::sleep_for(50ms);
        std::fprintf(stderr, "[bench] server ready\n");
        // Peak-memory baseline (GSBENCH_MEM=1): captured now -- server + bdev are up, but the arm ha...
        if (EnvU0("GSBENCH_MEM", 0)) {
            size_t f = 0, t = 0;
            if (cudaMemGetInfo(&f, &t) == cudaSuccess) g_mem_baseline_free = f;
        }
    }
};

// Flush CLIO's kFile bdev to the device (see Cfg::clio_fsync for why this must exist).
void ClioBdevSync(const Cfg& cfg) {
    if (!cfg.clio_fsync || cfg.bdev != "file") return;
    int fd = open(cfg.bdev_path.c_str(), O_RDONLY);
    if (fd < 0) return;                      // bdev file gone => nothing to flush
    fdatasync(fd);                           // Linux permits fsync on a read-only fd
    close(fd);
}

clio::cte::core::TagId MakeTag(const char* name) {
    auto t = CLIO_CTE_CLIENT->AsyncGetOrCreateTag(name);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    return t->tag_id_;
}

std::vector<byte_t> HostReadBlob(clio::cte::core::TagId tag, const std::string& name,
                                 uint64_t size) {
    ctp::ipc::FullPtr<char> buf = CLIO_CPU_IPC->AllocateBuffer(size);
    REQUIRE(!buf.IsNull());
    std::memset(buf.ptr_, 0, size);
    ctp::ipc::ShmPtr<> shm = buf.shm_.template Cast<void>();
    auto t = CLIO_CTE_CLIENT->AsyncGetBlob(tag, name, clio::run::u64(0), size,
                                           clio::run::u32(0), shm);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    std::vector<byte_t> out(size);
    std::memcpy(out.data(), buf.ptr_, size);
    return out;
}

// Pre-create `snaps` snapshot datasets (distinct path => distinct tag). Kept well under
// the ~16-large-backend ceiling by the caller (snaps <= ~12).
//
// pool_size (M) / grid_size (G) carry the bounded-pool contract to the dataset: M resident
// data buffers, G producer blocks, pipeline depth D = M/G. 0/0 (the sync and async arms)
// means M == N and G == M, i.e. one buffer per chunk — the historical shape those two arms
// require, since their fill (TwCopyKernel) and their fire are SEPARATE kernels and would
// otherwise clobber a shared buffer. Only the pooled arm passes nonzero values.
void MakeSnapDatasets(clio::run::IpcManager* ipc, clio::run::IpcManagerGpuInfo gpu_info,
                      const char* prefix, const Cfg& cfg,
                      std::vector<kvhdf5::GpuCteDataset>& out,
                      std::vector<clio::cte::core::TagId>& tags,
                      unsigned pool_size = 0, unsigned grid_size = 0) {
    kvhdf5::Layout layout{/*dims=*/{cfg.cells()},
                          /*chunk_dims=*/{cfg.cells() / cfg.chunks},
                          /*elem_size=*/sizeof(float)};
    REQUIRE(layout.ChunkCount() == cfg.chunks);
    const auto data_kind = cfg.data_pinned
        ? kvhdf5::GpuCteDataset::MemKind::kPinnedHost
        : kvhdf5::GpuCteDataset::MemKind::kDeviceMem;
    out.clear(); tags.clear(); out.reserve(cfg.snaps);
    for (unsigned s = 0; s < cfg.snaps; ++s) {
        char path[160];
        std::snprintf(path, sizeof(path), "%s/v/step_%04u", prefix, s);
        out.emplace_back(kvhdf5::GpuCteDataset::FromPath(
            ipc, gpu_info, /*gpu_id=*/0, CLIO_CTE_CLIENT, path, layout,
            pool_size, data_kind, grid_size));
        tags.push_back(MakeTag(kvhdf5::tagpath::CanonicalTag(path).c_str()));
    }
}

// The reader's global PDF accumulator: ONE histogram over every snapshot every arm reads
// back. Its fold (below) is the cross-arm READ correctness gate -- see the HistKernel note.
struct Pdf {
    unsigned long long* d_ = nullptr;
    uint64_t bytes_ = 0;                 // total bytes binned (the reader's I/O volume)
    Pdf() {
        REQUIRE(cudaMalloc(&d_, kHistBins * sizeof(unsigned long long)) == cudaSuccess);
        REQUIRE(cudaMemset(d_, 0, kHistBins * sizeof(unsigned long long)) == cudaSuccess);
    }
    ~Pdf() { if (d_) cudaFree(d_); }
    Pdf(const Pdf&) = delete;
    Pdf& operator=(const Pdf&) = delete;

    // Bin `bytes` of float data at a DEVICE-ADDRESSABLE pointer. Device-backed datasets pass
    // their chunk buffer straight through; host arms stage into a device scratch first.
    void Add(const void* data, uint64_t bytes) {
        const uint64_t n = bytes / sizeof(float);
        unsigned blocks = unsigned((n + 255) / 256);
        if (blocks > 2048) blocks = 2048;
        if (blocks < 1) blocks = 1;
        HistKernel<<<blocks, 256>>>(static_cast<const float*>(data), n, d_);
        bytes_ += bytes;
    }

    // FNV over the bin counts: one number that MUST match across every reader arm.
    uint64_t Checksum() const {
        std::vector<unsigned long long> h(kHistBins);
        ctp::GpuApi::Synchronize();
        REQUIRE(cudaMemcpy(h.data(), d_, kHistBins * sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        uint64_t x = 1469598103934665603ull;
        for (auto v : h) { x ^= v; x *= 1099511628211ull; }
        return x;
    }
    // Total binned elements — must equal snaps*cells, i.e. proof nothing was short-read.
    uint64_t Count() const {
        std::vector<unsigned long long> h(kHistBins);
        ctp::GpuApi::Synchronize();
        REQUIRE(cudaMemcpy(h.data(), d_, kHistBins * sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        uint64_t t = 0;
        for (auto v : h) t += v;
        return t;
    }
};

// Time a read-back pass.
template <typename ReadSnapFn>
double RunReadPhase(const Cfg& cfg, ReadSnapFn read_snap) {
    ctp::GpuApi::Synchronize();
    auto t0 = std::chrono::steady_clock::now();
    for (unsigned si = 0; si < cfg.snaps; ++si) read_snap(si);
    ctp::GpuApi::Synchronize();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

void PrintReadResult(const char* arm, const Cfg& cfg, double ms, const Pdf& pdf,
                     const char* reader) {
    const double mb = double(cfg.total_bytes()) / (1024.0 * 1024.0);
    std::fprintf(stderr,
        "GSBENCH_READ arm=%s reader=%s N=%u chunks=%u snaps=%u bdev=%s pinned=%u MB=%.1f "
        "ms=%.2f MBps=%.1f pdf=%llu cells=%llu\n",
        arm, reader, cfg.N, cfg.chunks, cfg.snaps, cfg.bdev.c_str(), ReportedPinned(cfg),
        mb, ms, mb / (ms / 1000.0),
        (unsigned long long)pdf.Checksum(), (unsigned long long)pdf.Count());
}

// GPU-initiated reader shared by every CLIO/GPUH5 arm: ONE reused dataset, re-tagged per sna...
double GpuReadPdf(const Cfg& cfg, kvhdf5::GpuCteDataset& rd,
                  const std::vector<clio::cte::core::TagId>& tags,
                  unsigned submit_grid, Pdf& pdf) {
    const uint64_t rchunk = cfg.grid_bytes() / cfg.chunks;
    return RunReadPhase(cfg, [&](unsigned si) {
        rd.Rearm(tags[si]);
        TwReadKernel<<<submit_grid, 256>>>(rd.Handle());
        ctp::GpuApi::Synchronize();     // reads landed => safe to bin, then to re-arm
        for (unsigned c = 0; c < cfg.chunks; ++c) pdf.Add(rd.DeviceData(c), rchunk);
        ctp::GpuApi::Synchronize();
    });
}

// ASYNC (pipelined) GPU reader: 2 reused read buffers.
double GpuReadPdfAsync(const Cfg& cfg, std::vector<kvhdf5::GpuCteDataset>& groups,
                       const std::vector<clio::cte::core::TagId>& tags,
                       unsigned submit_grid, Pdf& pdf) {
    REQUIRE(groups.size() >= 2);
    const uint64_t rchunk = cfg.grid_bytes() / cfg.chunks;
    // Device tag table so the fire kernel can self-stamp. Setup, outside the timed region.
    clio::cte::core::TagId* d_tags = nullptr;
    REQUIRE(cudaMalloc(&d_tags, cfg.snaps * sizeof(clio::cte::core::TagId)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_tags, tags.data(), cfg.snaps * sizeof(clio::cte::core::TagId),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    ctp::GpuApi::Synchronize();
    const auto t0 = std::chrono::steady_clock::now();
    TwReadFireKernel<<<submit_grid, 256>>>(groups[0].Handle(), d_tags, 0);   // prime the pipe
    for (unsigned si = 0; si < cfg.snaps; ++si) {
        if (si + 1 < cfg.snaps)   // fetch the NEXT snapshot while this one reduces
            TwReadFireKernel<<<submit_grid, 256>>>(groups[(si + 1) % 2].Handle(), d_tags, si + 1);
        kvhdf5::GpuCteDataset& g = groups[si % 2];
        TwReadDrainKernel<<<submit_grid, 32>>>(g.Handle());
        for (unsigned c = 0; c < cfg.chunks; ++c) pdf.Add(g.DeviceData(c), rchunk);
    }
    ctp::GpuApi::Synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    cudaFree(d_tags);
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Host-side arms (raw / hdf5 / hostclio) read into a HOST buffer, then stage H2D through thi...
struct HostReadStage {
    byte_t* d_ = nullptr;
    uint64_t bytes_ = 0;
    explicit HostReadStage(uint64_t bytes) : bytes_(bytes) {
        REQUIRE(cudaMalloc(&d_, bytes) == cudaSuccess);
    }
    ~HostReadStage() { if (d_) cudaFree(d_); }
    HostReadStage(const HostReadStage&) = delete;
    HostReadStage& operator=(const HostReadStage&) = delete;
    void Bin(Pdf& pdf, const void* host_src) {
        REQUIRE(cudaMemcpy(d_, host_src, bytes_, cudaMemcpyHostToDevice) == cudaSuccess);
        pdf.Add(d_, bytes_);
    }
};

// Read every snapshot back and fold into one checksum (proves what was persisted).
uint64_t ChecksumSnapshots(const std::vector<clio::cte::core::TagId>& tags,
                           const Cfg& cfg) {
    const uint64_t chunk_bytes = cfg.grid_bytes() / cfg.chunks;
    uint64_t h = 1469598103934665603ull;
    for (unsigned s = 0; s < cfg.snaps; ++s)
        for (unsigned c = 0; c < cfg.chunks; ++c) {
            auto got = HostReadBlob(tags[s], std::to_string(c), chunk_bytes);
            h = Fnv1a(got.data(), got.size(), h);
        }
    return h;
}

// The three GPU-producer submission shapes.
enum class ClioMode { kSync, kAsync, kPooled };

const char* ClioModeName(ClioMode m) {
    switch (m) {
        case ClioMode::kSync:   return "gpuh5_sync_relaunch";
        case ClioMode::kAsync:  return "gpuh5_noreuse";
        default:                return "pooled";
    }
}

// Run a CLIO arm. Returns ms; fills checksum from persisted bytes.
double RunClioArm(const Cfg& cfg, ClioMode mode, const char* prefix, uint64_t* checksum) {
    const bool async = (mode == ClioMode::kAsync);
    const bool pooled = (mode == ClioMode::kPooled);
    const bool sync = (mode == ClioMode::kSync);
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

    // Arm the submit probe BEFORE the datasets are built: MakeSnapDatasets copies gpu_info by va...
    SubmitProbeHarness probe(EnvU0("GSBENCH_SUBMIT_PROBE", 0) != 0,
                             cfg.snaps * cfg.chunks + 64, &gpu_info);

    // The pooled arm is the only one that carries a (M, G) pool contract into the dataset; sync/...
    const unsigned pool_m = pooled ? cfg.pool : 0u;
    const unsigned pool_g = pooled ? cfg.submit_grid() : 0u;

    std::vector<kvhdf5::GpuCteDataset> ds;
    std::vector<clio::cte::core::TagId> tags;
    if (sync || pooled) {
        // ONE reused dataset, re-tagged per snapshot (Rearm(tags[si])), bounded regardless of snaps....
        tags.reserve(cfg.snaps);
        for (unsigned s = 0; s < cfg.snaps; ++s) {
            char p[160];
            std::snprintf(p, sizeof(p), "%s/v/step_%04u", prefix, s);
            tags.push_back(MakeTag(kvhdf5::tagpath::CanonicalTag(p).c_str()));
        }
        kvhdf5::Layout layout{/*dims=*/{cfg.cells()},
                              /*chunk_dims=*/{cfg.cells() / cfg.chunks},
                              /*elem_size=*/sizeof(float)};
        const auto data_kind = cfg.data_pinned
            ? kvhdf5::GpuCteDataset::MemKind::kPinnedHost
            : kvhdf5::GpuCteDataset::MemKind::kDeviceMem;
        char path[160];
        std::snprintf(path, sizeof(path), "%s/v/reuse_sync", prefix);
        ds.emplace_back(kvhdf5::GpuCteDataset::FromPath(
            ipc, gpu_info, /*gpu_id=*/0, CLIO_CTE_CLIENT, path, layout,
            pool_m, data_kind, pool_g));
    } else {
        MakeSnapDatasets(ipc, gpu_info, prefix, cfg, ds, tags, pool_m, pool_g);
    }
    {   // deterministic I/O buffer footprint: (sync/pooled) 1 reused dataset, async = snaps
        uint64_t io = 0;
        for (auto& d : ds) io += d.DeviceDataBytes();
        g_io_buf_bytes = io;
    }
    if (pooled) {
        const uint64_t chunk_bytes = cfg.grid_bytes() / cfg.chunks;
        std::fprintf(stderr,
            "GSBENCH_POOLED N=%u M=%u G=%u D=%u chunk_bytes=%llu "
            "resident_data_bytes=%llu (unpooled M=N would be %llu)\n",
            ds[0].ChunkCount(), ds[0].PoolSize(), ds[0].GridSize(), ds[0].Depth(),
            (unsigned long long)chunk_bytes,
            (unsigned long long)(uint64_t(ds[0].PoolSize()) * chunk_bytes),
            (unsigned long long)(uint64_t(cfg.chunks) * chunk_bytes));
    }

    Masker masker(cfg.N, cfg.incompressible != 0);
    Grids g = MakeGrids(cfg.N);
    // pooled, like async, defers its outstanding Puts to a tail drain in finalize(), so it
    // gets a drain bucket too.
    PhaseTrace trace(EnvU("GSBENCH_TRACE", 0) != 0, cfg.snaps, async || pooled);
    // Grid sizing for the decoupled bulk copy: one gridDim.y per chunk, enough blocks/chunk
    // (each thread copies one 4-byte word, grid-strided) to saturate HBM.
    const uint64_t copy_chunk_bytes = cfg.grid_bytes() / cfg.chunks;
    const unsigned copy_words = unsigned(copy_chunk_bytes / sizeof(uint32_t));
    unsigned copy_bpc = (copy_words + 255) / 256;
    if (copy_bpc < 1) copy_bpc = 1;
    if (copy_bpc > 2048) copy_bpc = 2048;
    const unsigned submit_grid = cfg.submit_grid();  // "number of GPU blocks" axis
    // CUDA 12 loads a kernel's device code lazily, ON ITS FIRST LAUNCH (CUDA_MODULE_LOADING=LAZY...
    if (cfg.prewarm) {
        cudaFuncAttributes fa;
        cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(TwCopyKernel));
        // Warm whichever submit-kernel instantiation this run actually launches
        // (see the probe.on_ branch at the launch site below).
        if (probe.on_) {
            cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(TwSnapFireKernel<true>));
            cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(TwSnapSyncKernel<true>));
            cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(TwSnapPooledKernel<true>));
        } else {
            cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(TwSnapFireKernel<false>));
            cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(TwSnapSyncKernel<false>));
            cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(TwSnapPooledKernel<false>));
        }
        cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(TwDrainKernel));
    }
    // ---- Bound host run-ahead for the POOLED arm (GSBENCH_PACE) -------------- The sync arm's ...
    const unsigned pace_dflt = 512u / (cfg.steps_per + 2u);
    const unsigned pace_k = EnvU0("GSBENCH_PACE", pace_dflt < 2u ? 2u : pace_dflt);
    std::vector<cudaEvent_t> pace_ev;
    if (pooled && pace_k) {
        pace_ev.resize(cfg.snaps);
        for (auto& e : pace_ev) cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
    }
    auto pace = [&](unsigned si) {
        if (pace_ev.empty()) return;          // not pooled, or GSBENCH_PACE=0
        cudaEventRecord(pace_ev[si]);
        if (si >= pace_k) cudaEventSynchronize(pace_ev[si - pace_k]);
    };
    // PER-SNAPSHOT durability (GSBENCH_CLIO_PERSNAP): drain THIS snapshot's Puts to the bdev, th...
    auto durable_persnap = [&](kvhdf5::GpuDatasetHandle h) {
        if (!cfg.clio_persnap) return;
        TwDrainKernel<<<submit_grid, 32>>>(h);   // land this snapshot's writes into the bdev
        ctp::GpuApi::Synchronize();
        ClioBdevSync(cfg);                        // fdatasync the bdev backing file
    };

    auto snap = [&](unsigned si, float* v_curr) {
        float* src = masker.Apply(v_curr);   // incompressible view (or v_curr if disabled)
        const byte_t* bsrc = reinterpret_cast<const byte_t*>(src);
        if (pooled) {
            // ONE fused kernel: fill + fire + bounded intra-snapshot drain over M buffers.
            kvhdf5::GpuCteDataset& d = ds[0];
            d.Rearm(tags[si]);
            if (probe.on_)
                TwSnapPooledKernel<true><<<d.GridSize(), 256>>>(d.Handle(), bsrc);
            else
                TwSnapPooledKernel<false><<<d.GridSize(), 256>>>(d.Handle(), bsrc);
            TwDrainKernel<<<d.GridSize(), 32>>>(d.Handle());   // land this snapshot's tail
            ctp::GpuApi::Synchronize();                         // free the buffer before re-arm
            durable_persnap(d.Handle());   // per-snapshot durable (no-op unless armed)
            return;
        }
        // sync reuses ONE dataset, re-tagged per snapshot (race-free: it drains below before
        // the next re-arm); async keeps one dataset per snapshot (all its writes are in flight).
        kvhdf5::GpuCteDataset& d = sync ? ds[0] : ds[si];
        if (sync) d.Rearm(tags[si]);
        TwCopyKernel<<<dim3(copy_bpc, cfg.chunks), 256>>>(d.Handle(), bsrc);  // multi-block stage
        if (async) {
            // Instantiate the non-probing kernel unless the submit probe is armed
            // this run — the <false> path drops SendIn's probe registers.
            if (probe.on_) TwSnapFireKernel<true><<<submit_grid, 256>>>(d.Handle());
            else TwSnapFireKernel<false><<<submit_grid, 256>>>(d.Handle());
        } else {
            if (probe.on_) TwSnapSyncKernel<true><<<submit_grid, 256>>>(d.Handle());
            else TwSnapSyncKernel<false><<<submit_grid, 256>>>(d.Handle());
            // Bound the CUDA pending-launch queue. The sync arm's in-kernel SubmitWait
            // spins waiting for the IN-PROCESS server to flip the completion flag. If the
            // host races ahead and fills the ~1024-deep launch queue (once the run's total
            // kernels, 12*(steps_per+2), exceed it) it blocks inside cudaLaunchKernel while
            // the GPU is stalled on that spin — and the server can't make forward progress
            // from the same process → DEADLOCK (repros at steps_per>=96). A per-snapshot
            // sync caps in-flight work to one interval. Free for the sync arm (it already
            // waits on every put); the async arm must NOT do this — racing ahead IS its
            // overlap, and it never wedges (its waits are deferred to the end drain).
            ctp::GpuApi::Synchronize();
        }
        durable_persnap(d.Handle());   // per-snapshot durable (no-op unless armed)
    };
    auto finalize = [&]() {
        // Both fire-and-defer arms drain here, at the END of the timed region, so each snapshot's I/...
        if (async) for (auto& d : ds) TwDrainKernel<<<submit_grid, 32>>>(d.Handle());
        if (pooled) for (auto& d : ds) TwDrainKernel<<<d.GridSize(), 32>>>(d.Handle());
        // The drain kernels must land before we can flush what they wrote: only then has the server ...
        ctp::GpuApi::Synchronize();
        ClioBdevSync(cfg);   // durability parity with raw/hdf5 — see Cfg::clio_fsync
    };
    double ms = RunSim(cfg, g, snap, finalize, nullptr, &trace);
    // finalize() closed with a Synchronize, so the device is idle and these are dead.
    for (auto& e : pace_ev) cudaEventDestroy(e);

    // A PutBlob that does not fit in the bdev FAILS, and until recently did so SILENTLY (the run...
    ctp::GpuApi::Synchronize();  // async arm's drain kernels must land first
    for (unsigned s = 0; s < ds.size(); ++s)
        ds[s].ThrowIfIoFailed(
            ("gsbench snapshot " + std::to_string(s)).c_str());

    FreeGrids(g);
    trace.Report(ClioModeName(mode));   // GSBENCH_TRACE=1: emit phase breakdown
    // Dumped only after ThrowIfIoFailed above: a breakdown from a run with failed puts
    // would be timing an I/O that never happened.
    probe.Dump(ClioModeName(mode), EnvS("GSBENCH_PROBE_DIR", ".").c_str());
    // READER: GPU-initiated read-back + PDF.
    if (cfg.read_pdf) {
        Pdf pdf;
        const bool amode = (cfg.read_async && ds.size() >= 2);
        const double rms = amode ? GpuReadPdfAsync(cfg, ds, tags, submit_grid, pdf)
                                 : GpuReadPdf(cfg, ds[0], tags, submit_grid, pdf);
        PrintReadResult(ClioModeName(mode), cfg, rms, pdf, amode ? "async" : "sync");
    }
    *checksum = ChecksumSnapshots(tags, cfg);
    return ms;
}

// REUSE arm (DESIGN §7 "Option B", relaunched): the async arm's well-parallelized separate k...
double RunReuseArm(const Cfg& cfg, const char* prefix, uint64_t* checksum) {
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

    const kvhdf5::Layout layout{/*dims=*/{cfg.cells()},
                                /*chunk_dims=*/{cfg.cells() / cfg.chunks},
                                /*elem_size=*/sizeof(float)};
    REQUIRE(layout.ChunkCount() == cfg.chunks);
    const auto data_kind = cfg.data_pinned
        ? kvhdf5::GpuCteDataset::MemKind::kPinnedHost
        : kvhdf5::GpuCteDataset::MemKind::kDeviceMem;

    // One tag per snapshot (distinct dataset), resolved off the hot path, plus a
    // device-visible TagId table the fire kernel indexes by snapshot.
    std::vector<clio::cte::core::TagId> tags;
    tags.reserve(cfg.snaps);
    for (unsigned s = 0; s < cfg.snaps; ++s) {
        char path[160];
        std::snprintf(path, sizeof(path), "%s/v/step_%04u", prefix, s);
        tags.push_back(MakeTag(kvhdf5::tagpath::CanonicalTag(path).c_str()));
    }
    clio::cte::core::TagId* d_tag_table = nullptr;
    REQUIRE(cudaMalloc(&d_tag_table,
                       cfg.snaps * sizeof(clio::cte::core::TagId)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_tag_table, tags.data(),
                       cfg.snaps * sizeof(clio::cte::core::TagId),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    // The buffer groups (GSBENCH_GROUPS, default 2; capped at snaps), constructed ONCE and reuse...
    const unsigned ngroups = std::min(EnvU("GSBENCH_GROUPS", 2), cfg.snaps);
    std::vector<kvhdf5::GpuCteDataset> groups;
    groups.reserve(ngroups);
    for (unsigned gi = 0; gi < ngroups; ++gi) {
        char path[160];
        std::snprintf(path, sizeof(path), "%s/v/group_%u", prefix, gi);
        groups.emplace_back(kvhdf5::GpuCteDataset::FromPath(
            ipc, gpu_info, /*gpu_id=*/0, CLIO_CTE_CLIENT, path, layout,
            /*pool_size=*/0, data_kind, /*grid_size=*/0));
    }
    {
        const uint64_t chunk_bytes = cfg.grid_bytes() / cfg.chunks;
        uint64_t resident = 0;
        for (auto& d : groups) resident += d.DeviceDataBytes();
        g_io_buf_bytes = resident;   // deterministic I/O buffer footprint (2 reused groups)
        std::fprintf(stderr,
            "GSBENCH_REUSE groups=%u resident_data_bytes=%llu (async M=N x snaps "
            "would be %llu)\n",
            ngroups, (unsigned long long)resident,
            (unsigned long long)(uint64_t(cfg.chunks) * chunk_bytes * cfg.snaps));
    }

    Masker masker(cfg.N, cfg.incompressible != 0);
    Grids g = MakeGrids(cfg.N);
    PhaseTrace trace(EnvU("GSBENCH_TRACE", 0) != 0, cfg.snaps, /*async=*/true);

    const uint64_t copy_chunk_bytes = cfg.grid_bytes() / cfg.chunks;
    const unsigned copy_words = unsigned(copy_chunk_bytes / sizeof(uint32_t));
    unsigned copy_bpc = (copy_words + 255) / 256;
    if (copy_bpc < 1) copy_bpc = 1;
    if (copy_bpc > 2048) copy_bpc = 2048;
    const unsigned submit_grid = cfg.submit_grid();

    if (cfg.prewarm) {
        cudaFuncAttributes fa;
        cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(TwCopyKernel));
        cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(ReuseFireKernel));
        cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(TwDrainKernel));
    }

    // Bound host run-ahead (GSBENCH_PACE).
    const unsigned pace_dflt = 512u / (cfg.steps_per + 2u);
    const unsigned pace_k = EnvU0("GSBENCH_PACE", pace_dflt < 2u ? 2u : pace_dflt);
    std::vector<cudaEvent_t> pace_ev;
    if (pace_k) {
        pace_ev.resize(cfg.snaps);
        for (auto& e : pace_ev) cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
    }
    auto pace = [&](unsigned si) {
        if (pace_ev.empty()) return;
        cudaEventRecord(pace_ev[si]);
        if (si >= pace_k) cudaEventSynchronize(pace_ev[si - pace_k]);
    };

    auto snap = [&](unsigned si, float* v_curr) {
        const unsigned gi = si % ngroups;
        float* src = masker.Apply(v_curr);
        const byte_t* bsrc = reinterpret_cast<const byte_t*>(src);
        // DRAIN-BEFORE-REFILL: reclaim this group's buffers from snapshot si-ngroups, in-stream (the...
        if (si >= ngroups)
            TwDrainKernel<<<submit_grid, 32>>>(groups[gi].Handle());
        TwCopyKernel<<<dim3(copy_bpc, cfg.chunks), 256>>>(groups[gi].Handle(), bsrc);
        ReuseFireKernel<<<submit_grid, 256>>>(groups[gi].Handle(), d_tag_table, si);
        // PER-SNAPSHOT durability (GSBENCH_CLIO_PERSNAP): drain THIS snapshot's Puts to the bdev + f...
        if (cfg.clio_persnap) {
            TwDrainKernel<<<submit_grid, 32>>>(groups[gi].Handle());
            ctp::GpuApi::Synchronize();
            ClioBdevSync(cfg);
        }
        pace(si);
    };
    auto finalize = [&]() {
        for (auto& d : groups) TwDrainKernel<<<submit_grid, 32>>>(d.Handle());
        ctp::GpuApi::Synchronize();
        ClioBdevSync(cfg);   // durability parity with raw/hdf5 — see Cfg::clio_fsync
    };

    double ms = RunSim(cfg, g, snap, finalize, nullptr, &trace);
    for (auto& e : pace_ev) cudaEventDestroy(e);

    ctp::GpuApi::Synchronize();
    // Best-effort I/O-failure check on the reused groups (each holds only its LAST snapshot's pe...
    for (unsigned gi = 0; gi < ngroups; ++gi)
        groups[gi].ThrowIfIoFailed(("gsbench reuse group " + std::to_string(gi)).c_str());

    FreeGrids(g);
    cudaFree(d_tag_table);
    trace.Report("reuse");
    if (cfg.read_pdf) {   // GPU-initiated read-back + PDF (one reused group, re-tagged)
        Pdf pdf;
        const bool amode = (cfg.read_async && groups.size() >= 2);
        const double rms = amode ? GpuReadPdfAsync(cfg, groups, tags, submit_grid, pdf)
                                 : GpuReadPdf(cfg, groups[0], tags, submit_grid, pdf);
        PrintReadResult("gpuh5_relaunch", cfg, rms, pdf, amode ? "async" : "sync");
    }
    *checksum = ChecksumSnapshots(tags, cfg);
    return ms;
}

// PERSISTENT arm (DESIGN §7 "Option A"): the whole snapshot loop in ONE resident cooperative...
double RunPersistentArm(const Cfg& cfg, const char* prefix, uint64_t* checksum,
                        bool async_submit = true) {
    // Select the resident-kernel submit mode: async fire (double-buffered producer) or
    // fire-AND-wait per snapshot (the persistent analog of gpuh5_sync).
    const void* kfn = async_submit
        ? reinterpret_cast<const void*>(GsPersistentKernel<true>)
        : reinterpret_cast<const void*>(GsPersistentKernel<false>);
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

    const kvhdf5::Layout layout{/*dims=*/{cfg.cells()},
                                /*chunk_dims=*/{cfg.cells() / cfg.chunks},
                                /*elem_size=*/sizeof(float)};
    REQUIRE(layout.ChunkCount() == cfg.chunks);
    // MANDATORY no-device-op data backend: the server must complete Puts WITHOUT any device oper...
    const bool uvm = EnvU0("GSBENCH_PERSIST_UVM", 0) != 0;
    const auto data_kind = uvm ? kvhdf5::GpuCteDataset::MemKind::kManagedUvm
                               : kvhdf5::GpuCteDataset::MemKind::kPinnedHost;
    g_actual_pinned = 1;   // FORCED off kDeviceMem regardless of GSBENCH_DATA_PINNED

    std::vector<clio::cte::core::TagId> tags;
    tags.reserve(cfg.snaps);
    for (unsigned s = 0; s < cfg.snaps; ++s) {
        char path[160];
        std::snprintf(path, sizeof(path), "%s/v/step_%04u", prefix, s);
        tags.push_back(MakeTag(kvhdf5::tagpath::CanonicalTag(path).c_str()));
    }
    clio::cte::core::TagId* d_tag_table = nullptr;
    REQUIRE(cudaMalloc(&d_tag_table,
                       cfg.snaps * sizeof(clio::cte::core::TagId)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_tag_table, tags.data(),
                       cfg.snaps * sizeof(clio::cte::core::TagId),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    // Reused buffer groups.
    const unsigned ngroups = async_submit
        ? std::min(std::max(2u, EnvU("GSBENCH_GROUPS", 2)), cfg.snaps)
        : 1u;
    std::vector<kvhdf5::GpuCteDataset> groups;
    groups.reserve(ngroups);
    for (unsigned gi = 0; gi < ngroups; ++gi) {
        char path[160];
        std::snprintf(path, sizeof(path), "%s/v/group_%u", prefix, gi);
        groups.emplace_back(kvhdf5::GpuCteDataset::FromPath(
            ipc, gpu_info, /*gpu_id=*/0, CLIO_CTE_CLIENT, path, layout,
            /*pool_size=*/0, data_kind, /*grid_size=*/0));
    }
    // The kernel indexes a DEVICE ARRAY of handles (groups[s % G]) rather than taking them as by...
    std::vector<kvhdf5::GpuDatasetHandle> h_handles;
    h_handles.reserve(ngroups);
    for (auto& d : groups) h_handles.push_back(d.Handle());
    kvhdf5::GpuDatasetHandle* d_handles = nullptr;
    REQUIRE(cudaMalloc(&d_handles,
                       ngroups * sizeof(kvhdf5::GpuDatasetHandle)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_handles, h_handles.data(),
                       ngroups * sizeof(kvhdf5::GpuDatasetHandle),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    {
        const uint64_t chunk_bytes = cfg.grid_bytes() / cfg.chunks;
        uint64_t resident = 0;
        for (auto& d : groups) resident += d.DeviceDataBytes();
        g_io_buf_bytes = resident;   // deterministic I/O buffer footprint (2 pinned groups)
        std::fprintf(stderr,
            "GSBENCH_PERSISTENT groups=%u backend=%s resident_data_bytes=%llu (async M=N x snaps "
            "would be %llu)\n", ngroups, uvm ? "uvm" : "pinned",
            (unsigned long long)resident,
            (unsigned long long)(uint64_t(cfg.chunks) * chunk_bytes * cfg.snaps));
    }

    Masker masker(cfg.N, cfg.incompressible != 0);
    Grids g = MakeGrids(cfg.N);

    // Occupancy-sized grid: cooperative launch requires the whole grid co-resident. Grid-stride
    // (compute over cells, I/O over chunks) makes any grid <= that limit correct.
    int dev = 0; cudaGetDevice(&dev);
    int num_sm = 0;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev);
    const int block = 256;
    int blocks_per_sm = 0;
    REQUIRE(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, kfn, block, 0) == cudaSuccess);
    REQUIRE(blocks_per_sm > 0);   // 0 => the kernel can't be co-resident (cooperative launch fails)
    const int grid = num_sm * blocks_per_sm;
    std::fprintf(stderr, "GSBENCH_PERSISTENT grid=%d (%d SMs x %d blocks/SM) block=%d\n",
                 grid, num_sm, blocks_per_sm, block);

    // Prewarm is a CORRECTNESS requirement (cold-launch deadlock): force the module resident
    // before the timed cooperative launch.
    if (cfg.prewarm) {
        cudaFuncAttributes fa;
        cudaFuncGetAttributes(&fa, kfn);
    }

    unsigned N = cfg.N, snaps = cfg.snaps, steps_per = cfg.steps_per;
    const uint32_t* d_mask = masker.MaskPtr();
    unsigned ngroups_arg = ngroups;
    void* args[] = {&g.u_curr, &g.v_curr, &g.u_next, &g.v_next,
                    &d_handles, &ngroups_arg,
                    &d_tag_table, &d_mask, const_cast<GsParams*>(&kGs),
                    &N, &snaps, &steps_per};

    ctp::GpuApi::Synchronize();   // settle the seed before timing
    MemSampler mem;
    mem.Start();                  // no-op unless GSBENCH_MEM=1 (persistent bypasses RunSim)
    auto t0 = std::chrono::steady_clock::now();
    cudaError_t lerr = cudaLaunchCooperativeKernel(
        const_cast<void*>(kfn), dim3(grid), dim3(block), args, 0, 0);
    REQUIRE(lerr == cudaSuccess);
    ctp::GpuApi::Synchronize();
    auto t1 = std::chrono::steady_clock::now();
    mem.Stop();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    ClioBdevSync(cfg);   // durability parity (end flush) — matches the other CLIO arms

    for (unsigned gi = 0; gi < ngroups; ++gi)
        groups[gi].ThrowIfIoFailed(("gsbench persistent group " + std::to_string(gi)).c_str());

    FreeGrids(g);
    cudaFree(d_tag_table);
    cudaFree(d_handles);
    if (cfg.read_pdf) {   // GPU-initiated read-back + PDF (one reused group, re-tagged)
        Pdf pdf;
        const bool amode = (cfg.read_async && groups.size() >= 2);
        const double rms = amode ? GpuReadPdfAsync(cfg, groups, tags, cfg.submit_grid(), pdf)
                                 : GpuReadPdf(cfg, groups[0], tags, cfg.submit_grid(), pdf);
        PrintReadResult(async_submit ? "gpuh5" : "gpuh5_sync", cfg, rms, pdf,
                        amode ? "async" : "sync");
    }
    *checksum = ChecksumSnapshots(tags, cfg);
    return ms;
}

// COMPUTE-ONLY isolation of the persistent kernel (fig_write_decomp2.tex's compute-floor probe).
double RunPersistentFloorArm(const Cfg& cfg) {
    Grids g = MakeGrids(cfg.N);

    const void* kfn = reinterpret_cast<const void*>(GsPersistentComputeOnlyKernel);
    int dev = 0; cudaGetDevice(&dev);
    int num_sm = 0;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev);
    const int block = 256;
    int blocks_per_sm = 0;
    REQUIRE(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, kfn, block, 0) == cudaSuccess);
    REQUIRE(blocks_per_sm > 0);
    const int grid = num_sm * blocks_per_sm;
    std::fprintf(stderr, "GSBENCH_PERSISTENT_FLOOR grid=%d (%d SMs x %d blocks/SM) block=%d\n",
                 grid, num_sm, blocks_per_sm, block);

    // Prewarm is a CORRECTNESS requirement for cooperative launches (cold-launch deadlock),
    // same as RunPersistentArm.
    if (cfg.prewarm) {
        cudaFuncAttributes fa;
        cudaFuncGetAttributes(&fa, kfn);
    }

    unsigned N = cfg.N, snaps = cfg.snaps, steps_per = cfg.steps_per;
    // Every I/O arg is unused inside GsPersistentComputeOnlyKernel (see its definition) --
    // null/zero is safe and skips CLIO/dataset/tag-table setup entirely.
    const kvhdf5::GpuDatasetHandle* d_handles = nullptr;
    unsigned ngroups_arg = 0;
    const clio::cte::core::TagId* d_tag_table = nullptr;
    const uint32_t* d_mask = nullptr;
    void* args[] = {&g.u_curr, &g.v_curr, &g.u_next, &g.v_next,
                    &d_handles, &ngroups_arg,
                    &d_tag_table, &d_mask, const_cast<GsParams*>(&kGs),
                    &N, &snaps, &steps_per};

    ctp::GpuApi::Synchronize();   // settle the seed before timing
    auto t0 = std::chrono::steady_clock::now();
    cudaError_t lerr = cudaLaunchCooperativeKernel(
        const_cast<void*>(kfn), dim3(grid), dim3(block), args, 0, 0);
    REQUIRE(lerr == cudaSuccess);
    ctp::GpuApi::Synchronize();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    FreeGrids(g);
    return ms;
}

// Host-driven CLIO arm: the SAME host flow as the raw arm (synchronous D2H into a host
// buffer, then a host-side write call), but the sink is a CLIO PutBlob instead of a
// file write+fsync. It does NOT use the GPU-producer model — no device-side task
// submission, no GpuCteDataset / registered device backends. So comparing it against:
//   - the RAW arm      isolates the STORAGE-PATH cost (CLIO server+bdev vs a plain file);
//   - the SYNC arm     isolates the SUBMISSION MODEL (host-orchestrated vs GPU-producer).
// Durable like CLIO-sync: each PutBlob is waited (bdev write completion).
double RunHostClioArm(const Cfg& cfg, const char* prefix, uint64_t* checksum) {
    const uint64_t gbytes = cfg.grid_bytes();
    const uint64_t chunk_bytes = gbytes / cfg.chunks;

    // One tag per snapshot (distinct path -> tag), same key scheme as the CLIO arms so
    // ChecksumSnapshots reads them back identically. No GPU backends => not bound by the
    // ~16-large-backend ceiling, but we keep cfg.snaps equal for a matched comparison.
    std::vector<clio::cte::core::TagId> tags;
    tags.reserve(cfg.snaps);
    for (unsigned s = 0; s < cfg.snaps; ++s) {
        char path[160];
        std::snprintf(path, sizeof(path), "%s/v/step_%04u", prefix, s);
        tags.push_back(MakeTag(kvhdf5::tagpath::CanonicalTag(path).c_str()));
    }

    // Host staging buffer (shm) that PutBlob DMAs from — the host-side counterpart of raw's
    // pinned D2H buffer.
    ctp::ipc::FullPtr<char> buf = CLIO_CPU_IPC->AllocateBuffer(gbytes);
    REQUIRE(!buf.IsNull());
    g_io_buf_bytes = gbytes;   // single host staging buffer

    // Put this arm's D2H on the same pinned fast path the GPU-producer arms already enjoy (see C...
    bool registered = false;
    if (cfg.hostclio_pin) {
        cudaError_t rc = cudaHostRegister(buf.ptr_, gbytes, cudaHostRegisterDefault);
        registered = (rc == cudaSuccess);
        if (!registered)
            std::fprintf(stderr,
                "[hostclio] WARNING: cudaHostRegister failed (%s); D2H stays on the slow "
                "pageable path and this arm is handicapped vs sync/async\n",
                cudaGetErrorString(rc));
    }
    std::fprintf(stderr, "[hostclio] shm staging buffer registered=%d\n", int(registered));

    Masker masker(cfg.N, cfg.incompressible != 0);
    D2HTrace d2h(EnvU0("GSBENCH_D2H_TRACE", 0) != 0, cfg.snaps, gbytes);
    Grids g = MakeGrids(cfg.N);
    auto snap = [&](unsigned si, float* v_curr) {
        // Same host D2H as raw: pull the (incompressible) current grid into the host buffer.
        d2h.Copy(buf.ptr_, masker.Apply(v_curr));
        // Then persist each chunk to CLIO from the host (synchronous wait == durable).
        for (unsigned c = 0; c < cfg.chunks; ++c) {
            ctp::ipc::ShmPtr<> shm = buf.shm_.template Cast<void>();
            shm.off_ += uint64_t(c) * chunk_bytes;   // point at chunk c within the buffer
            auto t = CLIO_CTE_CLIENT->AsyncPutBlob(tags[si], std::to_string(c),
                                                   clio::run::u64(0), chunk_bytes, shm);
            t.Wait();
            REQUIRE(t->GetReturnCode() == 0);
        }
    };
    auto finalize = [&]() {
        ClioBdevSync(cfg);   // durability parity with raw/hdf5 — see Cfg::clio_fsync
    };
    double ms = RunSim(cfg, g, snap, finalize, nullptr);
    FreeGrids(g);
    d2h.Report("hostclio");
    // Host-side CLIO reader: GetBlob each chunk back into the SAME reused shm staging buffer the...
    if (cfg.read_pdf) {
        Pdf pdf;
        HostReadStage stage(gbytes);
        const double rms = RunReadPhase(cfg, [&](unsigned si) {
            for (unsigned c = 0; c < cfg.chunks; ++c) {
                ctp::ipc::ShmPtr<> shm = buf.shm_.template Cast<void>();
                shm.off_ += uint64_t(c) * chunk_bytes;
                auto t = CLIO_CTE_CLIENT->AsyncGetBlob(tags[si], std::to_string(c),
                                                       clio::run::u64(0), chunk_bytes,
                                                       clio::run::u32(0), shm);
                t.Wait();
                REQUIRE(t->GetReturnCode() == 0);
            }
            stage.Bin(pdf, buf.ptr_);
        });
        PrintReadResult("hostclio", cfg, rms, pdf, "host");
    }
    if (registered) cudaHostUnregister(buf.ptr_);
    *checksum = ChecksumSnapshots(tags, cfg);
    return ms;
}

// ---- raw (no-CLIO) disk arm -----------------------------------------------

void EnsureDir(const std::string& d) {
    if (mkdir(d.c_str(), 0755) != 0 && errno != EEXIST)
        throw std::runtime_error("mkdir " + d);
}
// Write in <=1 MiB O_DIRECT chunks (matches the CLIO bdev's block loop; a single huge
// O_DIRECT pwrite from CUDA-pinned memory hit a ~5x-slow kernel path on this box).
void WriteAllAt(int fd, off_t off, const void* data, size_t bytes) {
    const auto* p = static_cast<const uint8_t*>(data);
    constexpr size_t kBlk = 1u << 20;
    while (bytes) {
        size_t want = bytes < kBlk ? bytes : kBlk;
        ssize_t n = pwrite(fd, p, want, off);
        if (n < 0) { if (errno == EINTR) continue; throw std::runtime_error("pwrite"); }
        p += n; off += n; bytes -= size_t(n);
    }
}

// Fault in the whole [0, bytes) extent of an ALREADY-CREATED file by writing real zeros over...
void PrefaultFile(const std::string& path, uint64_t bytes) {
    int fd = open(path.c_str(), O_WRONLY);   // no O_DIRECT, no O_TRUNC, no O_CREAT
    REQUIRE(fd >= 0);
    constexpr size_t kBlk = 4u << 20;
    std::vector<uint8_t> zeros(kBlk, 0);
    off_t off = 0;
    while (uint64_t(off) < bytes) {
        size_t want = size_t(std::min<uint64_t>(kBlk, bytes - uint64_t(off)));
        ssize_t n = pwrite(fd, zeros.data(), want, off);
        if (n < 0) { if (errno == EINTR) continue; close(fd); throw std::runtime_error("prefault pwrite"); }
        off += n;
    }
    fdatasync(fd);
    close(fd);
}

// A competent (deliberately NOT maximally-tuned) decoupled writer: ONE background thread
// + a small pinned-buffer pool (standard double-buffered checkpoint I/O). It lets disk
// I/O OVERLAP the subsequent sim steps instead of stalling inline — the same structural
// benefit CLIO gets by offloading I/O to its server. No libaio / queue-depth / thread
// fan-out (that would be an "expert" baseline; we keep it fair). O_DIRECT for cache-bypass
// parity with CLIO's kFile bdev.
class BgWriter {
public:
    BgWriter(std::string path, uint64_t gbytes, uint64_t wbytes, unsigned nbuf,
             unsigned snaps, bool odirect, bool do_fsync)
        : gbytes_(gbytes), wbytes_(wbytes), fsync_(do_fsync) {
        // ONE pre-allocated checkpoint file, snapshots written at distinct offsets — same
        // as CLIO's bdev (single truncated file, offset per blob). Avoids the per-snapshot
        // file create/O_TRUNC + async-discard churn that throttled the 12-fresh-files
        // pattern ~5x when the writes were spaced out by GPU work.
        int flags = O_WRONLY | O_CREAT | O_TRUNC | (odirect ? O_DIRECT : 0);
        fd_ = open(path.c_str(), flags, 0644);
        REQUIRE(fd_ >= 0);
        REQUIRE(ftruncate(fd_, off_t(snaps) * off_t(wbytes_)) == 0);  // preallocate size
        bufs_.resize(nbuf);
        for (unsigned i = 0; i < nbuf; ++i) {
            REQUIRE(cudaMallocHost(reinterpret_cast<void**>(&bufs_[i]), wbytes_)
                    == cudaSuccess);                 // pinned: fast D2H
            std::memset(bufs_[i] + gbytes_, 0, wbytes_ - gbytes_);  // O_DIRECT pad tail
            free_.push_back(int(i));
        }
        th_ = std::thread([this] { Run(); });
    }
    // Producer (CUDA thread): grab a free buffer to D2H into (blocks if all in flight).
    uint8_t* Acquire(int* idx) {
        std::unique_lock<std::mutex> lk(m_);
        cv_free_.wait(lk, [this] { return !free_.empty(); });
        int i = free_.back(); free_.pop_back();
        *idx = i; return bufs_[i];
    }
    // Producer: buffer filled (D2H complete) -> hand to the writer.
    void Submit(int idx, unsigned s) {
        { std::lock_guard<std::mutex> lk(m_); work_.push_back({idx, s}); }
        cv_work_.notify_one();
    }
    // Drain + join (call INSIDE the timed region). Yields writer busy-ms. Checksum is
    // computed by the caller AFTER timing (readback), matching the CLIO arms.
    void Finish(double* writer_ms) {
        { std::lock_guard<std::mutex> lk(m_); done_ = true; }
        cv_work_.notify_one();
        th_.join();
        if (fd_ >= 0) close(fd_);
        for (auto* b : bufs_) cudaFreeHost(b);
        *writer_ms = writer_ms_;
        REQUIRE(!err_);
    }
private:
    void Run() {
        for (;;) {
            std::pair<int, unsigned> job;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_work_.wait(lk, [this] { return !work_.empty() || done_; });
                if (work_.empty() && done_) return;
                job = work_.front(); work_.pop_front();
            }
            auto t0 = std::chrono::steady_clock::now();
            uint8_t* buf = bufs_[job.first];
            // NB: checksum is computed AFTER timing (post-run readback, like the CLIO arms)
            // — folding a scalar FNV over 1.9 GB here would add ~6 s to the timed region and
            // unfairly penalize raw vs CLIO (whose ChecksumSnapshots runs post-timing).
            WriteAllAt(fd_, off_t(job.second) * off_t(wbytes_), buf, wbytes_);  // offset slot
            if (fsync_ && fdatasync(fd_) != 0) err_ = true;  // durability parity with CLIO
            writer_ms_ += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            { std::lock_guard<std::mutex> lk(m_); free_.push_back(job.first); }
            cv_free_.notify_one();
        }
    }
    int fd_ = -1; uint64_t gbytes_, wbytes_; bool fsync_ = true;
    std::vector<uint8_t*> bufs_;
    std::mutex m_; std::condition_variable cv_free_, cv_work_;
    std::vector<int> free_; std::deque<std::pair<int, unsigned>> work_;
    bool done_ = false, err_ = false; std::thread th_;
    double writer_ms_ = 0;
};

// Read the persisted checkpoint back and fold FNV in snapshot order — AFTER timing (a
// scalar FNV over 1.9 GB is ~6 s and must NOT sit in the timed region; the CLIO arms
// likewise checksum post-timing).
uint64_t RawReadbackChecksum(const std::string& path, const Cfg& cfg,
                             uint64_t gbytes, uint64_t wbytes) {
    uint64_t h = 1469598103934665603ull;
    int fd = open(path.c_str(), O_RDONLY);
    REQUIRE(fd >= 0);
    std::vector<uint8_t> rb(gbytes);
    for (unsigned s = 0; s < cfg.snaps; ++s) {
        off_t base = off_t(s) * off_t(wbytes);
        size_t got = 0;
        while (got < gbytes) {
            ssize_t n = pread(fd, rb.data() + got, gbytes - got, base + off_t(got));
            REQUIRE(n > 0);
            got += size_t(n);
        }
        h = Fnv1a(rb.data(), gbytes, h);
    }
    close(fd);
    return h;
}

// raw reader: pread each snapshot back, stage H2D, bin into the PDF (same GPU histogram
// every other arm uses, so only the READ path differs between arms).
double RawReadPdf(const std::string& path, const Cfg& cfg,
                  uint64_t gbytes, uint64_t wbytes, Pdf& pdf) {
    int fd = open(path.c_str(), O_RDONLY);
    REQUIRE(fd >= 0);
    std::vector<uint8_t> rb(gbytes);
    HostReadStage stage(gbytes);
    const double ms = RunReadPhase(cfg, [&](unsigned s) {
        const off_t base = off_t(s) * off_t(wbytes);
        size_t got = 0;
        while (got < gbytes) {
            ssize_t n = pread(fd, rb.data() + got, gbytes - got, base + off_t(got));
            REQUIRE(n > 0);
            got += size_t(n);
        }
        stage.Bin(pdf, rb.data());
    });
    close(fd);
    return ms;
}

// Raw arm (no CLIO): identical sim + timed loop; each snapshot D2Hs the (incompressible)
// v-grid to host and persists it into one pre-allocated file. Two structures selected by
// GSBENCH_RAW_INLINE:
//   0 = background writer thread — I/O overlaps the next sim steps (the natural design, but
//       this box throttles GPU-concurrent disk I/O ~5x, penalizing the overlap);
//   1 = inline synchronous — GPU idle during the write, matching host-CLIO / sync-CLIO, for
//       a storage-path comparison free of that throttle.
double RunRawArm(const Cfg& cfg, uint64_t* checksum) {
    EnsureDir(cfg.disk_dir);
    const uint64_t gbytes = cfg.grid_bytes();
    constexpr uint64_t kAlign = 4096;
    const uint64_t wbytes = (gbytes + kAlign - 1) & ~(kAlign - 1);  // O_DIRECT length
    const std::string path = cfg.disk_dir + "/checkpoint.bin";
    Masker masker(cfg.N, cfg.incompressible != 0);
    D2HTrace d2h(EnvU0("GSBENCH_D2H_TRACE", 0) != 0, cfg.snaps, gbytes);

    if (cfg.raw_inline) {
        int flags = O_WRONLY | O_CREAT | O_TRUNC | (cfg.raw_odirect ? O_DIRECT : 0);
        int fd = open(path.c_str(), flags, 0644);
        REQUIRE(fd >= 0);
        REQUIRE(ftruncate(fd, off_t(cfg.snaps) * off_t(wbytes)) == 0);
        // ftruncate only sizes the file; the pages are still sparse. Fault them in now, off
        // the clock, so we time the write and not the kernel's page allocator (Cfg::prefault).
        if (cfg.prefault) PrefaultFile(path, uint64_t(cfg.snaps) * wbytes);
        uint8_t* buf = nullptr;
        REQUIRE(cudaMallocHost(reinterpret_cast<void**>(&buf), wbytes) == cudaSuccess);
        std::memset(buf + gbytes, 0, wbytes - gbytes);
        g_io_buf_bytes = wbytes;   // single inline staging buffer
        Grids g = MakeGrids(cfg.N);
        double write_ms = 0;
        auto snap = [&](unsigned si, float* v_curr) {
            d2h.Copy(buf, masker.Apply(v_curr));
            auto a = std::chrono::steady_clock::now();          // GPU idle during this write
            WriteAllAt(fd, off_t(si) * off_t(wbytes), buf, wbytes);
            if (cfg.raw_fsync) fdatasync(fd);
            write_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - a).count();
        };
        auto finalize = [&]() {};
        double ms = RunSim(cfg, g, snap, finalize, nullptr);
        FreeGrids(g);
        d2h.Report("raw_inline");
        close(fd);
        cudaFreeHost(buf);
        std::fprintf(stderr, "[raw] total=%.1f ms  write=%.1f ms  (inline, GPU idle)\n",
                     ms, write_ms);
        if (cfg.read_pdf) {
            Pdf pdf;
            const double rms = RawReadPdf(path, cfg, gbytes, wbytes, pdf);
            PrintReadResult("raw_inline", cfg, rms, pdf, "host");
        }
        *checksum = RawReadbackChecksum(path, cfg, gbytes, wbytes);
        return ms;
    }

    g_io_buf_bytes = uint64_t(3) * wbytes;   // BgWriter's nbuf=3 pinned staging buffers
    BgWriter writer(path, gbytes, wbytes, /*nbuf=*/3, cfg.snaps,
                    /*odirect=*/cfg.raw_odirect != 0, /*do_fsync=*/cfg.raw_fsync != 0);
    // The ctor has open()ed + ftruncate()d the file; fault its pages in through a separate
    // non-O_DIRECT fd, still BEFORE RunSim starts the clock (Cfg::prefault).
    if (cfg.prefault) PrefaultFile(path, uint64_t(cfg.snaps) * wbytes);
    Grids g = MakeGrids(cfg.N);
    auto snap = [&](unsigned si, float* v_curr) {
        int idx;
        uint8_t* buf = writer.Acquire(&idx);
        // SYNCHRONOUS D2H on the DEFAULT stream (ordered after the step / mask kernel), then
        // hand the buffer to the writer, which writes it while the NEXT sim steps run.
        d2h.Copy(buf, masker.Apply(v_curr));
        writer.Submit(idx, si);
    };
    double writer_ms = 0;
    auto finalize = [&]() { writer.Finish(&writer_ms); };  // drain inside timed region
    double ms = RunSim(cfg, g, snap, finalize, nullptr);
    FreeGrids(g);
    d2h.Report("raw_threaded");
    std::fprintf(stderr,
        "[raw] total=%.1f ms  writer-busy=%.1f ms (overlapped with compute)  nbuf=3\n",
        ms, writer_ms);
    if (cfg.read_pdf) {
        Pdf pdf;
        const double rms = RawReadPdf(path, cfg, gbytes, wbytes, pdf);
        PrintReadResult("raw_threaded", cfg, rms, pdf, "host");
    }
    *checksum = RawReadbackChecksum(path, cfg, gbytes, wbytes);
    return ms;
}

// ---- hdf5 arm (the conventional path: GPU -> D2H -> HDF5) -------------------
#if GSBENCH_HAVE_HDF5

#define H5CHK(expr)                                                        \
    do {                                                                   \
        if ((expr) < 0) {                                                  \
            H5Eprint2(H5E_DEFAULT, stderr);                                \
            throw std::runtime_error("HDF5 call failed: " #expr);          \
        }                                                                  \
    } while (0)

// The HDF5 side of the arm: one file, one CHUNKED dataset per snapshot at /v/step_NNNN, chun...
class Hdf5Sink {
public:
    explicit Hdf5Sink(const Cfg& cfg)
        : cfg_(cfg),
          rows_(cfg.N / cfg.chunks),
          chunk_bytes_(uint64_t(cfg.N / cfg.chunks) * cfg.N * sizeof(float)) {}

    void Open(const std::string& path) {
        // sec2 (the default driver): one buffered fd — the same kernel path the raw arm's
        // pwrite() takes. Swapping in core/direct would change what is being measured.
        hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
        H5CHK(fapl);
        stdio_ = (cfg_.hdf5_vfd == "stdio");
        if (stdio_) H5CHK(H5Pset_fapl_stdio(fapl));
        else        H5CHK(H5Pset_fapl_sec2(fapl));
        if (cfg_.hdf5_meta_block_kb)
            H5CHK(H5Pset_meta_block_size(fapl, hsize_t(cfg_.hdf5_meta_block_kb) << 10));
        file_ = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        H5CHK(file_);
        H5CHK(H5Pclose(fapl));
        grp_ = H5Gcreate2(file_, "/v", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5CHK(grp_);
        // DURABILITY PARITY.
        void* h = nullptr;
        H5CHK(H5Fget_vfd_handle(file_, H5P_DEFAULT, &h));
        // H5Fget_vfd_handle yields a pointer TO the driver's handle, for both drivers: int* for sec2...
        if (stdio_) fp_ = h ? *static_cast<FILE**>(h) : nullptr;
        else        fd_ = h ? *static_cast<int*>(h) : -1;
    }

    // Push this snapshot all the way to the device, matching raw's per-snapshot fdatasync.
    void Sync() {
        H5CHK(H5Fflush(file_, H5F_SCOPE_GLOBAL));   // HDF5's caches -> the driver
        if (stdio_) {
            if (fp_) { std::fflush(fp_); fdatasync(fileno(fp_)); }
        } else if (fd_ >= 0) {
            fdatasync(fd_);
        }
    }

    // Create all `snaps` datasets up front and hold them open. Called OUTSIDE the timed
    // region (see Cfg::hdf5_precreate for why that is parity rather than a favour).
    void PrecreateAll() {
        if (!cfg_.hdf5_precreate) return;
        dsets_.resize(cfg_.snaps, -1);
        for (unsigned s = 0; s < cfg_.snaps; ++s) dsets_[s] = CreateDataset(s);
    }

    // PAGE-FAULT PARITY (Cfg::prefault).
    void PrefaultAll(const void* zeros) {
        if (!cfg_.prefault || !cfg_.hdf5_precreate) return;
        for (unsigned s = 0; s < cfg_.snaps; ++s) WriteSnap(s, zeros);
    }

    // Persist one snapshot from a host buffer holding the full N*N masked grid.
    void WriteSnap(unsigned si, const void* host) {
        const bool pre = cfg_.hdf5_precreate != 0;
        hid_t dset = pre ? dsets_[si] : CreateDataset(si);

        // Naive forces the high-level H5Dwrite of the whole array (H5Dwrite_chunk is illegal on
        // a contiguous dataset anyway).
        if (cfg_.hdf5_direct_chunk && !cfg_.hdf5_naive) {
            const auto* p = static_cast<const uint8_t*>(host);
            for (unsigned c = 0; c < cfg_.chunks; ++c) {
                hsize_t off[2] = {hsize_t(c) * hsize_t(rows_), 0};
                H5CHK(H5Dwrite_chunk(dset, H5P_DEFAULT, /*filter_mask=*/0, off,
                                     size_t(chunk_bytes_),
                                     p + uint64_t(c) * chunk_bytes_));
            }
        } else {
            H5CHK(H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, host));
        }
        if (!pre) H5CHK(H5Dclose(dset));
        if (cfg_.raw_fsync) Sync();   // durability parity with raw's per-snapshot fdatasync
    }

    // ---- async-VOL variant (arm `hdf5_async`) ------------------------------ Issue snapshot si...
    void WriteSnapAsync(unsigned si, const void* host, hid_t es) {
        const bool pre = cfg_.hdf5_precreate != 0;
        hid_t dset = pre ? dsets_[si] : CreateDatasetAsync(si, es);
        H5CHK(H5Dwrite_async(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, host,
                             es));
        if (!pre) H5CHK(H5Dclose_async(dset, es));
    }

    void Close() {
        for (hid_t d : dsets_) if (d >= 0) H5Dclose(d);
        dsets_.clear();
        if (grp_ >= 0) { H5Gclose(grp_); grp_ = -1; }
        if (file_ >= 0) { H5Fclose(file_); file_ = -1; }
    }

    // Is the async VOL actually the connector on this file, or did we silently fall back to
    // native (which would make the arm a mislabelled copy of the inline one)?
    bool AsyncVolActive() const {
        char name[64] = {0};
        ssize_t n = H5VLget_connector_name(file_, name, sizeof(name));
        return n > 0 && std::strcmp(name, "async") == 0;
    }

    // vol-async's OWN drain API (H5Fwait, "gov.lbl.async.file.wait"), reached through the public...
    bool FileWait() const {
        static int op = -2;   // -2 = not looked up yet, -1 = unavailable
        if (op == -2) {
            if (H5VLfind_opt_operation(H5VL_SUBCLS_FILE, "gov.lbl.async.file.wait", &op) < 0)
                op = -1;
        }
        if (op < 0) return false;
        H5VL_optional_args_t args;
        args.op_type = op;
        args.args = nullptr;
        return H5VLfile_optional_op(file_, &args, H5P_DEFAULT, H5ES_NONE) >= 0;
    }

private:
    // Property lists for snapshot si's dataset. Shared by the sync and async create paths so
    // both arms get byte-identical chunk geometry and tuning.
    void MakeDatasetPlists(hid_t* space, hid_t* dcpl, hid_t* dapl) {
        const hsize_t dims[2] = {hsize_t(cfg_.N), hsize_t(cfg_.N)};
        const hsize_t cdims[2] = {hsize_t(rows_), hsize_t(cfg_.N)};

        *space = H5Screate_simple(2, dims, nullptr);
        H5CHK(*space);
        *dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5CHK(*dcpl);
        // Naive contiguous: leave the DCPL at its default -> CONTIGUOUS layout, no fill/alloc tuning...
        const bool chunk_it = !cfg_.hdf5_naive || cfg_.hdf5_naive_chunked;
        if (chunk_it)
            H5CHK(H5Pset_chunk(*dcpl, 2, cdims));   // geometry parity with the CLIO arms
        if (!cfg_.hdf5_stock && !cfg_.hdf5_naive) {
            H5CHK(H5Pset_fill_time(*dcpl, H5D_FILL_TIME_NEVER));
            if (cfg_.hdf5_early_alloc)
                H5CHK(H5Pset_alloc_time(*dcpl, H5D_ALLOC_TIME_EARLY));
        }

        *dapl = H5P_DEFAULT;
        if (!cfg_.hdf5_stock && !cfg_.hdf5_naive) {
            *dapl = H5Pcreate(H5P_DATASET_ACCESS);
            H5CHK(*dapl);
            // nslots wants to be prime-ish and comfortably > the chunk count (HDF5 hashes chunk index ->...
            size_t nslots = size_t(cfg_.chunks) * 10 + 1;
            if (nslots < 521) nslots = 521;
            const size_t nbytes = size_t(cfg_.hdf5_rdcc_mb) << 20;
            // w0=1.0: fully-written chunks are the preferred eviction victims — we never
            // read a chunk back during the run, so keeping one resident buys nothing.
            H5CHK(H5Pset_chunk_cache(*dapl, nslots, nbytes, 1.0));
        }
    }

    hid_t CreateDatasetAsync(unsigned si, hid_t es) {
        hid_t space, dcpl, dapl;
        MakeDatasetPlists(&space, &dcpl, &dapl);
        char name[64];
        std::snprintf(name, sizeof(name), "step_%04u", si);
        hid_t dset = H5Dcreate_async(grp_, name, H5T_IEEE_F32LE, space, H5P_DEFAULT, dcpl,
                                     dapl, es);
        H5CHK(dset);
        if (dapl != H5P_DEFAULT) H5CHK(H5Pclose(dapl));
        H5CHK(H5Pclose(dcpl));
        H5CHK(H5Sclose(space));
        return dset;
    }

    hid_t CreateDataset(unsigned si) {
        hid_t space, dcpl, dapl;
        MakeDatasetPlists(&space, &dcpl, &dapl);
        char name[64];
        std::snprintf(name, sizeof(name), "step_%04u", si);
        // File type IEEE_F32LE == the native x86 float, so HDF5 takes its no-conversion (memcpy) pat...
        hid_t dset = H5Dcreate2(grp_, name, H5T_IEEE_F32LE, space, H5P_DEFAULT, dcpl, dapl);
        H5CHK(dset);
        if (dapl != H5P_DEFAULT) H5CHK(H5Pclose(dapl));
        H5CHK(H5Pclose(dcpl));
        H5CHK(H5Sclose(space));
        return dset;
    }

    std::vector<hid_t> dsets_;   // empty unless precreate
    const Cfg& cfg_;
    unsigned rows_;
    uint64_t chunk_bytes_;
    hid_t file_ = -1, grp_ = -1;
    bool stdio_ = false;
    int fd_ = -1;        // sec2
    FILE* fp_ = nullptr; // stdio
};

// Background HDF5 writer: the structural twin of the raw arm's BgWriter (one thread, a small...
uint8_t* Hdf5AllocBuf(const Cfg& cfg, uint64_t bytes) {
    uint8_t* p = nullptr;
    if (cfg.hdf5_pinned) {
        REQUIRE(cudaMallocHost(reinterpret_cast<void**>(&p), bytes) == cudaSuccess);
    } else {
        p = static_cast<uint8_t*>(std::malloc(bytes));
        REQUIRE(p != nullptr);
    }
    return p;
}
void Hdf5FreeBuf(const Cfg& cfg, uint8_t* p) {
    if (!p) return;
    if (cfg.hdf5_pinned) cudaFreeHost(p); else std::free(p);
}

class Hdf5Writer {
public:
    Hdf5Writer(const Cfg& cfg, std::string path, uint64_t gbytes, unsigned nbuf)
        : cfg_(cfg), path_(std::move(path)), gbytes_(gbytes) {
        bufs_.resize(nbuf);
        for (unsigned i = 0; i < nbuf; ++i) {
            bufs_[i] = Hdf5AllocBuf(cfg_, gbytes_);
            free_.push_back(int(i));
        }
        th_ = std::thread([this] { Run(); });
        // BLOCK until the thread has opened the file and pre-created the datasets.
        std::string err;
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_ready_.wait(lk, [this] { return ready_; });
            err = err_;
        }
        if (!err.empty()) {
            th_.join();   // else ~thread on a joinable thread => std::terminate
            throw std::runtime_error("hdf5 writer open: " + err);
        }
    }
    uint8_t* Acquire(int* idx) {
        std::unique_lock<std::mutex> lk(m_);
        cv_free_.wait(lk, [this] { return !free_.empty(); });
        int i = free_.back(); free_.pop_back();
        *idx = i; return bufs_[i];
    }
    void Submit(int idx, unsigned s) {
        { std::lock_guard<std::mutex> lk(m_); work_.push_back({idx, s}); }
        cv_work_.notify_one();
    }
    // Drain + join + close the file. Called INSIDE the timed region (raw does the same).
    void Finish(double* writer_ms) {
        { std::lock_guard<std::mutex> lk(m_); done_ = true; }
        cv_work_.notify_one();
        th_.join();
        for (auto* b : bufs_) Hdf5FreeBuf(cfg_, b);
        *writer_ms = writer_ms_;
        if (!err_.empty()) throw std::runtime_error("hdf5 writer thread: " + err_);
    }
private:
    // Signal the constructor that setup finished (successfully or not) exactly once.
    void SignalReady() {
        { std::lock_guard<std::mutex> lk(m_); ready_ = true; }
        cv_ready_.notify_all();
    }
    void Run() {
        Hdf5Sink sink(cfg_);
        try {
            sink.Open(path_);
            sink.PrecreateAll();   // outside the timed region: ctor blocks until we get here
            // Same window, same reason: fault in the file's pages (Cfg::prefault).
            if (cfg_.prefault && !bufs_.empty()) {
                std::memset(bufs_[0], 0, gbytes_);
                sink.PrefaultAll(bufs_[0]);
            }
            SignalReady();
            for (;;) {
                std::pair<int, unsigned> job;
                {
                    std::unique_lock<std::mutex> lk(m_);
                    cv_work_.wait(lk, [this] { return !work_.empty() || done_; });
                    if (work_.empty() && done_) break;
                    job = work_.front(); work_.pop_front();
                }
                auto t0 = std::chrono::steady_clock::now();
                sink.WriteSnap(job.second, bufs_[job.first]);
                writer_ms_ += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                { std::lock_guard<std::mutex> lk(m_); free_.push_back(job.first); }
                cv_free_.notify_one();
            }
            sink.Close();   // file close is part of the timed region, as spec'd
        } catch (const std::exception& e) {
            { std::lock_guard<std::mutex> lk(m_); err_ = e.what();
              // Never strand the producer waiting on a buffer that will never come back.
              for (size_t i = 0; i < bufs_.size(); ++i) free_.push_back(int(i)); }
            cv_free_.notify_all();
            SignalReady();   // no-op if setup already succeeded; unblocks the ctor if not
        }
    }
    const Cfg& cfg_;
    std::string path_;
    uint64_t gbytes_;
    std::vector<uint8_t*> bufs_;
    std::mutex m_; std::condition_variable cv_free_, cv_work_, cv_ready_;
    std::vector<int> free_; std::deque<std::pair<int, unsigned>> work_;
    bool done_ = false, ready_ = false; std::string err_; std::thread th_;
    double writer_ms_ = 0;
};

// Read every snapshot dataset back out of the file and fold FNV in snapshot order — AFTER th...
uint64_t Hdf5ReadbackChecksum(const std::string& path, const Cfg& cfg, uint64_t gbytes) {
    uint64_t h = 1469598103934665603ull;
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    H5CHK(file);
    std::vector<uint8_t> rb(gbytes);
    for (unsigned s = 0; s < cfg.snaps; ++s) {
        char name[80];
        std::snprintf(name, sizeof(name), "/v/step_%04u", s);
        hid_t dset = H5Dopen2(file, name, H5P_DEFAULT);
        H5CHK(dset);
        H5CHK(H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, rb.data()));
        H5CHK(H5Dclose(dset));
        h = Fnv1a(rb.data(), gbytes, h);
    }
    H5CHK(H5Fclose(file));
    return h;
}

// hdf5 reader: H5Dread each snapshot back, stage H2D, bin into the PDF.
double Hdf5ReadPdf(const std::string& path, const Cfg& cfg, uint64_t gbytes, Pdf& pdf) {
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    H5CHK(file);
    std::vector<uint8_t> rb(gbytes);
    HostReadStage stage(gbytes);
    const double ms = RunReadPhase(cfg, [&](unsigned s) {
        char name[80];
        std::snprintf(name, sizeof(name), "/v/step_%04u", s);
        hid_t dset = H5Dopen2(file, name, H5P_DEFAULT);
        H5CHK(dset);
        H5CHK(H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, rb.data()));
        H5CHK(H5Dclose(dset));
        stage.Bin(pdf, rb.data());
    });
    H5CHK(H5Fclose(file));
    return ms;
}

// The conventional path, and the paper's real baseline: GPU compute -> cudaMemcpy D2H -> HDF...
double RunHdf5Arm(const Cfg& cfg_in, uint64_t* checksum) {
    // Local copy: PREFAULT IMPLIES PRECREATE for this arm (Hdf5Sink::PrefaultAll explains why — ...
    Cfg cfg = cfg_in;
    if (cfg.prefault) cfg.hdf5_precreate = 1;

    EnsureDir(cfg.hdf5_dir);
    const uint64_t gbytes = cfg.grid_bytes();
    const std::string path = cfg.hdf5_dir + "/checkpoint.h5";
    Masker masker(cfg.N, cfg.incompressible != 0);
    D2HTrace d2h(EnvU0("GSBENCH_D2H_TRACE", 0) != 0, cfg.snaps, gbytes);

    const char* mode = cfg.hdf5_naive ? (cfg.hdf5_naive_chunked ? "NAIVE (chunked, untuned)"
                                                                : "NAIVE (contiguous, untuned)")
                     : cfg.hdf5_stock ? "STOCK (untuned)" : "tuned";
    std::fprintf(stderr,
        "[hdf5] %s: vfd=%s rdcc=%uMB early_alloc=%u direct_chunk=%u precreate=%u "
        "metablk=%uKB fsync=%u chunk=%ux%u\n",
        mode, cfg.hdf5_vfd.c_str(), cfg.hdf5_rdcc_mb, cfg.hdf5_early_alloc,
        cfg.hdf5_direct_chunk, cfg.hdf5_precreate, cfg.hdf5_meta_block_kb,
        cfg.raw_fsync, cfg.N / cfg.chunks, cfg.N);

    if (cfg.raw_inline) {
        Hdf5Sink sink(cfg);
        sink.Open(path);
        sink.PrecreateAll();      // outside the timed region (parity — see Cfg::hdf5_precreate)
        uint8_t* buf = Hdf5AllocBuf(cfg, gbytes);
        g_io_buf_bytes = gbytes;   // single inline staging buffer
        if (cfg.prefault) {       // also outside the timed region (Cfg::prefault)
            std::memset(buf, 0, gbytes);
            sink.PrefaultAll(buf);
        }
        Grids g = MakeGrids(cfg.N);
        double write_ms = 0;
        auto snap = [&](unsigned si, float* v_curr) {
            d2h.Copy(buf, masker.Apply(v_curr));
            auto a = std::chrono::steady_clock::now();      // GPU idle during this write
            sink.WriteSnap(si, buf);
            write_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - a).count();
        };
        auto finalize = [&]() { sink.Close(); };
        double ms = RunSim(cfg, g, snap, finalize, nullptr);
        FreeGrids(g);
        d2h.Report("hdf5_inline");
        Hdf5FreeBuf(cfg, buf);
        std::fprintf(stderr, "[hdf5] total=%.1f ms  write=%.1f ms  (inline, GPU idle)\n",
                     ms, write_ms);
        if (cfg.read_pdf) {
            Pdf pdf;
            const double rms = Hdf5ReadPdf(path, cfg, gbytes, pdf);
            PrintReadResult("hdf5_inline", cfg, rms, pdf, "host");
        }
        *checksum = Hdf5ReadbackChecksum(path, cfg, gbytes);
        return ms;
    }

    g_io_buf_bytes = uint64_t(cfg.hdf5_nbuf) * gbytes;   // nbuf pinned staging buffers
    Hdf5Writer writer(cfg, path, gbytes, /*nbuf=*/cfg.hdf5_nbuf);
    Grids g = MakeGrids(cfg.N);
    auto snap = [&](unsigned si, float* v_curr) {
        int idx;
        uint8_t* buf = writer.Acquire(&idx);
        d2h.Copy(buf, masker.Apply(v_curr));
        writer.Submit(idx, si);
    };
    double writer_ms = 0;
    auto finalize = [&]() { writer.Finish(&writer_ms); };   // drain inside timed region
    double ms = RunSim(cfg, g, snap, finalize, nullptr);
    FreeGrids(g);
    d2h.Report("hdf5_threaded");
    std::fprintf(stderr,
        "[hdf5] total=%.1f ms  writer-busy=%.1f ms (overlapped with compute)  nbuf=%u\n",
        ms, writer_ms, cfg.hdf5_nbuf);
    if (cfg.read_pdf) {
        Pdf pdf;
        const double rms = Hdf5ReadPdf(path, cfg, gbytes, pdf);
        PrintReadResult("hdf5_threaded", cfg, rms, pdf, "host");
    }
    *checksum = Hdf5ReadbackChecksum(path, cfg, gbytes);
    return ms;
}

// Arm `hdf5_async`: the same conventional path, but through the HDF5 Asynchronous I/O VOL co...
double RunHdf5AsyncArm(const Cfg& cfg_in, uint64_t* checksum) {
    // Prefault implies precreate here too (see RunHdf5Arm / Hdf5Sink::PrefaultAll).
    Cfg cfg = cfg_in;
    if (cfg.prefault) cfg.hdf5_precreate = 1;

    EnsureDir(cfg.hdf5_dir);
    const uint64_t gbytes = cfg.grid_bytes();
    const std::string path = cfg.hdf5_dir + "/checkpoint_async.h5";
    Masker masker(cfg.N, cfg.incompressible != 0);

    Hdf5Sink sink(cfg);
    sink.Open(path);
    if (!sink.AsyncVolActive()) {
        sink.Close();
        throw std::runtime_error(
            "hdf5_async: the async VOL connector is NOT loaded (H5VLget_connector_name != "
            "\"async\"). Set HDF5_VOL_CONNECTOR=\"async under_vol=0;under_info={}\", "
            "HDF5_PLUGIN_PATH=<vol-async lib>, and LD_LIBRARY_PATH to a THREAD-SAFE libhdf5 "
            "+ Argobots. Refusing to report a synchronous run as async.");
    }
    std::fprintf(stderr, "[hdf5_async] async VOL connector CONFIRMED loaded\n");

    // GSBENCH_HDF5_ASYNC_POOL=<M>: bound the staging footprint to M buffers, snapshot si writing...
    const unsigned pool = EnvU0("GSBENCH_HDF5_ASYNC_POOL", 0);
    // M >= snaps is pooling that never wraps, i.e. exactly the unpooled arm; clamp so we do not
    // allocate (and report) buffers no snapshot will ever touch.
    const unsigned nbuf = pool ? std::min(pool, cfg.snaps) : cfg.snaps;
    // One pinned buffer per slot (see the buffer contract above).
    std::vector<uint8_t*> bufs(nbuf, nullptr);
    for (unsigned s = 0; s < nbuf; ++s) bufs[s] = Hdf5AllocBuf(cfg, gbytes);
    // Deterministic footprint: our nbuf staging buffers -- unbounded (== snaps, like gpuh5_noreu...
    g_io_buf_bytes = uint64_t(nbuf) * gbytes;

    sink.PrecreateAll();                 // outside the timed region
    if (cfg.prefault) {                  // ditto — synchronous zero pass (Cfg::prefault)
        std::memset(bufs[0], 0, gbytes);
        sink.PrefaultAll(bufs[0]);
    }

    hid_t es = H5EScreate();
    H5CHK(es);

    // Instrumentation (GSBENCH_HDF5_ASYNC_TRACE=1): where does the time actually go?
    const bool trace_async = EnvU0("GSBENCH_HDF5_ASYNC_TRACE", 0) != 0;
    // Drain the event set after every snapshot instead of once at the end. Bounds the
    // connector's memory growth and lets each write overlap the NEXT snapshot's compute.
    const unsigned drain_every = EnvU0("GSBENCH_HDF5_ASYNC_DRAIN_EVERY", 0);
    // Drain through the connector's own H5Fwait before H5ESwait (see Hdf5Sink::FileWait).
    const bool use_fwait = EnvU0("GSBENCH_HDF5_ASYNC_FWAIT", 1) != 0;
    double t_d2h = 0, t_fire = 0, t_wait = 0, t_sync = 0;
    using Clk = std::chrono::steady_clock;
    auto secs = [](Clk::time_point a, Clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto drain = [&]() {
        if (use_fwait) sink.FileWait();   // no-op + fall through to H5ESwait if unsupported
        size_t n_in_progress = 0;
        hbool_t err_occurred = 0;
        H5CHK(H5ESwait(es, H5ES_WAIT_FOREVER, &n_in_progress, &err_occurred));
        if (err_occurred)
            throw std::runtime_error("hdf5_async: an async HDF5 operation failed");
    };

    // t_d2h below is the ORIGINAL, CONFLATED figure: a host clock around a cudaMemcpy that is qu...
    D2HTrace d2h(EnvU0("GSBENCH_D2H_TRACE", 0) != 0, cfg.snaps, gbytes);
    Grids g = MakeGrids(cfg.N);
    auto snap = [&](unsigned si, float* v_curr) {
        // MANDATORY pre-reuse drain (pooled only; never taken when nbuf == cfg.snaps, so the unpoole...
        if (si >= nbuf) {
            auto w0 = Clk::now();
            drain();
            t_wait += secs(w0, Clk::now());
        }
        // Same D2H as raw/hdf5/hostclio, into THIS snapshot's slot.
        uint8_t* buf = bufs[si % nbuf];
        auto a = Clk::now();
        d2h.Copy(buf, masker.Apply(v_curr));
        auto b = Clk::now();
        sink.WriteSnapAsync(si, buf, es);        // returns immediately; VOL drains it
        auto c = Clk::now();
        if (drain_every && (si + 1) % drain_every == 0) drain();
        auto d = Clk::now();
        t_d2h += secs(a, b);
        t_fire += secs(b, c);
        t_wait += secs(c, d);
    };
    auto finalize = [&]() {                      // tail-drain, inside the timed region
        auto a = Clk::now();
        drain();
        auto b = Clk::now();
        if (cfg.raw_fsync) sink.Sync();
        auto c = Clk::now();
        t_wait += secs(a, b);
        t_sync += secs(b, c);
    };
    double ms = RunSim(cfg, g, snap, finalize, nullptr);
    FreeGrids(g);
    d2h.Report("hdf5_async");
    if (trace_async)
        std::fprintf(stderr,
                     "[hdf5_async] BREAKDOWN d2h=%.1f(CONFLATED: includes compute tail) "
                     "fire=%.1f wait=%.1f sync=%.1f other(compute)=%.1f ms\n",
                     t_d2h, t_fire, t_wait, t_sync,
                     ms - t_d2h - t_fire - t_wait - t_sync);

    H5CHK(H5ESclose(es));
    sink.Close();
    for (auto* b : bufs) Hdf5FreeBuf(cfg, b);
    // The GSBENCH_RESULT arm name stays "async_VOL" whether or not pooling is on.
    std::fprintf(stderr,
                 "[hdf5_async] total=%.1f ms  (event-set fire-all + drain-at-end, "
                 "bufs=%u%s)\n",
                 ms, nbuf, pool ? " POOLED: full drain before every slot reuse" : "");
    if (cfg.read_pdf) {
        Pdf pdf;
        const double rms = Hdf5ReadPdf(path, cfg, gbytes, pdf);
        PrintReadResult("async_VOL", cfg, rms, pdf, "host");
    }
    *checksum = Hdf5ReadbackChecksum(path, cfg, gbytes);
    return ms;
}

#endif  // GSBENCH_HAVE_HDF5

}  // namespace

// ---- kernel register-usage report (no simulation, no CLIO server) ----------...
TEST_CASE("GSBENCH kernel register report",
          "[.][gpu][gsbench][gsbench_kernel_regs]") {
    struct KernelInfo {
        const char* name;
        const char* arm;
        const void* fn;
    };
    const KernelInfo kernels[] = {
        {"GsStepKernel", "all", reinterpret_cast<const void*>(GsStepKernel)},
        {"MaskKernel", "all", reinterpret_cast<const void*>(MaskKernel)},
        {"TwCopyKernel", "clio-sync,clio-async",
         reinterpret_cast<const void*>(TwCopyKernel)},
        {"TwSnapSyncKernel", "clio-sync",
         reinterpret_cast<const void*>(TwSnapSyncKernel<false>)},
        {"TwSnapFireKernel", "clio-async",
         reinterpret_cast<const void*>(TwSnapFireKernel<false>)},
        {"TwDrainKernel", "clio-async",
         reinterpret_cast<const void*>(TwDrainKernel)},
        {"GsPersistentKernel<true>", "gpuh5",
         reinterpret_cast<const void*>(GsPersistentKernel<true>)},
        {"GsPersistentKernel<false>", "gpuh5_sync",
         reinterpret_cast<const void*>(GsPersistentKernel<false>)},
        {"GsPersistentComputeOnlyKernel", "gpuh5-decomposition-probe (never launched)",
         reinterpret_cast<const void*>(GsPersistentComputeOnlyKernel)},
        {"ReuseFireKernel", "reuse",
         reinterpret_cast<const void*>(ReuseFireKernel)},
        {"TwSnapPooledKernel<false>", "pooled",
         reinterpret_cast<const void*>(TwSnapPooledKernel<false>)},
    };
    const unsigned kNumKernels = sizeof(kernels) / sizeof(kernels[0]);
    int regs[kNumKernels];
    for (unsigned i = 0; i < kNumKernels; ++i) {
        cudaFuncAttributes attr{};
        REQUIRE(cudaFuncGetAttributes(&attr, kernels[i].fn) == cudaSuccess);
        regs[i] = attr.numRegs;
        std::fprintf(stderr,
            "GSBENCH_KERNEL_REGS kernel=%s arm=%s regs=%d smem=%zu lmem=%zu "
            "maxThreadsPerBlock=%d\n",
            kernels[i].name, kernels[i].arm, attr.numRegs, attr.sharedSizeBytes,
            attr.localSizeBytes, attr.maxThreadsPerBlock);
    }

    // ---- theoretical occupancy: persistent kernel vs the stock stencil, @256 threads/block,
    // sm_89 (max 1536 threads/SM -> occupancy% = blocks_per_sm * 256 / 1536). ----
    {
        int blocks = 0;
        REQUIRE(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &blocks, GsStepKernel, 256, 0) == cudaSuccess);
        std::fprintf(stderr,
            "GSBENCH_KERNEL_OCCUPANCY kernel=GsStepKernel blockDim=256 blocks_per_sm=%d "
            "theoretical_occupancy_pct=%.1f\n",
            blocks, 100.0 * double(blocks) * 256.0 / 1536.0);

        blocks = 0;
        REQUIRE(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &blocks, GsPersistentKernel<true>, 256, 0) == cudaSuccess);
        std::fprintf(stderr,
            "GSBENCH_KERNEL_OCCUPANCY kernel=GsPersistentKernel<true> blockDim=256 "
            "blocks_per_sm=%d theoretical_occupancy_pct=%.1f\n",
            blocks, 100.0 * double(blocks) * 256.0 / 1536.0);

        blocks = 0;
        REQUIRE(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &blocks, GsPersistentComputeOnlyKernel, 256, 0) == cudaSuccess);
        std::fprintf(stderr,
            "GSBENCH_KERNEL_OCCUPANCY kernel=GsPersistentComputeOnlyKernel blockDim=256 "
            "blocks_per_sm=%d theoretical_occupancy_pct=%.1f\n",
            blocks, 100.0 * double(blocks) * 256.0 / 1536.0);
    }
    // indices into `kernels`/`regs`: 0=GsStepKernel 1=MaskKernel(excluded, shared test
    // artifact) 2=TwCopyKernel 3=TwSnapSyncKernel 4=TwSnapFireKernel 5=TwDrainKernel
    const int kGsStep = regs[0];
    struct ArmTotal {
        const char* arm;
        int total;
    };
    const ArmTotal arm_totals[] = {
        {"raw/hdf5/hdf5-async/hostclio", kGsStep},
        {"clio-sync", kGsStep + regs[2] + regs[3]},
        {"clio-async", kGsStep + regs[2] + regs[4] + regs[5]},
    };
    const int baseline = arm_totals[0].total;
    for (const auto& a : arm_totals) {
        const double pct_increase = 100.0 * double(a.total - baseline) / double(baseline);
        std::fprintf(stderr,
            "GSBENCH_KERNEL_REGS_ARM_TOTAL arm=\"%s\" total_regs=%d vs_baseline=%+.1f%%\n",
            a.arm, a.total, pct_increase);
    }
}

// ---- the three arm TEST_CASEs (run each in its OWN process) ----------------

// raw arm needs no CLIO server.
TEST_CASE("GSBENCH raw disk (no CLIO)", "[.][integration][gpu][gsbench][gsbench_raw]") {
    Cfg cfg;
    uint64_t checksum = 0;
    double ms = RunRawArm(cfg, &checksum);
    PrintResult(cfg.raw_inline ? "raw_inline" : "raw_threaded", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

// Tag alias [gsbench_sync] kept for back-compat with the older sweep/campaign scripts (run_b...
TEST_CASE("GSBENCH gpuh5 sync",
          "[.][integration][gpu][gsbench][gsbench_gpuh5_sync][gsbench_sync]") {
    Cfg cfg;
    static BenchEnv env(cfg);
    uint64_t checksum = 0;
    double ms = RunClioArm(cfg, ClioMode::kSync, "results/gsbench/gpuh5_sync", &checksum);
    PrintResult("gpuh5_sync_relaunch", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

// Host-driven CLIO (raw's flow, CLIO sink): isolates the storage path vs raw, and the
// submission model vs the GPU-producer sync/async arms.
TEST_CASE("GSBENCH clio host-driven", "[.][integration][gpu][gsbench][gsbench_hostclio]") {
    Cfg cfg;
    static BenchEnv env(cfg);
    uint64_t checksum = 0;
    double ms = RunHostClioArm(cfg, "results/gsbench/hostclio", &checksum);
    PrintResult("hostclio", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

// The conventional path (GPU -> D2H -> HDF5), and the baseline the paper is measured
// against. Needs no CLIO server, like raw. Compiled only when HDF5 was found.
#if GSBENCH_HAVE_HDF5
// The row is named for the writer STRUCTURE, not the sink, so that a sweep which runs this a...
TEST_CASE("GSBENCH hdf5", "[.][integration][gpu][gsbench][gsbench_hdf5]") {
    Cfg cfg;
    uint64_t checksum = 0;
    double ms = RunHdf5Arm(cfg, &checksum);
    PrintResult(cfg.raw_inline ? "hdf5_inline" : "hdf5_threaded", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

// The typical-user, unoptimized HDF5 baseline: contiguous layout by default (or chunked via...
TEST_CASE("GSBENCH hdf5 naive", "[.][integration][gpu][gsbench][gsbench_hdf5_naive]") {
    Cfg cfg;
    cfg.hdf5_naive = 1;
    cfg.raw_inline = 1;   // synchronous/GPU-idle structure, matching hdf5_inline
    uint64_t checksum = 0;
    double ms = RunHdf5Arm(cfg, &checksum);
    PrintResult("hdf5_naive", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

// The HDF5 Asynchronous I/O VOL arm.
TEST_CASE("GSBENCH hdf5 async-VOL",
          "[.][integration][gpu][gsbench][gsbench_hdf5_async]") {
    Cfg cfg;
    uint64_t checksum = 0;
    double ms = RunHdf5AsyncArm(cfg, &checksum);
    PrintResult("async_VOL", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}
#endif

// Tag alias [gsbench_async] kept for back-compat (canonical name gpuh5_noreuse).
TEST_CASE("GSBENCH gpuh5 noreuse",
          "[.][integration][gpu][gsbench][gsbench_gpuh5_noreuse][gsbench_async]") {
    Cfg cfg;
    static BenchEnv env(cfg);
    uint64_t checksum = 0;
    double ms = RunClioArm(cfg, ClioMode::kAsync, "results/gsbench/gpuh5_noreuse", &checksum);
    PrintResult("gpuh5_noreuse", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

// Bounded-pool double buffering: ONE fused fill+fire kernel per snapshot, N chunks streamed ...
TEST_CASE("GSBENCH clio pooled", "[.][integration][gpu][gsbench][gsbench_pooled]") {
    Cfg cfg;
    static BenchEnv env(cfg);
    uint64_t checksum = 0;
    double ms = RunClioArm(cfg, ClioMode::kPooled, "results/gsbench/pooled", &checksum);
    PrintResult("pooled", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

// REUSE arm (DESIGN §7 "Option B", relaunched): async's shape but memory constant in snapsho...
TEST_CASE("GSBENCH gpuh5 (reuse; default)",
          "[.][integration][gpu][gsbench][gsbench_gpuh5][gsbench_reuse]") {
    Cfg cfg;
    static BenchEnv env(cfg);
    uint64_t checksum = 0;
    double ms = RunReuseArm(cfg, "results/gsbench/gpuh5", &checksum);
    PrintResult("gpuh5_relaunch", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

// PERSISTENT arm (DESIGN §7 "Option A"): the whole snapshot loop in ONE resident cooperative...
TEST_CASE("GSBENCH clio persistent", "[.][integration][gpu][gsbench][gsbench_persistent]") {
    Cfg cfg;
    static BenchEnv env(cfg);
    uint64_t checksum = 0;
    double ms = RunPersistentArm(cfg, "results/gsbench/persistent", &checksum);
    PrintResult("gpuh5", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

// persistent_sync: the resident cooperative kernel but with SYNCHRONOUS submit (fire-AND-wai...
TEST_CASE("GSBENCH clio persistent_sync",
          "[.][integration][gpu][gsbench][gsbench_persistent_sync]") {
    Cfg cfg;
    static BenchEnv env(cfg);
    uint64_t checksum = 0;
    double ms = RunPersistentArm(cfg, "results/gsbench/persistent_sync", &checksum,
                                 /*async_submit=*/false);
    PrintResult("gpuh5_sync", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

// persistent_floor: COMPUTE-ONLY isolation of GsPersistentKernel via GsPersistentComputeOnly...
TEST_CASE("GSBENCH clio persistent floor (compute-only)",
          "[.][integration][gpu][gsbench][gsbench_persistent_floor]") {
    Cfg cfg;
    static BenchEnv env(cfg);
    double ms = RunPersistentFloorArm(cfg);
    const double total_steps = double(cfg.snaps) * double(cfg.steps_per);
    std::fprintf(stderr,
        "GSBENCH_PERSISTENT_FLOOR arm=gpuh5_floor N=%u snaps=%u steps_per=%u total_steps=%.0f "
        "compute_ms=%.3f ms_per_step=%.6f\n",
        cfg.N, cfg.snaps, cfg.steps_per, total_steps, ms, ms / total_steps);
    PrintResult("gpuh5_floor", cfg, ms, /*checksum=*/0);
    REQUIRE(ms > 0.0);
}

#endif  // !CTP_IS_DEVICE_PASS

#else
#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
