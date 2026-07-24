# clio-fs over the CTE shared-memory cache + async writes (issue #817)

Status: **IMPLEMENTED** — reads (phases 0-3) and async writes (4-5).
Phase 2 (zero-IPC stat) deliberately still open, see below.
Branch: `817-cfs-shm-reads-async-writes`, cut from `origin/dev` @ 2dc4911d

## Result so far

| | before | after |
|---|---|---|
| 4 KiB cached `pread` through `CfsIo` | 118–246 µs | **0.27–0.50 µs** (>100x) |
| 128 × 64 KiB overwrite + fsync | 6.86 ms | **5.77 ms** (~13%) |

Measured same-query on one host: the RPC number is the same read issued
through `filesystem::Client::AsyncRead` against the same open handle, so the
only difference is which path serves the bytes. It is ~2.7x the ~90 µs a bare
CTE `GetBlob` costs, which is the two-hop shape of a CFS read (client → fs
chimod → per-page `GetBlob`) plus the runtime's page loop.

Suites: `cfs_shm_read` new, CTE core functional 13/13, SHM cache model 7/7,
POSIX adapter 77 assertions / 8 cases.

### The blocker that was not in the plan

**The payload fast path was unreachable for clio-fs by construction.**
`ExtendBlob` caps every physical block at `kMaxBlockChunk = 64 KB` on purpose
(freed extents must be reusable under fragmentation — issues 074/521/522), so
block count scales with blob size: a 1 MiB page is **16 blocks**. With
`kMaxInlineBlocks = 8` every blob over 512 KB — i.e. every clio-fs page — was
marked truncated, and a truncated record was refused wholesale.

Two changes, both in the CTE core:
- A truncated record is now readable over the **prefix its cached blocks
  describe**. They are the blob's leading blocks in logical order, so the
  information is exact; refusing the whole blob threw away complete knowledge
  of the part it had. Reads bound against `CoveredBytes()`, not `total_size_`.
- `kMaxInlineBlocks` 8 → 16, so one whole page fits. Costs +256 B per cached
  blob (record is 560 B), and the layout version went to 2 so a v1 client
  refuses rather than misreads.

The prefix rule is the load-bearing half: it is what keeps the bound safe for
*any* blob larger than the inline array, rather than moving the cliff.

### The one that mattered most: it was switched off in every real deployment

Everything above was measured against a **co-located** runtime (the test
process hosting its own runtime). Under a real `clio_run` daemon, composed from
YAML, the fast path did not engage at all — `flags=0x0`, every read falling
back to RPC, with nothing to indicate it.

`BuildShmBlobRecord` decided a block was node-local by asking whether its
target's routing query was literally written as `PoolQuery::Local()`. But
`Runtime::Create` registers every composed target as
`DirectHash(target_node)` (the sliding neighborhood window). So `IsLocalMode()`
was false for every target in any composed daemon, `kShmBlobDirectReadable` was
never set, and the payload fast path could not turn on — on any node, for any
blob. The only thing that ever satisfied it was a test hand-registering a
target with `PoolQuery::Local()`, which is exactly what the #783 tests do.

`TargetIsNodeLocal` now *resolves* the query to a node (DirectHash → hash %
num_containers → ask the pool manager where that container lives; DirectId and
Physical likewise; Broadcast/Range/Dynamic refuse), mirroring
`ResolveDirectHashQuery` so the fast path and the router cannot disagree about
what "local" means.

**The test now spawns its own daemon and composes it from YAML.** That is the
lesson worth keeping: an in-process runtime shares the address space and
hand-registers its targets, so it cannot tell a working fast path from one that
is off everywhere that matters.

### The other thing that was not in the plan: `placement_gen_` was never bumped

The client's payload read validates `ShmBlobRecord::placement_gen_` before and
after the memcpy, and that check was the design's whole answer to "the
DataOrganizer moved this blob while I was copying it". **Nothing in the runtime
ever incremented it** — the comparison was `0 == 0` and could only ever catch
an outright erase. Making the read path hot for every clio-fs read made that
worth fixing rather than noting.

Three parts, because a generation alone is not enough:

- `BlobInfo::placement_gen_`, bumped by every mutation of `blocks_`
  (`ExtendBlob`, `TruncateBlobBlocks`, `FreeAllBlobBlocks`, `FlushData`).
- It is drawn from a **runtime-wide monotonic counter, not a per-blob
  increment**. `ReorganizeBlob` moves a blob by DelBlob + PutBlob, so the
  replacement `BlobInfo` starts at zero and a same-size blob re-extends into
  the same block count — landing on the exact generation it had before the
  move, and silently passing the check.
- **The mirror must be republished wherever blocks change.** `TruncateBlob`
  did not, so after a shrink the cache still described blocks that had gone
  back to the bdev free pool. A generation cannot rescue this on its own: a
  record that is never rewritten shows the reader the same value twice.

A test covers the shrink case end to end (truncate, then read inside and past
the new EOF).

### Remaining headroom (not pursued — the target is met with 2.6x margin)

At 0.386 µs the dominant cost is no longer IPC, it is copying: `TryReadBlobShm`
copies the whole 560 B blob record **twice** (the seqlock read, then the
placement re-validation) plus the ~100 B file record, so ~1.2 KB of metadata
moves to deliver a 4 KB read. Two obvious follow-ups, in order of payoff:
a re-validation that reads only `placement_gen_` instead of the whole record,
and dropping `bdev_type_`/`node_id_` from `ShmBlockDesc` (nothing reads them —
their conclusion is already baked into the direct-readable flag), which would
take the record from 560 B to 432 B. There is also one heap allocation per
read: `CfsIo::Read`/`Pread` copy the path string out of the fd table.

Companion issue: #818 — compressed blobs in the SHM cache (out of scope here,
but it constrains §4.3).

## 1. Problem

Every `read()` and `write()` through clio-fs pays a blocking client→runtime
round-trip, even when the bytes live in a node-local RAM bdev that the calling
process has already mapped.

Read path today — **two dispatches per page**:

1. `CfsIo::DoRead` (`context-transfer-engine/adapter/cfs/cfs_io.cc:53`)
   allocates a staging buffer, calls `AsyncRead`, and **blocks** on `t.Wait()`.
2. `Runtime::Read`
   (`context-transfer-engine/filesystem/src/filesystem_runtime.cc:351`) loops
   over 1 MiB pages issuing `cte_.AsyncGetBlob(tag_id, PageName(cur), ...)` —
   a second dispatch per page, potentially to a different container.
3. The client `memcpy`s out of the staging buffer and frees it.

Write path today — `CfsIo::DoWrite` (`cfs_io.cc:79`) copies into a staging
buffer and blocks on `t.Wait()`. `CfsIo::Sync` is a no-op that documents
"writes are synchronous" (`cfs_io.h:136`).

Measured floors from #783: SHM transport round trip ~72 µs, a real CTE
`GetBlob` RPC ~90–120 µs, versus **0.207 µs** for a direct shared-memory
payload read of a 4 KiB blob. A cached small `pread()` through clio-fs costs
roughly two round trips where it should cost one `memcpy`.

## 2. The #783 fast path is dead outside its own tests

`Tag::GetBlob` already attempts it (`core/src/tag.cc:194`):

```cpp
auto *cte_client = CLIO_CTE_CLIENT;
if (cte_client->HasShmCache() &&
    cte_client->TryReadBlobShm(tag_id_, blob_name, data, data_size, off)) {
  return;
}
```

`HasShmCache()` is always false in production. The **only** callers of
`AttachShmCache()` in the tree are `test/unit/test_core_functionality.cc:362`,
`:2754`, `:2912`. `ContentTransferEngine::ClientInit`
(`core/src/content_transfer_engine.cc:46`) creates/binds the pool, assigns
`cte_client->pool_id_`, and never attaches.

So phase 0 of this work is one call — and it benefits every adapter (FUSE,
HDF5 VFD, MPI-IO, `gpu_vector`), not just clio-fs.

## 3. What makes the client-side read path possible

Everything the client needs to build a cache key is derivable without asking
the runtime:

- **Tag name == the stripped path.** `Runtime::Open`
  (`filesystem_runtime.cc:268`) resolves the tag with
  `cte_.AsyncGetOrCreateTag(path, ...)`, so `TryGetTagIdShm(path)` returns the
  same `TagId` the chimod uses.
- **Page blob names are arithmetic.** `PageName(off)` is
  `std::to_string(off / kFsPageSize)` with `kFsPageSize = 1 MiB`
  (`filesystem_tasks.h:40`, `filesystem_runtime.cc:57`).
- **The CTE core client is already up.** `CLIO_CFS_CLIENT_INIT`
  (`filesystem/src/filesystem_client.cc`) calls
  `clio::cte::core::CLIO_CTE_CLIENT_INIT()` first, and the fs pool is layered
  over `kCtePoolId`, so `CLIO_CTE_CLIENT` is bound to exactly the pool whose
  cache root the client must attach.

## 4. Design

### 4.1 Read

```
CfsIo::Pread(fd, buf, count, off)
  ├─ dirty-page overlap with in-flight writes?     → RPC path (ordered behind them)
  ├─ file has pending deferred appends?            → RPC path
  ├─ logical size unknown / read crosses EOF?      → RPC path
  └─ per page in [off, off+count):
       TryReadBlobShm(tag_id, PageName(cur), dst, n, page_off)
         any failure → abandon the whole request, take the RPC path
```

Fallback is **per request, not per page**: a partially-fast-pathed read has to
reason about which bytes came from where, and the RPC already handles the whole
range correctly. Simplicity wins over the marginal case.

`TryReadBlobShm` already refuses non-local, non-RAM, GPU-tier, block-list-
truncated and placement-moved blobs, and validates `placement_gen_` before and
after the copy. Nothing in this design weakens that; a `false` always means
"use RPC", never "hole" or "EOF".

### 4.2 Logical size — the one real gap

EOF clamping and hole zeroing use `FileInfo::size_`
(`filesystem_runtime.cc:359`, `:370-384`), which is owned by the fs chimod and
is **not** the tag's physical size (`ShmTagRecord::total_size_`). They diverge
after `ftruncate`-grow, sparse writes, and deferred appends. Reading a hole
must produce zeros, and reading past EOF must produce a short read — getting
this from the wrong number is a correctness bug, not a perf regression.

**Chosen: publish the fs logical size into shared memory.** The
`MetadataDirectory` registration mechanism is already generic
(`RegisterRoot`/`FindRoot`, added in #783 phase 4), so the filesystem chimod
gets its own root and mirrors `path → {tag_id, logical_size, mode, uid, gid,
atime, mtime, ctime, flags}` on every change, best-effort and always *after*
the authoritative update so the cache can only lag.

This subsumes a second win: `getattr`/`stat` become zero-IPC. Today `ls -l`
costs one RPC per file (`CfsIo::StatPath` → `QueryGetattr` → `AsyncGetattr`).

Rejected alternative: have the client track size from `Open` plus its own
writes and revalidate by RPC when a read would cross the believed EOF. Cheap,
but silently wrong the moment a second process writes the file.

### 4.3 Write

Keep the staging copy (the caller's buffer is reusable on return, so the copy
is mandatory), drop the wait:

- `DoWrite` parks the `Future<WriteTask>` + its staging buffer in a
  per-descriptor in-flight list and returns `count` immediately.
- **Bounded window.** Configurable caps on in-flight bytes and count; on
  exceeding either, reap completions until under the limit. This is the
  back-pressure that keeps staging memory bounded — an unbounded queue turns a
  `dd` into an OOM.
- `fsync`/`fdatasync`/`close` become real: drain, free buffers, surface errors.
- **Sticky error latch.** A failed async write cannot set `errno` on the
  `write()` that queued it. Latch the error on the descriptor and report it
  from `fsync()`/`close()` — the same contract as kernel page-cache
  writeback. (As built, `write()` is excluded from that list deliberately: see
  the contract under "Async writes" below.) Document it in `cfs_io.h` where
  the "writes are synchronous" comment lives today.
- `O_SYNC`/`O_DSYNC` keep today's synchronous behaviour.
- **RYOW.** Track dirty page ranges per descriptor; a read overlapping one
  takes the RPC path (ordered behind the queued write) or is served from the
  staging buffer. This is the per-client pending-write set that #783 §5.4
  anticipated but never needed, because writes stayed synchronous.

### 4.4 Interactions to respect

- **Deferred appends.** `Runtime::Append` (`filesystem_runtime.cc:456`) stages
  bytes under `staging_tag_id_` and merges later via
  AppendSequence→AppendCollect→AppendExecution. Until a batch is sequenced the
  file's pages are not stably addressable, and `fi->size_` is explicitly
  best-effort. A file with pending appends must not fast-path.
- **Compressed blobs.** `ShmBlobRecord` carries no compression state and
  `kShmBlobDirectReadable` does not consult it (#818). Out of scope here; this
  design assumes raw page bytes and must be revisited if #818 lands option (B).
- **DataOrganizer.** Handled by the existing `placement_gen_` bracket — no new
  machinery.

## 5. Phasing

| Phase | Deliverable | Status |
|---|---|---|
| 0 | `ClientInit` attaches the SHM cache | **done** — test asserts `HasShmCache()` on an ordinary client |
| 1 | fs chimod publishes per-path attrs to SHM | **done** — `ShmFsCache`, mirrored at open/write/append/truncate/unlink/rename/utimens/chown |
| 2 | zero-IPC `getattr`/`stat` in `CfsIo` | **not done** — see below; reads do not need it |
| 3 | zero-IPC page reads in `Read`/`Pread` | **done** — 0.386 µs, fallback covered |
| 4 | async writes + bounded window + real `fsync`/`close` | **done** — 5.77 ms vs 6.86 ms, `CLIO_CFS_ASYNC_WRITES=0` reverts |
| 5 | RYOW dirty-page tracking | **done** — per-path write windows, drained by any read/stat/truncate that overlaps |
| 6 | regression sweep | partial — CTE + POSIX adapter green; xfstests/FUSE/VFD not run |

Phase 2 is deliberately still open. Accelerating `stat` means resolving a
NAME from a path-keyed cache, and that brings in the descendant problem a
directory rename creates: a renamed directory's children keep the right tag
(rename preserves the TagId, so pages follow it), which is exactly right for
an already-open descriptor and exactly wrong for name resolution. The read
path is reached only through an open fd, so it sidesteps this; a stat fast
path cannot, and needs a descendant sweep or a generation on the path space
first.

Phases 0–2 are independently valuable and independently mergeable; 3 depends on
1; 5 must land with or before 4 is enabled by default.

## 6. Benchmarks to report (all timings in ms)

- 4 KiB cached `pread` latency through the adapter, SHM vs RPC, same query.
- Sequential 1 MiB `write` throughput, sync vs async, at several window sizes.
- `stat()` latency, and `ls -l` wall time over a 1000-file directory.
- Write-then-read cycle — #783 measured only ~2x there because the cycle is
  write-dominated. Async writes attack exactly that half, so the cycle number
  is the one that shows whether the two halves compose.

## 7. Open questions

- Window sizing default, and whether it is per-descriptor or per-process.
- Whether `fsync` should also flush the fs chimod's own append pipeline, or
  only this descriptor's queued writes.
- Whether the fs attr mirror should be a separate map or reuse the CTE tag map
  with an fs-specific record (separate map is cleaner; costs another fixed
  capacity to size).


## Async writes (phases 4-5)

`write(2)` hands the bytes to the chimod and returns; `fsync`/`close` drain and
report. On by default; `CLIO_CFS_ASYNC_WRITES=0` restores blocking writes.

The contract:

- **Read-your-own-writes**, no fsync required. A queued write is invisible to
  both read paths until its task runs, so a read overlapping one waits for it.
  Tracked per **path**, not per descriptor — RYOW is a property of the file,
  and per-fd tracking would satisfy the common case while quietly breaking two
  descriptors on one file.
- `stat` / `lseek(SEEK_END)` / `truncate` / `unlink` / `close` drain first, so a
  size query never reports less than a completed `write(2)` established.
- **`write(2)` always succeeds.** A queued write that fails latches its errno
  on the path's window; `write` neither reports nor consumes it. Kernel
  page-cache writeback has the same contract: the call that hands bytes to
  writeback returns success, and reporting the failure on some later
  unrelated `write` would attribute it to bytes that are fine.
- **Errors surface at `fsync`/`close`**, and only there — which makes those
  two the *only* places a queued failure can reach the application. So the
  drain that `stat`/`lseek(SEEK_END)`/`ftruncate` perform must PEEK at the
  latch, never clear it; an intervening `stat(2)` silently eating the report
  was a real bug this contract turned from unlikely into routine.
- **Bounded window** (64 MiB / 8192 writes, env-overridable): past it,
  submitting blocks on the oldest. Without it a `dd` grows staging buffers
  without bound. The bound is back-pressure — it can make a write *wait*, but
  never fail. Staging comes out of the client's 256 MiB SHM allocator, so the
  two are sized together; if the allocator does refuse, `write` drains its own
  windows to reclaim buffers and retries rather than returning `ENOMEM`.
- `O_SYNC`/`O_DSYNC` still block **and still report**: that caller asked for
  durability over latency, and answering "success" for a write that failed
  would be the one case where always-succeed is simply wrong.

**What always-succeed does not do is make failures go away.** A full tier
still refuses the bytes; the application now learns at `fsync`/`close` instead
of at a `write`. Measured on a 2 GiB RAM target with 4 KiB writes: every
`write(2)` returns success through the point of refusal, and fio's `end_fsync`
reports the EIO (`sync offset=33550336`). An application that never calls
`fsync` and ignores `close`'s return learns nothing at all — which is the
POSIX writeback bargain, not a clio-fs quirk.

**Cost of the 64 MiB / 8192 window: none that is measurable.** Nine paired
runs (fio, 4 KiB, 64 MiB, fresh daemon per run, arms interleaved): median
ratio new/old 1.03, the new window faster in 5 of 9, medians 1621 vs 1568
IOPS. At 64 KiB the two are likewise indistinguishable.

That is a correction. A first three-run batch gave 3121 vs 3630 with
non-overlapping ranges, and this document briefly claimed a ~14% regression on
the strength of it. It did not replicate. **Write throughput on a loaded host
drifts far more than a three-run batch can see** — absolute numbers moved by
3x across batches an hour apart (3630 → 1568 IOPS for the *same* configuration)
— so a three-run A/B can separate cleanly and still be measuring the drift
rather than the change. Pair the arms, run at least eight, and report the
paired ratio; that is the same lesson as the fresh-vs-warmed-file confound
above, in a different disguise.

An intermediate arm ruled out the obvious mechanism: holding bytes at 64 MiB
and varying only the task count (64 vs 8192) made the DEEPER queue faster, so
the per-write cost was not the linear scan over the in-flight queue. (That scan
lived in the since-removed page-serialization wait; see "The page-serialization
was wrong" below.) Worth knowing before anyone optimizes a queue scan on
suspicion.

### How I measured this wrong, twice

The gain is modest — median 5.77 ms vs 6.86 ms over five runs each, ranges
overlapping. Getting to that number took two corrections worth recording:

1. **Timing a fresh file against a warmed one.** The first pass over a new file
   allocates every block; a second does not. Timing one mode on a fresh file
   and the other on the warmed file compares *allocation against overwrite*.
   That produced a confident "async is 2.3x SLOWER", which I had already
   written into the code as the documented reason to ship it disabled. The tell
   was that the two arms differed by 40% while running the **same code path**.
2. **Believing a plausible mechanism.** The first explanation for the (bogus)
   slowdown was per-blob write-token contention (#680), whose contender
   busy-polls on a ~2 ms cadence. It is a real mechanism and it predicted the
   result — so I implemented one-outstanding-write-per-page around it. That
   changed nothing, which should have been the signal to re-examine the
   measurement rather than the mechanism.

### The page-serialization was wrong, and cost the whole async win

The one-outstanding-write-per-page wait was kept anyway, on the argument that
it "costs nothing on the sequential path." That was exactly backwards. A page
is 1 MiB and the fs writes in 1 MiB pages, so a **sequential 4 KiB writer puts
256 writes on one page** — every one of them waited for its predecessor, and
async writes ran precisely as slow as synchronous ones. It was the sequential
path it cost the most.

Measured, 4 KiB, one batch: sequential (all one page) p50 **371 µs** / 1298
IOPS against random (across pages) p50 **4.8 µs** / 4367 IOPS — a 77x latency
gap from nothing but the wait.

The page wait was **too coarse** — it serialized disjoint sequential writes,
which is what made a 4 KiB writer synchronous. But removing it **entirely** was
too far, and finding the line between took three tries and a near-miss.

The first fix removed the wait outright, reasoning that write ordering is the
runtime's. A `overlapping-write ordering` test — 512 rewrites of one 4 KiB
range — passed on the dev box, so that looked confirmed. It failed on macOS CI.
So I narrowed the wait to **byte-range**: a write waits for an outstanding
write to the *same bytes* (never for a disjoint one), which keeps the
sequential speed while ordering overlapping writes.

Then I actually measured the overlapping case, and it is worse than a
granularity question. **The runtime does not order overlapping same-blob writes,
and the client cannot fully make it.** They race the #680 write token, and
`AsyncWrite`'s future signals *before* the write durably commits — so even
after waiting on the prior write's future, a "completed" earlier write can land
after a later one. Measured, 512 rewrites of one range, RPC read confirming the
durable state (not a mirror lag): the last write is lost **~30% of runs with
the byte-range wait, ~100% without it.**

At which point the right question is not "which wait fixes this" but "does this
pattern occur." It does not. Writes are byte-range partial puts, so two writes
to different bytes of one page never conflict; a sequential writer's offsets
are disjoint; a random writer's are distinct; and an application that issues two
`write()`s to the *same bytes* with nothing between, and depends on which wins,
has lost that race to itself. The only thing a same-range wait would guard
against is a workload nobody runs — and it does not even guard it fully. So the
client **waits for no write at all.** The wait is gone entirely, the storm test
with it (it asserted a guarantee neither the runtime nor a client provides), and
the residual is left to a runtime fix (#680): an application needing strict
last-writer-wins across overlapping unsynced writes must fsync between them or
use O_SYNC.

What #817 delivers stands: sequential 4 KiB async p50 **371 µs → 7.2 µs**,
overwrite+fsync ~13% over O_SYNC → **2.3x**.

Two methodological points. A concurrency invariant only the dev box has checked
is not established — the dev box passed the ordering test at every stage while
macOS, then an RPC-confirmed measurement, showed the guarantee never held. And
before hardening a path, ask whether the case is real: the honest fix for the
overlapping-write "bug" was to delete the test, not the latency.

## The SHM path off Linux

macOS had **no metadata segment at all**, so no #783 cache and no #817 fast
path — everything silently on RPC. The 8 GB default reservation is justified by
"never pre-faulted, only pages actually written consume RAM", which describes
`memfd_create` — and `SystemInfo::CreateNewSharedMemory` only uses memfd on
Linux. macOS/BSD fall back to a **regular file** under `/tmp/clio_$USER`, where
that reasoning does not hold: the ftruncate and mmap go against the disk.
Clamped to 1 GB where the segment is file-backed (~1.9M cached blobs at 560 B).

The test distinguishes "this feature failed" from "the substrate does not exist
here": it skips loudly when the runtime has no metadata allocator, and still
fails when the segment exists but the cache is off — which is the case that
would actually mean regression.
