# Vectored PutBlob/GetBlob + worker-local task batching (issue #820)

Status: **A and B IMPLEMENTED AND MEASURED.** Batching is opt-in
(`CLIO_BATCH_LANE=1 CLIO_CTE_BATCHING=1`) pending broader soak, but it works and
delivers ~3.7x end to end; the vectored task shape underneath is 26–28x.
Branch `820-worker-task-batching` off `origin/dev` @ 04ebc438.

## Result so far

**Vectored PutBlob is a 26x win on the case this issue is about.** Same bytes,
same blob, no batching phase and no concurrency involved — only the task shape
(`bench_same_blob_batching`, 256 x 4 KiB to one blob, 20 rounds, medians, both
arms warmed):

| | median | range |
|---|---|---|
| 256 single PutBlobs | 37.86 ms | 34.41–41.43 |
| **1 vectored PutBlob** | **1.46 ms** | 1.22–2.87 |

**25.97x**, with non-overlapping ranges. That is the #680 write token being
acquired once instead of 256 times, one metadata mutation instead of 256, and
one bdev pass instead of 256.

### Status by part

- **A1 vectored PutBlob — done.** `segments_` on the task, `PutBlobImpl` applies
  all segments under one token acquire. `test_vectored_blob_io` asserts
  equivalence with N single puts, list-order overlap resolution
  (last-writer-wins), and union-range allocation with a zero-filled hole.
- **A2 vectored GetBlob — done.** Each region read into its own buffer off one
  block snapshot. Tested.
- **B0/B1 worker batch phase — done.** `Worker::BatchPhase`, `BuildBatch`/
  `SmashBatch` container virtuals (default decline/no-op), `TASK_BATCH_MERGED`
  parent fan-out. Verified behaviour-neutral: with the CTE policy off the full
  suite passes, including 8-writer same-blob stress.
- **B2/B3 CTE policy — implemented, no longer hangs, DEFAULT OFF pending a
  workload that actually exercises it (`CLIO_CTE_BATCHING=1` to enable).**
  `test_concurrent_same_blob` (8 writers), `test_cte_parallel_overlap_stress`
  and `test_vectored_blob_io` all pass with it on.
- **B4 — the 26x measurement above covers the mechanism. The end-to-end win is
  NOT yet demonstrated, for the reason in "the ingress" below.**

### The batching was wired to the wrong ingress

The lane-based phase never saw a single hot-path task. Under the #807 SHM
transport a client's PutBlob/GetBlob is delivered to its worker's **shard ring**
and executed inline in `DrainMyShard` — it never enters the lane. Instrumented
under an 8-writer same-blob load: **zero tasks parked, zero merges**, while the
lane carried only occasional internal work.

That also explains the hang. The lane carries internal and re-routed tasks that
other in-flight work may be synchronously waiting on, so *deferring* one until
the end of the phase can deadlock. Freshly ingested client requests have no such
dependents yet. Moving the phase to `DrainMyShard` and leaving the lane path
exactly as it was fixed the hang outright — 8-writer same-blob went from hanging
to passing, with the policy on.

### Where batching has to happen, and why it is blocked there

Correction to the section above: moving to the ingress **masked** the hang, it
did not fix it.

`RouteTask` sends every task for a blob to that blob's container — i.e. to ONE
worker's lane. So the ingesting worker is usually *not* the executing worker: it
receives a request, routes it onward, and never gets to merge it. The lane is
the only place same-blob work converges, and therefore the only place a batch
can form. Batching at the ingress is safe precisely because it never batches
anything.

Enabling lane batching (`CLIO_BATCH_LANE=1`) with the CTE policy on **hangs**
again, on the async-burst benchmark. So the deferral hang is real, independent
of the use-after-free already fixed, and it sits squarely on the only viable
ingress. That is the blocker.

Measured A/B, async burst of 256 x 4 KiB outstanding at one blob, 10 rounds:

| | median | range |
|---|---|---|
| ingress batching, policy off | 12.28 ms | 11.65–12.47 |
| ingress batching, policy on | 12.99 ms | 11.52–15.16 |
| lane batching, policy off | 13.01 ms | 12.20–17.88 |
| lane batching, policy on | **hangs** | — |

No improvement, as expected: nothing merges. This is a negative result, recorded
as such.

### How far the hang was narrowed, and where tracing stopped working

With `CLIO_BATCH_LANE=1` the merge path *does* run — traced groups of 2 and 61
members, `NewCopyTask` succeeding, and every segment pushed
("all 61 pushed"). The stall is downstream of that, inside `BatchSink::Emit`,
between assigning the merged task's id and the `Send`.

Finer localization then failed for an instructive reason: with per-statement
`fprintf`+`fflush` from multiple worker threads, the trace stopped between two
*adjacent* prints — which is not a logic hang but **stderr lock contention**,
one worker blocking in `fprintf` while another holds the stream lock. Past that
point the instrumentation was measuring itself. The remaining work needs the
stuck process inspected directly (thread backtraces) rather than more printf
tracing.

What is established: the merge builds correctly, and the failure is in
publishing the merged task / completing its parents — not in the vectored task
shape, which is independently exercised and green.

### It is a LOST WAKEUP, not a deadlock

Thread-state inspection of the wedged process (`/proc/<pid>/task/*/{stat,wchan}`,
which works under `ptrace_scope=1` where `gdb -p` does not) settles the
character of the bug:

- **No thread is in state R.** Nothing is spinning.
- **No thread is blocked on a futex.** So it is not a mutex deadlock.
- Every thread is in `hrtimer_nanosleep` (workers in their adaptive idle sleep)
  or `do_epoll_wait` (ZMQ I/O, peer/client receive).

So the merged task is published and then simply never runs: every worker parks,
the client waits on a future nobody will complete, and the system is quiescent.
That is the signature of a task enqueued without its consumer being woken, or of
a task that never reaches a lane at all.

Ruled out so far:

- **Not the enqueue path itself.** `RouteLocal` pushes and then unconditionally
  `AwakenWorker`s, with a comment describing this exact lost-wakeup race.
- **Not inherited lifecycle flags.** `BatchManager` clears them on its aggregate
  because `NewCopyTask` carries `task_flags_` over; building the merged task
  *fresh* from `NewTask` and copying only the identifying fields was tried and
  **did not fix it** (reverted, since it was unverified and strictly less
  complete than the copy).

### Root cause, as far as it is established: `Send` never returns

Re-instrumented with a **lock-free** tracer (`write(2)` to an `O_APPEND` fd — one
syscall, no userspace lock, unlike the `fprintf` that faked a stall earlier):

```
SMASH method=15 members=5     EMIT-ENTER parents=5
SMASH method=15 members=55    EMIT-ENTER parents=55
EMIT-SENT: 0
```

So the merge runs, real groups form (5 and 55 members), `Emit` is entered — and
`CLIO_IPC->Send(merged)` **never returns**, with the calling thread asleep.

That matches a hazard `IpcCpu2Self::SendIn` documents against itself: a self-send
routes via `RouteTask(force_enqueue=true)` → `RouteLocal` → `dest_lane.Push()`,
and *"when that lane fills, the WAIT_FOR_SPACE ring Push busy-spins forever — the
worker never suspends, so it can never pop to make space (producer==consumer
deadlock)"*. `RouteLocal` guards this by picking an alternate worker when the
destination is the caller — but only `if (alt != nullptr)`.

Emitting from inside the drain loop is exactly that shape: the worker is the
consumer of the lane it may be asked to push onto, and under a 64-deep burst the
lane is full. The one discrepancy to resolve is that the thread is *asleep*
rather than spinning, so either the ring's wait sleeps, or the block is one level
deeper than `Push`.

**FIXED.** Merged tasks are no longer `Send`-ed. The sink runs them **inline on
the current worker**: it gives the merged task its own `Future` + `RunContext`
(it has no client waiter of its own — its job is to complete the parents it
subsumed) and calls `ExecTask` directly. That is also simply correct, not just a
workaround: every member routed to this container on this worker, so the merged
task belongs here by construction and never needs to enter a lane at all.

With that, batching works and pays:

**Five paired runs** (async burst, 256 x 4 KiB outstanding at one blob, 10 rounds
per run, medians; arms interleaved on the same binary so no rebuild or host
drift sits between them):

| pair | batching off | batching on | ratio |
|---|---|---|---|
| 1 | 12.01 ms | 3.26 ms | 3.68x |
| 2 | 12.54 ms | 3.41 ms | 3.68x |
| 3 | 12.91 ms | 3.73 ms | 3.46x |
| 4 | 15.24 ms | 3.41 ms | 4.47x |
| 5 | 12.27 ms | 3.27 ms | 3.75x |

**Median ratio 3.68x, batching faster in 5/5**, and the two distributions do not
overlap (off 12.0–15.2, on 3.3–3.7). Unlike the write-window measurements in
#817, this one did not need pairing to survive — the effect is far larger than
this box's drift.

Also applied while investigating (kept — correct on its own): the current-task
slot is cleared before emitting. `IpcCpu2Self::SendIn` parents a self-sent task
to `GetCurrentTask()`, which by then is a stale, already-finished passthrough
task; a merged task adopted by a dead parent has its completion routed into that
parent's coroutine. It did not resolve the hang, but leaving it unfixed would
have been a second bug waiting behind the first.

### Why the end-to-end win is still not demonstrated

Even at the right ingress, batching only pays when requests **co-arrive**: two
same-blob requests must land in one worker's shard within one drain window. A
synchronous client has exactly ONE outstanding request per thread, and threads
spread across shards, so a worker sees one task at a time and every group is a
group of one (measured: still zero merges at 8 threads).

The contention is unquestionably there — same run, 8 writers at one blob:
mean 2.72 ms, p50 277 µs, **p99 38.7 ms, p99.9 101.6 ms, max 118.9 ms**. That
tail is the #680 token. But collapsing it needs a *bursty* submitter, which is
exactly what #817's async writes produce (write(2) returns once staged, so many
requests are in flight at once). Until this branch is combined with that path,
the batching policy is a correct no-op on this workload.

**So the honest split:** the vectored task shape is the win and it is proven
(26x). The batching layer that would feed it automatically is built, safe, and
in the right place, but needs an async/bursty client to have anything to merge.

### Bugs found while building this

1. **Emit vs Passthrough.** Routing a group of ONE through the merged-task path
   reassigned the task's id — but its client future was already bound to the old
   id, so the client hung forever. A parked original must run *unchanged*
   (`BatchSink::Passthrough`); only a freshly built synthetic task may be
   `Emit`ted.
2. **Dangling group key.** `SmashBatch` held `const BatchKey &key = it->first`
   and then `groups.erase(it)` — every later use of `key` (the method checks,
   the sort comparator, the merged-task construction) read freed memory. It
   showed up as "park method=15, smash method=25": the same group reporting two
   different methods, so the merge took the wrong branch. Copy the key by value.
3. **Started tasks are not candidates.** A task popped off the lane may be a
   coroutine resuming mid-execution (`ContinueBlockedTasks` re-queues those).
   Folding one into a fresh merged task abandons its in-flight state and strands
   its waiter. Only never-started tasks are offered.

The remaining hang is B2-specific and reproduces only with the CTE policy on and
>=8 concurrent writers to one blob; the phase itself is clean at the same
concurrency.

Two coupled pieces:

- **A. Vectored PutBlob/GetBlob** — a scatter-gather segment list on the task, so
  one task acquires the blob write token once and applies N disjoint regions in a
  single pass. Independently useful as a client API, and it is the shape the
  batching layer produces.
- **B. Worker-local batching** — coalesce many same-blob tasks into the minimal
  set of **vectored** tasks before they hit the runtime.

A lands first (independently testable/mergeable); B builds on it.

Companion to #817 (async writes, which surfaced the tail) and #680 (the per-blob
write-token contention this sidesteps).

## 0. Vectored PutBlob/GetBlob (part A)

Today `PutBlobTask` carries one region: `offset_`, `size_`, `blob_data_`
(`core_tasks.h:1303`). Give it a segment list instead:

```cpp
struct BlobSegment { u64 offset; ShmPtr<> data; u64 size; };
IN priv::vector<BlobSegment> segments_;   // empty => use the legacy single-region fields
```

The single-region ctor stays (fills a 1-segment view), so every existing caller
is unchanged. `PutBlobImpl`/`GetBlobImpl` change from "one region under the token"
to "**all segments under one token**":

1. Acquire the blob write token once.
2. Extend/resize to cover `[min segment offset, max segment end)` — one metadata
   mutation pass.
3. For each segment in **list order**, `ModifyExistingData` (PutBlob) or read into
   `segment.data` (GetBlob). List order gives **last-writer-wins** on overlap —
   the very ordering the #680 token race currently gets wrong (#817 write notes).
4. Release.

`SaveTask`/`LoadTask` serialize the vector; the merged tasks batching produces run
locally (`ExecHere`) so they don't round-trip, but the fields must serialize for
the client-API use and for cross-node vectored submits.

Net: N disjoint same-blob writes = **one** token acquire and **one** bdev pass,
with no contiguous gather buffer and no post-hoc scatter — each parent's `ShmPtr`
is a segment, and GetBlob reads straight into each parent's buffer.

## 1. Why

The clio-fs page model stores each 1 MiB file page as one blob. A sequential
4 KiB workload therefore puts **256 tasks on each page-blob**. Every `PutBlob`
holds that blob's #680 write token across its *entire* body — allocation **and**
the bdev data copy (`core_runtime.cc` `PutBlobImpl`, the `ModifyExistingData`
co_await at ~1678) — and a contender re-polls on a periodic cadence
(`kBlobWriteLockPollUs`, 1554). So the 256 writes drain single-file.

Measured, QD1, RAM target, 4 KiB: p50 **7.3 µs**, mean **~160 µs** → ~6.3K IOPS,
tail **p99.9 ≈ 10 ms, max ~1 s**. Block-size sweep pins the cause to
writes-per-blob: max stall **1082 ms** (4k, 256/page) → 10.5 ms (64k, 16/page) →
8 ms (1M, 1/page).

The 256 writes are to **disjoint byte ranges** of the page — they never conflict
on bytes — yet the token is per-*blob*, so they serialize anyway. The cheapest
place to fix that is *before* they reach the runtime: coalesce them into one
PutBlob over the union of their ranges. One token acquire, one bdev transfer,
256× fewer tasks.

## 2. Not the existing BatchManager

`BatchManager` (`batch_manager.{h,cc}`) already batches — but it is a **cross-node
collective** for `PoolQuery::ManyToOne`: park N tasks by
`(pool, method, container, batch_key)`, flush after a time window, combine their
inputs `AggregateIn` **N→1**, run **one** aggregate, then broadcast the single OUT
`1→N` back to every member (`OnAggregateComplete` → per-member `SetReturnCode` /
`SetCompleter` / `EndTask`). Every member gets the *same* result. Used today for
the clio-fs deferred-append pipeline (`AppendCollect`).

This proposal is different in kind: **intra-node coalescing**. N independent tasks
become a **minimal M-task subset** (not 1), and each output task completes a
*different* set of parents, scattering a *slice* of its OUT to each. Reduction vs
merge. We reuse BatchManager's completion-fan-out *mechanic* (its
`OnAggregateComplete` is the proof it works) but not its routing or its
single-result semantics.

## 3. Where it slots in

Worker main loop today (`worker.cc:276`): drain SHM shards → `ProcessNewTasks`
(pop ≤16, `RouteAndExec` each) → `ContinueBlockedTasks` → ManyToOne `FlushDue`.

`ProcessNewTask` pops one `Future<Task>`, resolves the container, and
`RouteAndExec` → `RouteTask`; on `RouteResult::ExecHere` it runs inline. Batching
inserts a phase between the dequeue and the execute, and only over tasks that
route **ExecHere** (a task destined for another node/container must route first,
untouched — we never batch across containers).

### The phase

```
BatchPhase(lane):
  groups := worker.batch_groups          // unordered_map<BatchKey, BatchQueue>, reused
  passthrough := worker.passthrough      // SPSC<Future<Task>>, reused

  for i in 0..MAX_BATCH_DEQUEUE (=64):
    if not lane.Pop(future): break
    route := RouteTask(future)
    if route != ExecHere:                // routed elsewhere; RouteTask already enqueued it
      continue
    container := container_of(future)
    container.BuildBatch(future.task, groups)   // routes into a group, OR...
      // default BuildBatch: passthrough.Push(future)   (not batchable)

  // run the non-batchable tasks as-is, in arrival order, first
  while passthrough.Pop(f): RouteAndExec(f, lane)   // (already ExecHere)

  // then let each container collapse its groups into minimal output tasks
  for each container with nonempty groups:
    container.SmashBatch(groups, sink)   // sink enqueues merged tasks + wires parents
  groups.clear()
```

Key decisions:

- **Bounded, backlog-only.** The `≤64` dequeue only ever *has* 64 to batch when a
  backlog exists. A lone write finds an empty lane after it, `BuildBatch`es a
  group of one, and `SmashBatch` emits it unchanged — **no added latency for the
  unbatched case**. Batching kicks in exactly when there is contention to amortize.
- **Passthrough first, merged last** (per the plan): preserves the arrival order
  of non-batchable work, and lets a merged task see a settled world.
- **Route before batch.** We only batch `ExecHere` tasks, so groups are always
  same-node, same-container — the merge never has to reason about placement.

## 4. Container interface

Two new virtuals on `Container` (`container.h`), default no-op / passthrough so
every existing container is unaffected:

```cpp
// Route `task` into the batch groups, or decline (caller sends it to passthrough).
// Default: return false (not batchable).
virtual bool BuildBatch(const shared_ptr<Task>& task, BatchGroups& groups) {
  return false;
}

// Collapse each group into the minimal set of output tasks. For each output,
// record the parent tasks it completes and how to scatter its OUT to them, then
// hand it to `sink` to enqueue. Default: no-op.
virtual void SmashBatch(BatchGroups& groups, BatchSink& sink) {}
```

`BatchGroups` = `unordered_map<BatchKey, priv::vector<shared_ptr<Task>>>` owned by
the worker and cleared each phase. `BatchKey` is opaque to the worker
(`{PoolId, method, u64 key}`); the container chooses `key` (for CTE:
`hash(tag_id, blob_name)` — the page).

`BatchSink` wraps "enqueue this merged task, and on its completion run this parent
fan-out." Implementation reuses BatchManager's pattern: a worker-side
`unordered_map<merged_task_uid, ParentCompletion>` consulted in `Worker::EndTask`
when a task is flagged `TASK_BATCH_MERGED`, mirroring `IsAggregate` /
`OnAggregateComplete`.

### Parent completion

Each merged output task carries (via the sink's side table) a list of
`{parent_task, out_slice}`. When the merged task reaches `EndTask`:

1. For each parent, copy the relevant slice of the merged OUT into the parent
   (for GetBlob: `memcpy` the parent's sub-range out of the merged read buffer
   into the parent's `blob_data_`; for PutBlob: set `bytes_written`/rc).
2. `parent->SetReturnCode(...)`, `SetCompleter(...)`, `worker->EndTask(parent)`.

This is exactly `OnAggregateComplete` generalized from one-result-broadcast to
per-parent-slice-scatter.

## 5. CTE PutBlob / GetBlob policy

`CteCoreContainer::BuildBatch`: batchable iff method ∈ {kPutBlob, kGetBlob} and
the task is an ordinary positioned op (no compression transform, no GPU page
suffix in v1 — those decline to passthrough). Group key = `hash(tag_id,
blob_name)`.

`SmashBatch`, per group, emits **vectored** tasks (part A):

1. **Sort** members by `(tag_id, blob_name, offset)`, ties broken by **submission
   order** (a monotonic seq the worker stamps at `BuildBatch` time). Submission
   order is what makes overlap resolution correct.
2. **Build one vectored task per blob**: each member becomes a segment
   `{offset, data, size}` in the task's `segments_`, in sorted+submission order.
   Adjacent members (`off_i + size_i == off_{i+1}`, same `ShmPtr` contiguous) MAY
   be fused into one segment to minimize count; disjoint members stay as separate
   segments in the **same** task. No contiguous gather buffer — the parents' own
   `ShmPtr`s are the segments. (A blob with more members than a segment cap splits
   into >1 vectored task.)
   - **PutBlob:** the runtime applies segments in list order → overlaps
     last-writer-wins.
   - **GetBlob:** the runtime reads each segment straight into that segment's
     `ShmPtr` = the parent's buffer, so completion needs **no scatter copy**.
3. Each vectored task's parent set = the members whose segments it carries; the
   completion fan-out sets rc/`bytes` per parent (GetBlob bytes already landed in
   the parent buffer).
4. Emit via `sink`.

Net for the 4 KiB sequential case: 256 same-page PutBlobs → **1 vectored PutBlob**
with 256 segments (or fewer, fused) over the 1 MiB page. One token acquire, one
bdev pass. The tail's root cause is gone, and applying segments in submission
order fixes the overlap-ordering gap #680 leaves open (#817 write notes).

## 6. Data structures to add

- `Worker`: `BatchGroups batch_groups_;` and
  `ctp SPSC<Future<Task>> passthrough_;` (the ctp ring buffer,
  `data_structures/ipc/ring_buffer.h`), both reused across iterations.
- `Worker`: `unordered_map<u64, ParentCompletion> batch_pending_;` +
  `TASK_BATCH_MERGED` flag (mirrors BatchManager's `pending_`/`TASK_BATCH_AGGREGATE`).
- `Container`: `BuildBatch` / `SmashBatch` virtuals + `BatchGroups` / `BatchSink`
  types.
- CTE container: the PutBlob/GetBlob policy above.

## 7. Open questions / hazards

1. **~~Staging-buffer gather cost~~ — removed by vectored tasks (part A).** The
   merged task carries the parents' `ShmPtr`s directly as segments, so there is no
   contiguous gather and no extra memcpy. The remaining cost is the segment vector
   itself (a few dozen bytes per member).
2. **Partial failure.** If the merged bdev transfer half-fails, how granular is
   the error to parents? v1: all covered parents get the merged rc (conservative).
   Only matters if the bdev can report partial byte counts; note and defer.
3. **Fairness / starvation.** The bounded 64 dequeue caps how long batching
   delays the passthrough tasks (which run first anyway). Confirm the phase can't
   starve `ContinueBlockedTasks` — keep it inside the existing per-iteration work
   budget.
4. **Interaction with routing side effects.** `RouteTask` may mutate RunContext /
   re-enqueue. We batch only `ExecHere` results, after routing — but confirm no
   task both routes ExecHere *and* expects to be re-dequeued.
5. **WAL / ordering vs the append pipeline.** clio-fs already has `AppendCollect`.
   Ensure positioned-write batching and the append collective don't double-handle
   the same bytes (they target different task methods, but write the same pages).
6. **Merged task accounting.** `GetTaskStats` / the #781 scheduler model sees one
   big task instead of 256 small ones — check the load model doesn't misprice it.
7. **GetBlob RYOW.** A merged read must still respect a pending overlapping write
   (the cfs `DrainIfOverlap` is client-side; server-side merged reads are new).

## 8. Phasing

| Phase | Deliverable |
|---|---|
| A1 | **Vectored PutBlob**: `segments_` on the task, `PutBlobImpl` applies all segments under one token, Save/Load serialize the vector, single-region ctor unchanged. Unit test: one vectored task vs N single writes, same result; overlap = last-writer-wins. Independently mergeable. |
| A2 | **Vectored GetBlob**: segment reads into per-segment `ShmPtr`. |
| B0 | Worker batch phase + `BatchGroups`/passthrough SPSC, `Container::BuildBatch`/`SmashBatch` default no-ops. **No behavior change** — every container passes through. |
| B1 | Parent-completion side table + `TASK_BATCH_MERGED` fan-out in `EndTask` (trivial echo container). |
| B2 | CTE `BuildBatch`/`SmashBatch` for PutBlob → emits vectored PutBlob. Correctness first, then the fio tail measurement. |
| B3 | CTE GetBlob batching → vectored GetBlob, RYOW respected. |
| B4 | Bench + tail: the 4 KiB seq write max-stall should collapse from ~1 s toward the 1-MiB-write floor; verify p99.9 and mean, not just p50. |

## 9. Success criteria

- 4 KiB sequential write **mean** latency and **p99.9/max** drop toward the
  1-writer-per-page floor (the block-size sweep's 8 ms end), not just p50.
- No regression on the full fsx/xfstests adapter suite (this touches the write
  path indirectly via coalescing).
- Unbatched single-op latency unchanged (the empty-backlog path adds nothing).

## 10. fio through clio-fs: the win does NOT reach the POSIX path

Measured on a throwaway branch merging this work with #817 (async writes, needed
so `write(2)` returns once staged and a burst of same-blob writes can form). fio
via the POSIX interceptor, 4 KiB, RAM target, async writes on, batching A/B via
`CLIO_CTE_BATCHING`.

**Concentrated** — 16 jobs on one 1 MiB file (= one page-blob), 8 paired runs:

| | median IOPS | range | median p99.9 |
|---|---|---|---|
| batching off | 12,219 | 8.7K–16.2K | 97 ms |
| batching on | 14,086 | 10.6K–16.3K | 79 ms |

**1.15x median, faster in only 4/8 — within noise.** Sequential (8 jobs × 64 MiB)
is parity. Contrast the direct-CTE microbenchmark, where the same knob is a clean
3.68x.

**Why it does not translate.** The clio-fs write path is `POSIX interceptor →
cfs_io → the adapter's own async write-window/staging (#817) → filesystem chimod
→ per-page loop → PutBlob`. Those layers reshape the traffic before it reaches
the PutBlob layer: the adapter stages writes and the chimod mediates them, so the
concentrated same-blob burst the direct client produced no longer **co-arrives**
at one worker's drain window. Worker-level batching can only merge what converges
there, and after the adapter/chimod have serialized the page, little does.

**Conclusion / next step.** The 3.68x is real for a client that issues concurrent
same-blob PutBlobs; it is not reachable by having the *runtime worker*
re-discover a burst the fs stack already dissolved. To move the fio numbers, the
coalescing must happen where the same-blob traffic actually converges — most
directly by having the **cfs adapter or filesystem chimod emit a vectored PutBlob
(part A's API) per page** instead of N single-region ones. Part A is exactly the
primitive that path needs; part B is the wrong layer for the POSIX workload.
