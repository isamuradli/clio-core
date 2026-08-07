/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "fuse_cte.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <string>
#include <vector>
#include <cerrno>                 // errno
#include <cstdio>                 // snprintf, fprintf
#include <cstdlib>                // getenv, strtod, strtoul
#include <map>                    // multimap (open-handle registry)
#include <mutex>                  // mutex, lock_guard
#include <atomic>                 // atomic (write-path counters)
#include <chrono>                 // steady_clock (write-path timing)
#include <fcntl.h>                // O_CREAT, O_RDWR
#ifdef __linux__
#include <linux/falloc.h>         // FALLOC_FL_* (fallocate mode flags)
#endif

#ifndef _WIN32
#include <sys/statvfs.h>          // struct statvfs (statfs op)
#include <unistd.h>               // read, getuid, getgid
#ifdef __APPLE__
#include <sys/mount.h>            // struct statfs (macFUSE's statfs callback type)
#endif
#endif  // _WIN32
// The mount / process-entry glue (fuse_main, the apptainer --fusemount custom-io
// path, and the fuse_lowlevel/dlsym/mount/uio machinery it needs) now lives in
// fuse_cte_main.cc. This file holds only the operation callbacks.

#include "clio_runtime/clio_runtime.h"
#include "clio_cte/core/content_transfer_engine.h"
#include "clio_cte/core/core_client.h"  // CLIO_CTE_CLIENT + GetCapacity
#include "clio_cte/filesystem/filesystem_client.h"

// Bridge the POSIX type spellings the callbacks use to the concrete types
// each platform's FUSE layer expects in `struct fuse_operations`. On Linux
// (libfuse) the high-level API uses the POSIX types directly; on Windows the
// shim maps them to WinFsp's fuse_* types and supplies getuid/getgid/S_IF*.
#ifdef _WIN32
#include "fuse_win_compat.h"
#else
using cte_stat_t = struct stat;
using cte_off_t = off_t;
using cte_mode_t = mode_t;
using cte_timespec_t = struct timespec;
#ifdef __APPLE__
// macFUSE's statfs callback reports through Darwin's struct statfs, not
// statvfs (the two share f_bsize/f_blocks/f_bfree/f_bavail/f_files/f_ffree;
// f_frsize/f_favail/f_namemax exist only on statvfs — see cte_fuse_statfs).
using cte_statvfs_t = struct statfs;
// Darwin spells the nanosecond stat members st_*timespec. Macro-alias the
// Linux spellings AFTER all system headers so member accesses in the
// callbacks (and in the test TU that #includes this file) resolve.
#define st_atim st_atimespec
#define st_mtim st_mtimespec
#define st_ctim st_ctimespec
#else
using cte_statvfs_t = struct statvfs;
#endif
#endif

using namespace clio::cae::fuse;

// ============================================================================
// Helpers
// ============================================================================
//
// The libfuse adapter is now a thin shim over the filesystem chimod (issue
// #552): every FUSE callback resolves to a path/handle operation on the
// process-wide filesystem client (CLIO_CFS_CLIENT), which owns the path->tag
// mapping, page-blob I/O, and per-file logical-size metadata. Reads/writes
// are synchronous from FUSE's perspective (the client Waits on each op), so
// there is no per-fd pending-write queue here anymore.

namespace {
/** Per-open-file state: the chimod handle + the path it was opened on. */
// ---- write coalescing (issue #933) -----------------------------------------
// The chimod's Write handler splits a write into kFsPageSize pages and awaits
// one AsyncPutBlob per page, and the pages of one file are named by offset --
// so a SEQUENTIAL 4 KiB write stream sends 256 separate puts to the SAME page,
// i.e. the same blob, which serializes them on that blob's write token. The
// runtime's own comment calls that token "the dominant clio-fs write-latency
// tail". Measured: 4 KiB writes 17.2 MiB/s against 1 MiB writes 304.7 MiB/s on
// the same mount, an 18x gap that is entirely per-blob contention rather than
// bandwidth.
//
// Neither layer above could fix it. Client-side deferral pipelines 256 writes
// deep, but they all queue on the one token, so it measured identical (17.2 vs
// 17.7 blocking). Kernel writeback measured WORSE (13.9) -- it adds a page
// cache without merging the upcalls.
//
// What does fix it is sending FEWER, BIGGER writes: accumulate contiguous
// writes here and hand the chimod one put per page. This is the same shape as
// JuiceFS's client buffer, and JuiceFS is the FUSE peer that outruns us 3x on
// this exact workload.
struct CfsHandle;

// Every open handle, keyed by path, so an operation on a FILE can flush
// buffers held by handles it does not own. Without this, buffered bytes would
// be visible only through the descriptor that wrote them -- a second open of
// the same file would read stale data, which is a POSIX violation rather than
// a performance tradeoff. Guarded by a single mutex: open/release are rare
// next to write, so this never contends on the hot path.
std::mutex g_open_mu;
std::multimap<std::string, CfsHandle *> g_open_by_path;

struct CfsHandle {
  clio::run::u64 fh = 0;
  std::string path;

  // Coalescing buffer. Holds [buf_off, buf_off+buf_len) of the file, always
  // within ONE kFsPageSize page so a flush is exactly one put.
  std::mutex mu;
  std::vector<char> buf;
  clio::run::u64 buf_off = 0;
  size_t buf_len = 0;
};

CfsHandle *GetHandle(struct fuse_file_info *fi) {
  return reinterpret_cast<CfsHandle *>(fi->fh);
}

// ---- write-path instrumentation (issue #933) -------------------------------
// Four separate fixes -- client-side deferral, kernel writeback, page
// coalescing -- each targeted a different layer and none moved 4 KiB writes off
// ~15 MiB/s. That is ~4,400 write upcalls/s, i.e. ~227 us each, against 10-20 us
// for an ordinary FUSE upcall; meanwhile 1 MiB writes are bandwidth-limited,
// not overhead-limited. Guessing which layer owns those 227 us has now been
// wrong twice, so measure it: count upcalls, count the puts actually submitted,
// and time both. Printed once at unmount. Zero cost when off.
std::atomic<uint64_t> g_w_calls{0}, g_w_bytes{0}, g_w_ns{0};
std::atomic<uint64_t> g_put_calls{0}, g_put_bytes{0}, g_put_ns{0};

// How often to emit. MUST be below the upcall count of the smallest run being
// measured or nothing is ever printed: a 64 MiB 4 KiB write phase is only
// 16,384 upcalls, and the daemon restarts per combination so counters never
// accumulate past one phase. A 20000 threshold therefore reported NOTHING for
// exactly the case under investigation. Configurable so a threshold mistake
// costs an env var rather than a 25-minute rebuild.
uint64_t WriteStatsEvery() {
  static const uint64_t v = [] {
    if (const char *e = std::getenv("CLIO_CTE_FUSE_WRITE_STATS_EVERY")) {
      char *end = nullptr;
      unsigned long long n = std::strtoull(e, &end, 10);
      if (end != e && n > 0) return (uint64_t)n;
    }
    return (uint64_t)2000;
  }();
  return v;
}

bool WriteStatsEnabled() {
  static const bool v = [] {
    const char *e = std::getenv("CLIO_CTE_FUSE_WRITE_STATS");
    return e != nullptr && std::string(e) != "0";
  }();
  return v;
}

void ReportWriteStats() {
  if (!WriteStatsEnabled()) return;
  const uint64_t c = g_w_calls.load(), b = g_w_bytes.load(), ns = g_w_ns.load();
  const uint64_t pc = g_put_calls.load(), pb = g_put_bytes.load(),
                 pns = g_put_ns.load();
  if (c == 0) return;
  fprintf(stderr,
          "clio_cte_fuse WRITE STATS: upcalls=%lu bytes=%lu avg_size=%lu "
          "avg_upcall_us=%.1f | puts=%lu put_bytes=%lu avg_put_size=%lu "
          "avg_put_us=%.1f | put_share_of_upcall=%.0f%%\n",
          (unsigned long)c, (unsigned long)b, (unsigned long)(b / c),
          (double)ns / c / 1000.0, (unsigned long)pc, (unsigned long)pb,
          (unsigned long)(pc ? pb / pc : 0),
          pc ? (double)pns / pc / 1000.0 : 0.0,
          ns ? 100.0 * (double)pns / (double)ns : 0.0);
}

bool CoalesceEnabled() {
  static const bool v = [] {
    const char *e = std::getenv("CLIO_CTE_FUSE_COALESCE");
    return e == nullptr || !(std::string(e) == "0" || std::string(e) == "false");
  }();
  return v;
}

// Submit whatever is buffered. Caller MUST hold h->mu.
// Returns 0, or a negative errno.
int FlushLocked(CfsHandle *h) {
  if (h->buf_len == 0) {
    return 0;
  }
  auto *cfs = CLIO_CFS_CLIENT;
  const size_t len = h->buf_len;
  const clio::run::u64 off = h->buf_off;
  // Clear BEFORE submitting: on failure the bytes are gone from the buffer
  // either way (the error is latched against the path and reported by
  // flush/fsync/close), and leaving them would let a retry double-write.
  h->buf_len = 0;
  const bool st = WriteStatsEnabled();
  const auto t0 = st ? std::chrono::steady_clock::now()
                     : std::chrono::steady_clock::time_point{};
  const ssize_t w = cfs->Write(h->fh, h->path, off, h->buf.data(), len, false);
  if (st) {
    g_put_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t0).count();
    g_put_calls++; g_put_bytes += len;
  }
  return (w < 0) ? -errno : 0;
}

int FlushHandle(CfsHandle *h) {
  std::lock_guard<std::mutex> lk(h->mu);
  return FlushLocked(h);
}

// Flush every open handle on `path`. Used by the read and metadata paths,
// which must observe bytes that are still sitting in some handle's buffer.
int FlushPathBuffers(const std::string &path) {
  std::vector<CfsHandle *> hs;
  {
    std::lock_guard<std::mutex> lk(g_open_mu);
    auto range = g_open_by_path.equal_range(path);
    for (auto it = range.first; it != range.second; ++it) {
      hs.push_back(it->second);
    }
  }
  int rc = 0;
  for (CfsHandle *h : hs) {
    const int r = FlushHandle(h);
    if (rc == 0) rc = r;
  }
  return rc;
}

// ---- mount tuning read from the environment (issue #933) -------------------
// Env rather than "-o" mount options: libfuse's own argv parser owns -o, and
// an unrecognized key there aborts the mount. A malformed VALUE here is a
// deployment mistake worth surfacing loudly, but not worth refusing to mount
// over -- warn and use the default, which is the historical behaviour.

double cte_fuse_env_double(const char *name, double dflt) {
  const char *v = std::getenv(name);
  if (v == nullptr || *v == '\0') {
    return dflt;
  }
  char *end = nullptr;
  const double parsed = std::strtod(v, &end);
  if (end == v || *end != '\0' || parsed < 0.0) {
    fprintf(stderr, "WARNING: %s='%s' is not a non-negative number; using %g\n",
            name, v, dflt);
    return dflt;
  }
  return parsed;
}

unsigned cte_fuse_env_uint(const char *name, unsigned dflt) {
  const char *v = std::getenv(name);
  if (v == nullptr || *v == '\0') {
    return dflt;
  }
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(v, &end, 10);
  if (end == v || *end != '\0' || parsed > UINT_MAX) {
    fprintf(stderr, "WARNING: %s='%s' is not an unsigned integer; using %u\n",
            name, v, dflt);
    return dflt;
  }
  return static_cast<unsigned>(parsed);
}

bool cte_fuse_env_bool(const char *name, bool dflt) {
  const char *v = std::getenv(name);
  if (v == nullptr || *v == '\0') {
    return dflt;
  }
  // Accept the spellings a YAML pipeline is likely to produce. Anything else
  // is a typo, and silently reading it as false would quietly disable the very
  // thing the operator was trying to switch on.
  const std::string s(v);
  if (s == "1" || s == "true" || s == "True" || s == "TRUE" ||
      s == "yes" || s == "on") {
    return true;
  }
  if (s == "0" || s == "false" || s == "False" || s == "FALSE" ||
      s == "no" || s == "off") {
    return false;
  }
  fprintf(stderr, "WARNING: %s='%s' is not a boolean; using %d\n", name, v,
          static_cast<int>(dflt));
  return dflt;
}
}  // namespace

// ============================================================================
// FUSE lifecycle
// ============================================================================

static void *cte_fuse_init(struct fuse_conn_info *conn,
                           struct fuse_config *cfg) {
  // Trust the inode numbers we report (st_ino in getattr, d_ino in readdir),
  // both derived from the tag id, instead of letting FUSE auto-generate them.
  // This makes stat and readdir agree on d_ino/st_ino (generic/637), and gives
  // hard-link aliases (which share a TagId) the same inode.
  cfg->use_ino = 1;
  // Keep the kernel page cache for file data (direct_io OFF). mmap on a FUSE
  // file is served generically by the page cache — there is no .mmap callback
  // in the high-level API; the kernel faults mapped pages through cte_fuse_read
  // and flushes dirty pages through cte_fuse_write. direct_io bypasses the page
  // cache, so the kernel returns ENODEV ("No such device") for any mmap (issue
  // #597).
  cfg->direct_io = 0;

  // ---- small-I/O amortization (issue #933) --------------------------------
  // Everything below defaults to the historical write-through, zero-cache
  // behaviour, so an unconfigured mount behaves exactly as before. They exist
  // because that behaviour costs a kernel->userspace upcall per operation,
  // which is the entire small-I/O deficit:
  //
  //   4 KiB, 1 rank, measured   CTE 15.0 MiB/s  =   3,840 ops/s
  //                             JuiceFS 540 MiB/s = 138,300 ops/s
  //
  // JuiceFS is also FUSE, on the same kernel, and does not sustain 138k
  // upcalls/s -- it ships attr/entry caches of 1.0s and a client-side write
  // buffer, so the kernel absorbs most operations and hands userspace far
  // fewer, larger ones. The gap is amortization, not FUSE and not the CTE data
  // path (at 1 MiB the two invert and CTE wins). Note also that a server-side
  // cache cannot help here: it sits on the far side of the upcall being
  // counted.
  //
  // These are env vars rather than mount options because libfuse's own argv
  // parser owns "-o"; the deployment sets them alongside CLIO_CTE_POOL.

  // Kernel attribute/entry caches. Zero means every getattr/lookup reaches the
  // chimod, which is the source of truth -- correct, and the reason for the
  // historical default: metadata (size, and especially st_nlink for hard
  // links) can change without this process being the one that triggered it,
  // and there is no invalidation upcall, so `ln a b; stat a` can return a's
  // stale cached nlink. A non-zero timeout trades exactly that window for the
  // upcalls. It also gates READ throughput, not just metadata: with
  // attr_timeout 0 the kernel revalidates on every access and drops the page
  // cache whenever size/mtime appear to move, so cached data never survives.
  cfg->attr_timeout = cte_fuse_env_double("CLIO_CTE_FUSE_ATTR_TIMEOUT", 0.0);
  cfg->entry_timeout = cte_fuse_env_double("CLIO_CTE_FUSE_ENTRY_TIMEOUT", 0.0);
  cfg->negative_timeout =
      cte_fuse_env_double("CLIO_CTE_FUSE_NEGATIVE_TIMEOUT", 0.0);

  // Writeback caching. Off, writes are write-through: every write() reaches
  // the chimod synchronously and the exact logical size is preserved (only
  // mmap dirty pages flush lazily, which is inherent to mmap). On, the kernel
  // buffers and coalesces, and becomes responsible for size -- so a partial
  // page write is padded to page granularity and the logical size a reader
  // observes is the kernel's, not the chimod's.
  if (cte_fuse_env_bool("CLIO_CTE_FUSE_WRITEBACK", false)) {
    if (conn->capable & FUSE_CAP_WRITEBACK_CACHE) {
      conn->want |= FUSE_CAP_WRITEBACK_CACHE;
    } else {
      fprintf(stderr,
              "WARNING: CLIO_CTE_FUSE_WRITEBACK set but the kernel does not "
              "advertise FUSE_CAP_WRITEBACK_CACHE; staying write-through\n");
    }
  }

  // Upcall SIZE and concurrency. Raising max_write lets one upcall carry more
  // bytes (the kernel caps this at its own limit, so an over-large request is
  // clamped, not rejected); max_background raises how many async requests may
  // be in flight before the kernel throttles the queue. Zero leaves the
  // libfuse/kernel default in place.
  const unsigned max_write_kib =
      cte_fuse_env_uint("CLIO_CTE_FUSE_MAX_WRITE_KIB", 0);
  if (max_write_kib > 0) {
    conn->max_write = max_write_kib * 1024u;
  }
  const unsigned max_background =
      cte_fuse_env_uint("CLIO_CTE_FUSE_MAX_BACKGROUND", 0);
  if (max_background > 0) {
    conn->max_background = max_background;
  }
  const unsigned max_readahead_kib =
      cte_fuse_env_uint("CLIO_CTE_FUSE_MAX_READAHEAD_KIB", 0);
  if (max_readahead_kib > 0) {
    conn->max_readahead = max_readahead_kib * 1024u;
  }

  // One line naming the whole tuning state. Without it a mount's cache
  // behaviour is invisible, and a benchmark cannot tell a tuned mount from an
  // untuned one -- which is exactly how the deficit above went unexplained.
  fprintf(stderr,
          "clio_cte_fuse: attr_timeout=%.3g entry_timeout=%.3g "
          "negative_timeout=%.3g writeback=%d max_write=%u "
          "max_background=%u max_readahead=%u\n",
          cfg->attr_timeout, cfg->entry_timeout, cfg->negative_timeout,
          (conn->want & FUSE_CAP_WRITEBACK_CACHE) ? 1 : 0, conn->max_write,
          conn->max_background, conn->max_readahead);

  bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
  if (!success) {
    fprintf(stderr, "ERROR: CLIO_INIT failed\n");
    return nullptr;
  }
  // Create-or-bind the filesystem chimod pool (which also brings up the CTE
  // core pool it sits over). Every FUSE op below routes through CLIO_CFS_CLIENT.
  if (!clio::cte::filesystem::CLIO_CFS_CLIENT_INIT()) {
    fprintf(stderr, "ERROR: filesystem client init failed\n");
    return nullptr;
  }
  // Bind the CTE core client to the same clio_cte_core pool (idempotent) so
  // statfs can query real capacity via GetCapacity.
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    fprintf(stderr, "WARNING: CTE core client init failed; statfs capacity=0\n");
  }
  return nullptr;
}

static void cte_fuse_destroy(void *private_data) {
  (void)private_data;
  ReportWriteStats();
  clio::run::CLIO_RUNTIME_FINALIZE();
}

// ============================================================================
// Metadata
// ============================================================================

// Decode a tag timestamp (stored as the two's-complement bits of an i64
// nanoseconds-since-epoch value in a u64 field) into a POSIX timespec. Using
// signed floor division makes pre-epoch (negative) times round-trip correctly
// — an unsigned divide turns e.g. Jan 1 1960 into a huge positive year
// (generic/258). For normal post-epoch times the value and result are
// identical to the old unsigned path (remainder is non-negative). tv_nsec is
// always normalized into [0, 1e9).
// NsecT is templated so this binds to the platform's timespec tv_nsec type:
// `long` on Linux/libfuse, `int64_t` on Windows/WinFsp (struct fuse_timespec).
template <typename NsecT>
static inline void NsBitsToTimespec(clio::run::u64 bits, time_t &sec,
                                    NsecT &nsec) {
  int64_t ns = static_cast<int64_t>(bits);
  int64_t s = ns / 1000000000LL;
  int64_t rem = ns % 1000000000LL;
  if (rem < 0) {
    s -= 1;
    rem += 1000000000LL;
  }
  sec = static_cast<time_t>(s);
  nsec = static_cast<NsecT>(rem);
}

static int cte_fuse_getattr_stat(const char *path, cte_stat_t *stbuf,
                                 struct fuse_file_info *fi) {
  (void)fi;
  memset(stbuf, 0, sizeof(*stbuf));

  std::string p(path);

  // Root is always a directory.
  if (p == "/") {
    stbuf->st_mode = S_IFDIR | 0755;
    stbuf->st_nlink = 2;
    stbuf->st_ino = 1;  // fixed root inode
    stbuf->st_uid = getuid();
    stbuf->st_gid = getgid();
    return 0;
  }

  // Delegate to the filesystem chimod: it owns exists/is-dir/logical-size.
  auto *cfs = CLIO_CFS_CLIENT;
  // Reconcile deferred writes BEFORE asking (issue #933). Writes now return
  // before they land, so a queued write has not advanced the published size.
  // Reporting the stale size here is not a cosmetic staleness: the kernel
  // caches st_size and CLAMPS READS TO IT, so a stat that under-reports turns
  // a subsequent read of just-written data into a short read or EOF.
  //
  // Client::GetAttr owns the decision of whether that requires waiting, and
  // makes it far better than an unconditional drain would: it serves from the
  // shared-memory mirror and only blocks when something in flight reaches PAST
  // the published end (DeferMaxPendingEnd > published). An overwrite cannot
  // change a file's size, and overwrites are most of what a steady-state
  // workload does, so the common case stays IPC-free. Draining on any pending
  // write instead — the obvious version — measured ~264 us per stat.
  //
  // Its answers are deliberately discarded: this callback needs mode, nlink,
  // times and ownership too, so the authoritative read is still AsyncGetattr
  // below. What the call buys is the ordering — by the time it returns, any
  // write that could have grown the file has landed.
  FlushPathBuffers(p);
  bool exists = false;
  clio::run::u64 pending_size = 0;
  (void)cfs->GetAttr(p, &exists, &pending_size);
  auto t = cfs->AsyncGetattr(p);
  t.Wait();
  if (t->GetReturnCode() != 0 || t->exists_ == 0) {
    return -ENOENT;
  }
  // Owner: a prior chown recorded an override (uid_/gid_ != 0xFFFFFFFF);
  // otherwise report the mounting user's uid/gid (files carry no stored owner).
  stbuf->st_uid =
      (t->uid_ != 0xFFFFFFFFu) ? static_cast<uid_t>(t->uid_) : getuid();
  stbuf->st_gid =
      (t->gid_ != 0xFFFFFFFFu) ? static_cast<gid_t>(t->gid_) : getgid();
  stbuf->st_ino = static_cast<ino_t>(t->ino_);  // stable inode = packed TagId
  // A chmod/create recorded permission bits (t->mode_ != 0xFFFFFFFF) win over
  // the synthesized defaults; keep only the low 12 bits (perms + setuid/gid/sticky).
  const bool have_mode = (t->mode_ != 0xFFFFFFFFu);
  const unsigned int perm = t->mode_ & 07777u;
  if (t->is_dir_) {
    stbuf->st_mode = S_IFDIR | (have_mode ? perm : 0755u);
    stbuf->st_nlink = 2;
  } else if (t->is_symlink_) {
    stbuf->st_mode = S_IFLNK | 0777;
    stbuf->st_nlink = 1;
    stbuf->st_size = static_cast<cte_off_t>(t->size_);  // target length
  } else {
    stbuf->st_mode = S_IFREG | (have_mode ? perm : 0644u);
    // POSIX link count = canonical name (1) + tag-level hard-link aliases.
    // Ask the CTE core how many extra names are bound to this tag.
    nlink_t nlink = 1;
    auto *cte = CLIO_CTE_CLIENT;
    if (cte != nullptr) {
      auto na = cte->AsyncGetNumAliases(p);
      na.Wait();
      if (na->return_code_ == 0 && na->found_) {
        nlink = static_cast<nlink_t>(na->num_aliases_) + 1;
      }
    }
    stbuf->st_nlink = nlink;
    // cte_off_t is off_t on Linux; the WinFsp shim maps it for Windows.
    stbuf->st_size = static_cast<cte_off_t>(t->size_);
  }
  // Report the 512-byte block count backing the file so stat(2) st_blocks is
  // non-zero for files that hold data (generic/615 asserts a buffered/direct
  // write shows allocated blocks). Derived from the logical size; directories
  // report 0. st_blksize advertises a sensible I/O unit for tools.
  stbuf->st_blksize = static_cast<decltype(stbuf->st_blksize)>(4096);
  stbuf->st_blocks = static_cast<decltype(stbuf->st_blocks)>(
      (static_cast<uint64_t>(stbuf->st_size) + 511) / 512);
  // Timestamps come from the tag as ns since the epoch (0 means the chimod had
  // no value, so leave that field at the epoch): ctime = last metadata change
  // (last_changed_), mtime = last content change (last_modified_), atime = last
  // access (last_read_). All three are surfaced from the same GetTagSize query.
  if (t->ctime_ != 0) {
    NsBitsToTimespec(t->ctime_, stbuf->st_ctim.tv_sec, stbuf->st_ctim.tv_nsec);
  }
  // Fall back to ctime when mtime is unknown, so a valid file never reports
  // mtime at the epoch while it has a real ctime (merged from #680).
  clio::run::u64 mtime_ns = (t->mtime_ != 0) ? t->mtime_ : t->ctime_;
  if (mtime_ns != 0) {
    NsBitsToTimespec(mtime_ns, stbuf->st_mtim.tv_sec, stbuf->st_mtim.tv_nsec);
  }
  if (t->atime_ != 0) {
    NsBitsToTimespec(t->atime_, stbuf->st_atim.tv_sec, stbuf->st_atim.tv_nsec);
  }
  return 0;
}

#ifdef __APPLE__
// macFUSE's high-level getattr callback fills a fuse_darwin_attr, not a
// struct stat. Compute into a struct stat (== cte_stat_t here) and translate.
static void CopyStatToDarwinAttr(const struct stat &st,
                                 struct fuse_darwin_attr *attr) {
  memset(attr, 0, sizeof(*attr));
  attr->ino = st.st_ino;
  attr->mode = st.st_mode;
  attr->nlink = st.st_nlink;
  attr->uid = st.st_uid;
  attr->gid = st.st_gid;
  attr->rdev = st.st_rdev;
  attr->size = st.st_size;
  attr->blocks = st.st_blocks;
  attr->blksize = st.st_blksize;
  attr->flags = st.st_flags;
  attr->atimespec = st.st_atimespec;
  attr->mtimespec = st.st_mtimespec;
  attr->ctimespec = st.st_ctimespec;
  attr->btimespec = st.st_birthtimespec;
}

static int cte_fuse_getattr(const char *path, struct fuse_darwin_attr *attr,
                            struct fuse_file_info *fi) {
  struct stat stbuf;
  int rc = cte_fuse_getattr_stat(path, &stbuf, fi);
  if (rc != 0) return rc;
  CopyStatToDarwinAttr(stbuf, attr);
  return 0;
}
#else
// Linux (struct stat) and Windows (WinFsp stat via cte_stat_t).
static int cte_fuse_getattr(const char *path, cte_stat_t *stbuf,
                            struct fuse_file_info *fi) {
  return cte_fuse_getattr_stat(path, stbuf, fi);
}
#endif

static int cte_fuse_utimens(const char *path, const cte_timespec_t tv[2],
                            struct fuse_file_info *fi) {
  (void)fi;
  // Translate the POSIX (atime, mtime) timespec pair into the chimod's flag
  // encoding: bit0/bit1 = explicit atime/mtime, bit2/bit3 = UTIME_NOW (resolved
  // server-side so it shares the tag clock). UTIME_OMIT leaves a field alone.
  clio::run::u32 flags = 0;
  clio::run::u64 atime_ns = 0, mtime_ns = 0;
#if defined(UTIME_NOW) && defined(UTIME_OMIT)
  if (tv != nullptr) {
    if (tv[0].tv_nsec == UTIME_NOW) {
      flags |= 0x4u;
    } else if (tv[0].tv_nsec != UTIME_OMIT) {
      flags |= 0x1u;
      // Signed arithmetic so pre-epoch times don't wrap (generic/258); stored
      // as the two's-complement bits, decoded symmetrically in NsBitsToTimespec.
      atime_ns = static_cast<clio::run::u64>(
          static_cast<int64_t>(tv[0].tv_sec) * 1000000000LL +
          static_cast<int64_t>(tv[0].tv_nsec));
    }
    if (tv[1].tv_nsec == UTIME_NOW) {
      flags |= 0x8u;
    } else if (tv[1].tv_nsec != UTIME_OMIT) {
      flags |= 0x2u;
      mtime_ns = static_cast<clio::run::u64>(
          static_cast<int64_t>(tv[1].tv_sec) * 1000000000LL +
          static_cast<int64_t>(tv[1].tv_nsec));
    }
  } else {
    // NULL tv means "set both to now".
    flags |= 0x4u | 0x8u;
  }
#else
  // Platform without UTIME_NOW/OMIT: treat as set-both-to-now.
  (void)tv;
  flags |= 0x4u | 0x8u;
#endif
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncUtimens(std::string(path), atime_ns, mtime_ns, flags);
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  return rc == 0 ? 0 : -rc;
}

// chmod records the permission bits as a per-file override in the chimod
// (surfaced by getattr), mirroring chown. Storing the mode lets +x stick, so a
// binary copied onto the fs can be executed (generic/452) and chmod-then-proceed
// callers (e.g. mount's mtab updater in generic/089) still succeed. AsyncChmod
// resolves the path and returns ENOENT for a missing file, so no separate
// existence probe is needed.
static int cte_fuse_chmod(const char *path, cte_mode_t mode,
                          struct fuse_file_info *fi) {
  (void)fi;
  auto *cfs = CLIO_CFS_CLIENT;
  std::string p(path);
  // Probe existence first: the chmod path resolves via GetOrCreateTag, so a
  // missing target would otherwise be silently created. chmod(2) must ENOENT.
  auto g = cfs->AsyncGetattr(p);
  g.Wait();
  if (g->GetReturnCode() != 0 || g->exists_ == 0) return -ENOENT;
  auto t = cfs->AsyncChmod(p, static_cast<clio::run::u32>(mode) & 07777u);
  t.Wait();
  return t->GetReturnCode() == 0 ? 0 : -EIO;
}

// chown records a per-file owner uid/gid override in the chimod, surfaced by
// getattr (files carry no stored POSIX owner otherwise). A uid/gid of
// (uid_t)-1 == 0xFFFFFFFF means "leave that field unchanged" (POSIX), which is
// exactly the chimod's "unchanged" sentinel, so no translation is needed.
static int cte_fuse_chown(const char *path, uid_t uid, gid_t gid,
                          struct fuse_file_info *fi) {
  (void)fi;
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncChown(std::string(path),
                           static_cast<clio::run::u32>(uid),
                           static_cast<clio::run::u32>(gid));
  t.Wait();
  return t->GetReturnCode() == 0 ? 0 : -EIO;
}

// ============================================================================
// Directory operations
// ============================================================================

#ifdef __APPLE__
using ClioFuseFillDirT = fuse_darwin_fill_dir_t;
#else
using ClioFuseFillDirT = fuse_fill_dir_t;
#endif

static int cte_fuse_readdir(const char *path, void *buf,
                            ClioFuseFillDirT filler, cte_off_t offset,
                            struct fuse_file_info *fi,
                            enum fuse_readdir_flags flags) {
  (void)offset;
  (void)fi;
  (void)flags;

  std::string p(path);

  filler(buf, ".", nullptr, 0, static_cast<fuse_fill_dir_flags>(0));
  filler(buf, "..", nullptr, 0, static_cast<fuse_fill_dir_flags>(0));

  // Delegate listing to the filesystem chimod. It returns the full tag paths
  // of the directory's children; strip the directory prefix to get basenames.
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncReaddir(p);
  t.Wait();
  if (t->GetReturnCode() != 0) {
    return 0;
  }
  size_t prefix_len = p.size();
  if (!p.empty() && p.back() != '/') prefix_len++;
  // entries_ and inos_ are index-aligned (the chimod builds them together).
  for (size_t i = 0; i < t->entries_.size(); ++i) {
    std::string full = t->entries_[i].str();
    std::string name = full.size() > prefix_len ? full.substr(prefix_len) : full;
    // A child sentinel directory may come back as "<dir>/<name>/"; drop the
    // trailing slash so it shows as a plain directory entry.
    if (!name.empty() && name.back() == '/') name.pop_back();
    if (name.empty()) continue;
    // Supply only d_ino (the child's tag-derived inode) so getdents agrees with
    // a subsequent stat (generic/637). Leave st_mode = 0 (DT_UNKNOWN): the entry
    // type is not reliably known here, so the kernel issues a getattr to resolve
    // it — setting a wrong d_type would mislead `rm -rf`/`find`.
    cte_stat_t st;
    memset(&st, 0, sizeof(st));
    st.st_ino = i < t->inos_.size() ? static_cast<ino_t>(t->inos_[i]) : 0;
#ifdef __APPLE__
    // macFUSE's fill-dir callback consumes a fuse_darwin_attr, not a
    // struct stat — translate (same mapping getattr uses).
    struct fuse_darwin_attr attr;
    CopyStatToDarwinAttr(st, &attr);
    filler(buf, name.c_str(), &attr, 0, static_cast<fuse_fill_dir_flags>(0));
#else
    filler(buf, name.c_str(), &st, 0, static_cast<fuse_fill_dir_flags>(0));
#endif
  }
  return 0;
}

static int cte_fuse_mkdir(const char *path, cte_mode_t mode) {
  (void)mode;
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncMkdir(std::string(path));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());  // errno-style (0/EEXIST/EIO)
  return rc == 0 ? 0 : -rc;
}

static int cte_fuse_rmdir(const char *path) {
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncRmdir(std::string(path));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());  // 0/ENOTEMPTY/ENOENT/EIO
  return rc == 0 ? 0 : -rc;
}

// ============================================================================
// File lifecycle
// ============================================================================

// O_DIRECT needs no special handling: we deliberately do NOT set the per-file
// direct_io flag. Our read/write handlers already work at arbitrary offsets and
// sizes, so an O_DIRECT open flows through the exact same buffered page-cache
// path as a regular open — it is never rejected. (Enabling per-file direct_io
// would make the kernel return ENODEV for any mmap of an O_DIRECT fd, breaking
// programs that both O_DIRECT and mmap the same file, e.g. fsx -Z; see #597.)

// Honor O_TRUNC: the chimod open resolves the tag but does not truncate, so an
// open/creat of an existing file with O_TRUNC would keep its old page-blobs
// (leaving stale data an app expects to be gone — e.g. reads of a re-created
// file's holes). Clear it to zero length here, which frees those blobs.
static inline void MaybeTruncateOnOpen(clio::cte::filesystem::Client *cfs,
                                       const std::string &p, int flags) {
  if (flags & O_TRUNC) {
    auto tr = cfs->AsyncTruncate(p, 0);
    tr.Wait();
  }
}

static int cte_fuse_create(const char *path, cte_mode_t mode,
                           struct fuse_file_info *fi) {
  std::string p(path);
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncOpen(p, O_CREAT | O_RDWR, static_cast<clio::run::u32>(mode));
  t.Wait();
  if (t->GetReturnCode() != 0) return -EIO;

  auto *handle = new CfsHandle();
  handle->fh = t->handle_;
  handle->path = p;
  {
    std::lock_guard<std::mutex> lk(g_open_mu);
    g_open_by_path.emplace(handle->path, handle);
  }
  fi->fh = reinterpret_cast<uint64_t>(handle);
  MaybeTruncateOnOpen(cfs, p, fi->flags);
  return 0;
}

static int cte_fuse_open(const char *path, struct fuse_file_info *fi) {
  std::string p(path);
  auto *cfs = CLIO_CFS_CLIENT;
  // The chimod honors O_CREAT: a plain open of a missing file returns
  // handle==0 so we can surface ENOENT.
  auto t = cfs->AsyncOpen(p, static_cast<clio::run::u32>(fi->flags), 0644);
  t.Wait();
  if (t->GetReturnCode() != 0) return -EIO;
  if (t->handle_ == 0) return -ENOENT;

  auto *handle = new CfsHandle();
  handle->fh = t->handle_;
  handle->path = p;
  {
    std::lock_guard<std::mutex> lk(g_open_mu);
    g_open_by_path.emplace(handle->path, handle);
  }
  fi->fh = reinterpret_cast<uint64_t>(handle);
  MaybeTruncateOnOpen(cfs, p, fi->flags);
  return 0;
}

// Writes are DEFERRED (issue #933): cte_fuse_write submits and returns without
// waiting, so these are the points where outstanding writes are made durable
// and where a failed write is finally reported. They were `return 0` no-ops,
// which was correct only while every write blocked. Leaving them that way once
// writes went async would mean close(2) could return before the data existed,
// and fsync(2) would be a lie.
//
// Client::Flush waits for every deferred write to `path` and takes the errno
// they latched, reporting it ONCE -- the same discipline as the kernel page
// cache, where a writeback failure surfaces at fsync rather than at some
// unrelated later write(2).
static int cte_fuse_flush(const char *path, struct fuse_file_info *fi) {
  auto *handle = GetHandle(fi);
  const std::string p = handle ? handle->path : std::string(path ? path : "");
  if (p.empty()) return 0;
  // Buffer first, then the deferred queue: a buffered write is not yet even
  // submitted, so draining the queue without it would report success for data
  // that has not been sent.
  const int brc = FlushPathBuffers(p);
  auto *cfs = CLIO_CFS_CLIENT;
  const int frc = cfs->Flush(p) == 0 ? 0 : -errno;
  return brc != 0 ? brc : frc;
}

// datasync is ignored deliberately: the deferred queue holds data writes, and
// this filesystem has no separately-deferred metadata to skip, so the
// datasync-only case has nothing cheaper to do than the full drain.
static int cte_fuse_fsync(const char *path, int /*datasync*/,
                          struct fuse_file_info *fi) {
  auto *handle = GetHandle(fi);
  const std::string p = handle ? handle->path : std::string(path ? path : "");
  if (p.empty()) return 0;
  // Buffer first, then the deferred queue: a buffered write is not yet even
  // submitted, so draining the queue without it would report success for data
  // that has not been sent.
  const int brc = FlushPathBuffers(p);
  auto *cfs = CLIO_CFS_CLIENT;
  const int frc = cfs->Flush(p) == 0 ? 0 : -errno;
  return brc != 0 ? brc : frc;
}

static int cte_fuse_release(const char *path, struct fuse_file_info *fi) {
  (void)path;
  auto *handle = GetHandle(fi);
  if (!handle) return 0;
  auto *cfs = CLIO_CFS_CLIENT;
  // Drain THIS file's deferred writes before closing its handle. The kernel
  // issues flush before release, but not on every path (and never for a
  // handle torn down without one), so closing a handle whose writes are still
  // queued would strand them against a dead fh. Take the error too -- this is
  // the last chance to report a write that failed after being acked.
  int rc = FlushHandle(handle);
  {
    std::lock_guard<std::mutex> lk(g_open_mu);
    auto range = g_open_by_path.equal_range(handle->path);
    for (auto it = range.first; it != range.second; ++it) {
      if (it->second == handle) { g_open_by_path.erase(it); break; }
    }
  }
  if (rc == 0) rc = (cfs->Flush(handle->path) == 0) ? 0 : -errno;
  auto t = cfs->AsyncClose(handle->fh);
  t.Wait();
  if (rc == 0 && t->GetReturnCode() != 0) {
    rc = -EIO;
  }
  delete handle;
  fi->fh = 0;
  return rc;
}

// ============================================================================
// Read / Write — delegated to the chimod's page-based I/O
// ============================================================================

static int cte_fuse_read(const char *path, char *buf, size_t size,
                         cte_off_t offset, struct fuse_file_info *fi) {
  (void)path;
  auto *handle = GetHandle(fi);
  if (!handle) return -EBADF;

  if (size > static_cast<size_t>(INT_MAX))
    size = static_cast<size_t>(INT_MAX);
  if (size == 0) return 0;

  // Client::Read, not a hand-rolled AsyncRead().Wait() (issue #933). Two
  // reasons, and the first is correctness rather than speed:
  //
  //  1. Now that writes are deferred, the authoritative bytes for a just
  //     written region may still be sitting in a pending write's staging
  //     buffer. Client::Read checks that first (DeferTryServe), so a read
  //     after a write sees the value written. Going straight to the chimod, as
  //     this did, would read the PRE-WRITE bytes -- silent corruption, not a
  //     slow path.
  //  2. It is also strictly cheaper: three tiers, cheapest first -- bytes in
  //     flight are copied out of staging (no wait, no IPC), committed bytes
  //     come from the shared-memory mirror (no IPC), and only what neither
  //     covers costs an RPC. The old path paid an SHM allocation and a full
  //     round trip unconditionally.
  // Buffered writes are not visible to the client yet (see CfsHandle), so a
  // read must push them out first -- including buffers held by OTHER handles
  // on this file, which is why the registry exists.
  FlushPathBuffers(handle->path);
  auto *cfs = CLIO_CFS_CLIENT;
  const ssize_t got = cfs->Read(handle->fh, handle->path,
                                static_cast<clio::run::u64>(offset), buf, size);
  if (got < 0) {
    return -errno;
  }
  return static_cast<int>(got);
}

static int cte_fuse_write_inner(const char *path, const char *buf, size_t size,
                                cte_off_t offset, struct fuse_file_info *fi);

// Thin timing wrapper so the measured span is exactly what FUSE hands us.
static int cte_fuse_write(const char *path, const char *buf, size_t size,
                          cte_off_t offset, struct fuse_file_info *fi) {
  if (!WriteStatsEnabled()) {
    return cte_fuse_write_inner(path, buf, size, offset, fi);
  }
  const auto t0 = std::chrono::steady_clock::now();
  const int rc = cte_fuse_write_inner(path, buf, size, offset, fi);
  g_w_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - t0).count();
  const uint64_t n = ++g_w_calls;
  if (rc > 0) g_w_bytes += (uint64_t)rc;
  // Report PERIODICALLY, not only from cte_fuse_destroy. The deployment stops
  // this daemon with SIGKILL (jarvis Kill()), so destroy never runs and a
  // destroy-only report is silently lost -- which is exactly what happened on
  // the first instrumented run.
  if (n % WriteStatsEvery() == 0) ReportWriteStats();
  return rc;
}

static int cte_fuse_write_inner(const char *path, const char *buf, size_t size,
                                cte_off_t offset, struct fuse_file_info *fi) {
  (void)path;
  auto *handle = GetHandle(fi);
  if (!handle) return -EBADF;

  if (size > static_cast<size_t>(INT_MAX))
    size = static_cast<size_t>(INT_MAX);
  if (size == 0) return 0;

  // Client::Write, not a hand-rolled AsyncWrite().Wait() (issue #933). The
  // client owns the deferred-write pipeline and this adapter was bypassing all
  // of it: it allocated a fresh SHM buffer per write, copied, then BLOCKED on
  // the round trip. Every small write therefore cost an allocator walk plus a
  // full chimod round trip, serialized -- which is the entire small-I/O write
  // deficit. Client::Write instead stages through the recycled pool (issue
  // #892 measured cold allocation as THE submit bottleneck), submits, and
  // returns without waiting; flow control is real shared-memory capacity
  // rather than one-at-a-time.
  //
  // `sync` honours O_SYNC/O_DSYNC: that caller asked for durability over
  // latency, and Client::Write blocks for exactly those.
  //
  // A deferred write ALWAYS reports success here. A queued write that later
  // fails latches its errno against the path, and cte_fuse_flush/cte_fuse_fsync
  // report it -- the same discipline the kernel page cache uses, where
  // writeback errors surface at fsync rather than at an unrelated later
  // write(2). Those two callbacks were `return 0` no-ops, which was harmless
  // only while every write blocked; they are real now.
  const bool sync = (fi->flags & (O_SYNC | O_DSYNC)) != 0;
  auto *cfs = CLIO_CFS_CLIENT;
  const clio::run::u64 off = static_cast<clio::run::u64>(offset);
  const clio::run::u64 kPage = clio::cte::filesystem::kFsPageSize;

  // Coalesce contiguous writes into whole pages before submitting (see the
  // note on CfsHandle). O_SYNC skips it: that caller asked for durability over
  // latency, and buffering is the opposite of that. A write already >= a page
  // skips it too -- it is a whole put by itself and buffering would add a copy
  // for nothing.
  if (CoalesceEnabled() && !sync && size < kPage) {
    std::lock_guard<std::mutex> lk(handle->mu);
    // Not contiguous with what is buffered? The buffer describes one extent,
    // so the old one has to go out before this one can start.
    if (handle->buf_len != 0 &&
        off != handle->buf_off + handle->buf_len) {
      const int rc = FlushLocked(handle);
      if (rc != 0) return rc;
    }
    if (handle->buf_len == 0) {
      handle->buf_off = off;
    }
    // Never let the buffer straddle a page: the chimod would split it into two
    // puts and we would be back to multiple writes hitting one blob, which is
    // the entire problem.
    const clio::run::u64 page_end =
        (handle->buf_off / kPage + 1) * kPage;
    const size_t room =
        static_cast<size_t>(page_end - (handle->buf_off + handle->buf_len));
    const size_t take = std::min(size, room);
    if (handle->buf.size() < static_cast<size_t>(kPage)) {
      handle->buf.resize(static_cast<size_t>(kPage));
    }
    memcpy(handle->buf.data() + handle->buf_len, buf, take);
    handle->buf_len += take;
    if (handle->buf_off + handle->buf_len >= page_end) {
      const int rc = FlushLocked(handle);
      if (rc != 0) return rc;
    }
    // The remainder starts the next page; recurse once through the same path.
    if (take < size) {
      handle->buf_off = off + take;
      const size_t rest = size - take;
      if (handle->buf.size() < static_cast<size_t>(kPage)) {
        handle->buf.resize(static_cast<size_t>(kPage));
      }
      memcpy(handle->buf.data(), buf + take, rest);
      handle->buf_len = rest;
    }
    // Buffered writes ALWAYS report success, exactly as deferred ones do: a
    // failure is latched against the path and reported by flush/fsync/close.
    return static_cast<int>(size);
  }

  // Uncoalesced path: flush anything buffered first so this write cannot
  // overtake bytes written before it.
  {
    std::lock_guard<std::mutex> lk(handle->mu);
    const int rc = FlushLocked(handle);
    if (rc != 0) return rc;
  }
  const ssize_t written = cfs->Write(handle->fh, handle->path, off, buf,
                                     size, sync);
  if (written < 0) {
    return -errno;
  }
  return static_cast<int>(written);
}

// ============================================================================
// Unlink / Truncate
// ============================================================================

static int cte_fuse_unlink(const char *path) {
  auto *cfs = CLIO_CFS_CLIENT;
  const std::string p(path);
  // Drain first (issue #933): a queued write landing after the unlink writes
  // to a tag that no longer exists, and its latched error would then surface
  // against a path the caller has already deleted. Also keeps the deferred
  // registry from holding entries keyed to a dead file.
  FlushPathBuffers(p);
  (void)cfs->Flush(p);
  auto t = cfs->AsyncUnlink(p);
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());  // 0/EISDIR/EIO
  return rc == 0 ? 0 : -rc;
}

static int cte_fuse_truncate(const char *path, cte_off_t size,
                             struct fuse_file_info *fi) {
  (void)fi;
  auto *cfs = CLIO_CFS_CLIENT;
  const std::string p(path);
  // Drain first (issue #933). Deferred writes are ordered against each other
  // but NOT against this: a queued write past `size` that lands after the
  // truncate re-grows the file and resurrects bytes the caller just discarded.
  // Unlike stat, there is no cheap partial answer here -- the whole point is
  // that nothing may be in flight when the length changes -- so this drains
  // unconditionally. ftruncate is rare next to write(2); correctness wins.
  FlushPathBuffers(p);
  (void)cfs->Flush(p);
  auto t = cfs->AsyncTruncate(p, static_cast<clio::run::u64>(size));
  t.Wait();
  return t->GetReturnCode() == 0 ? 0 : -EIO;
}

#ifdef __linux__
// Write `len` zero bytes at `off` through an open handle, chunked so a large
// range doesn't need one giant SHM buffer. Used by ZERO_RANGE. Returns 0 or a
// negative errno.
static int cte_fuse_write_zeros(CfsHandle *handle, cte_off_t off,
                                cte_off_t len) {
  auto *ipc = CLIO_IPC;
  auto *cfs = CLIO_CFS_CLIENT;
  constexpr cte_off_t kChunk = 1 << 20;  // 1 MiB
  const cte_off_t buf_sz = (len < kChunk) ? len : kChunk;
  ctp::ipc::FullPtr<char> zbuf = ipc->AllocateBuffer(buf_sz);
  if (zbuf.IsNull()) return -ENOMEM;
  memset(zbuf.ptr_, 0, static_cast<size_t>(buf_sz));
  int result = 0;
  for (cte_off_t done = 0; done < len;) {
    const cte_off_t n = ((len - done) < buf_sz) ? (len - done) : buf_sz;
    auto t = cfs->AsyncWrite(handle->fh, static_cast<clio::run::u64>(off + done),
                             static_cast<clio::run::u64>(n),
                             ctp::ipc::ShmPtr<>(zbuf.shm_));
    t.Wait();
    if (t->GetReturnCode() != 0) { result = -EIO; break; }
    done += n;
  }
  ipc->FreeBuffer(zbuf);
  return result;
}

// fallocate — page-blobs are created lazily on write and holes read back as
// zeros, so we never reserve storage ahead of time. Supported modes:
//   * mode==0            : grow EOF to offset+length (never shrinks).
//   * FALLOC_FL_KEEP_SIZE: no-op success (nothing to reserve).
//   * FALLOC_FL_ZERO_RANGE: make [offset,offset+length) read as zeros by
//     writing zeros through the open handle (with KEEP_SIZE, restore EOF after
//     if the write extended it). This is a correct ZERO_RANGE for our
//     hole-reads-as-zero model and unblocks the fzero xfstests.
// Layout-shifting modes (punch hole, collapse/insert range) would need a
// chimod-level block-dealloc/shift op and still return EOPNOTSUPP.
static int cte_fuse_fallocate(const char *path, int mode, cte_off_t offset,
                              cte_off_t length, struct fuse_file_info *fi) {
  const int kSupportedModes = FALLOC_FL_KEEP_SIZE | FALLOC_FL_ZERO_RANGE;
  if (mode & ~kSupportedModes) {
    return -EOPNOTSUPP;  // punch/collapse/insert: layout-changing
  }
  if (offset < 0 || length <= 0) {
    return -EINVAL;
  }

  if (mode & FALLOC_FL_ZERO_RANGE) {
    auto *handle = GetHandle(fi);
    if (!handle) return -EBADF;
    // Original size, so KEEP_SIZE can restore EOF if the zero-write grows it.
    cte_stat_t st;
    int rc = cte_fuse_getattr_stat(path, &st, fi);
    if (rc != 0) return rc;
    rc = cte_fuse_write_zeros(handle, offset, length);
    if (rc != 0) return rc;
    if ((mode & FALLOC_FL_KEEP_SIZE) && (offset + length) > st.st_size) {
      auto *cfs = CLIO_CFS_CLIENT;
      auto t = cfs->AsyncTruncate(std::string(path),
                                  static_cast<clio::run::u64>(st.st_size));
      t.Wait();
      if (t->GetReturnCode() != 0) return -EIO;
    }
    return 0;
  }

  if (mode & FALLOC_FL_KEEP_SIZE) {
    return 0;  // no size change requested, and there is nothing to reserve
  }

  // mode == 0: extend EOF to offset+length if the file is currently shorter.
  cte_stat_t st;
  int rc = cte_fuse_getattr_stat(path, &st, fi);
  if (rc != 0) return rc;
  cte_off_t need = offset + length;
  if (need <= st.st_size) return 0;  // already large enough; never shrink

  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncTruncate(std::string(path),
                              static_cast<clio::run::u64>(need));
  t.Wait();
  return t->GetReturnCode() == 0 ? 0 : -EIO;
}
#endif  // __linux__

static int cte_fuse_link(const char *from, const char *to) {
  // Hard link `to` -> existing file `from`. The chimod binds both names to the
  // same CTE tag (a tag-level alias), so they share all data.
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncLink(std::string(from), std::string(to));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  return rc == 0 ? 0 : -rc;  // chimod returns errno-style codes
}

static int cte_fuse_symlink(const char *target, const char *path) {
  // Create a symlink at `path` pointing at `target`. The chimod stores the
  // target string in a reserved marker blob under `path`'s tag.
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncSymlink(std::string(target), std::string(path));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());  // 0/EEXIST/EIO
  return rc == 0 ? 0 : -rc;  // chimod returns errno-style codes
}

static int cte_fuse_readlink(const char *path, char *buf, size_t size) {
  // Read the symlink target into `buf` (NUL-terminated). FUSE readlink returns
  // 0 on success (not the length).
  if (size == 0) {
    return -EINVAL;
  }
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncReadlink(std::string(path));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());  // 0/ENOENT/EINVAL
  if (rc != 0) {
    return -rc;
  }
  std::string target = t->target_.str();
  size_t n = std::min(target.size(), size - 1);
  std::memcpy(buf, target.data(), n);
  buf[n] = '\0';
  return 0;
}

static int cte_fuse_setxattr(const char *path, const char *name,
                             const char *value, size_t size, int flags) {
  // Set xattr `name` on `path`. `value` is raw bytes (may contain NULs), so
  // preserve its length rather than treating it as a C string. `flags` carries
  // XATTR_CREATE(1) / XATTR_REPLACE(2), matching the runtime's bit checks.
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncSetxattr(std::string(path), std::string(name),
                              std::string(value, size),
                              static_cast<unsigned int>(flags));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  return rc == 0 ? 0 : -rc;  // EEXIST/ENODATA/ENOENT/EIO -> negative errno
}

static int cte_fuse_getxattr(const char *path, const char *name, char *value,
                             size_t size) {
  // Read xattr `name` of `path`. Return the value length (POSIX getxattr);
  // size==0 is a length query. Missing attribute -> -ENODATA.
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncGetxattr(std::string(path), std::string(name));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  if (rc != 0) {
    return -rc;  // ENOENT (file absent)
  }
  if (t->found_ == 0) {
    return -ENODATA;  // attribute not present
  }
  std::string val = t->value_.str();
  size_t len = val.size();
  if (size == 0) {
    return static_cast<int>(len);  // length query
  }
  if (size < len) {
    return -ERANGE;
  }
  std::memcpy(value, val.data(), len);
  return static_cast<int>(len);
}

#ifdef __APPLE__
// macFUSE's xattr callbacks carry an extra `position` argument (resource-fork
// offset for the com.apple.ResourceFork attribute). We store xattrs whole, so
// only position 0 is meaningful; reject sub-range access like most non-HFS
// FUSE filesystems do.
static int cte_fuse_setxattr_darwin(const char *path, const char *name,
                                    const char *value, size_t size, int flags,
                                    uint32_t position) {
  if (position != 0) {
    return -EINVAL;
  }
  return cte_fuse_setxattr(path, name, value, size, flags);
}

static int cte_fuse_getxattr_darwin(const char *path, const char *name,
                                    char *value, size_t size,
                                    uint32_t position) {
  if (position != 0) {
    return -EINVAL;
  }
  return cte_fuse_getxattr(path, name, value, size);
}
#endif  // __APPLE__

static int cte_fuse_listxattr(const char *path, char *list, size_t size) {
  // Return the NUL-separated, NUL-terminated list of xattr names. size==0 is a
  // length query.
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncListxattr(std::string(path));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  if (rc != 0) {
    return -rc;  // ENOENT
  }
  std::string names = t->names_.str();
  size_t len = names.size();
  if (size == 0) {
    return static_cast<int>(len);  // length query
  }
  if (size < len) {
    return -ERANGE;
  }
  std::memcpy(list, names.data(), len);
  return static_cast<int>(len);
}

static int cte_fuse_removexattr(const char *path, const char *name) {
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncRemovexattr(std::string(path), std::string(name));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  return rc == 0 ? 0 : -rc;  // ENODATA/ENOENT/EIO -> negative errno
}

#ifndef RENAME_NOREPLACE
#define RENAME_NOREPLACE (1 << 0)  // from <linux/fs.h>; guarded to avoid header clash
#endif
#ifndef RENAME_EXCHANGE
#define RENAME_EXCHANGE (1 << 1)  // ditto; referenced by the unsupported-flag test
#endif

static int cte_fuse_rename(const char *from, const char *to,
                           unsigned int flags) {
  auto *cfs = CLIO_CFS_CLIENT;
  // RENAME_NOREPLACE: the rename must fail with EEXIST if `to` already exists.
  // Probe for the destination then fall through to a plain rename. This is the
  // standard high-level-FUSE approach (a tiny TOCTOU window vs a truly atomic
  // check, acceptable for a single-namespace rename). RENAME_EXCHANGE and
  // RENAME_WHITEOUT need chimod-level atomic swap / whiteout support and stay
  // EINVAL so callers fall back cleanly.
  if (flags & RENAME_NOREPLACE) {
    auto g = cfs->AsyncGetattr(std::string(to));
    g.Wait();
    if (g->GetReturnCode() == 0 && g->exists_ != 0) return -EEXIST;
    flags &= ~static_cast<unsigned int>(RENAME_NOREPLACE);
  }
  if (flags != 0) {
    return -EINVAL;  // RENAME_EXCHANGE / RENAME_WHITEOUT unsupported
  }
  // Drain BOTH names first (issue #933). The deferred registry is keyed by
  // PATH, so a write still queued against `from` when the rename lands would
  // be flushed to a name that no longer exists -- and no later Flush(to) could
  // find it, because its key is the old path. That is silent data loss, not a
  // stale read. Draining `to` as well covers the overwrite case, where the
  // destination has its own writes in flight that must not outlive it.
  FlushPathBuffers(std::string(from));
  FlushPathBuffers(std::string(to));
  (void)cfs->Flush(std::string(from));
  (void)cfs->Flush(std::string(to));
  auto t = cfs->AsyncRename(std::string(from), std::string(to));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  return rc == 0 ? 0 : -rc;  // chimod returns errno-style codes (ENOENT/EIO)
}

// ============================================================================
// Main
// ============================================================================

// Report filesystem statistics. Total and remaining capacity are the real
// cluster-wide values, obtained from the CTE: GetCapacity sums the registered
// targets' total and remaining capacity per node, and a Broadcast aggregates
// that across the cluster (the task's AggregateOut sums per-node results).
// Reporting a non-zero capacity also matters operationally: a 0-block fs is
// hidden by `df` (which lists no path), which breaks tools that probe free
// space and xfstests' mount detection.
static int cte_fuse_statfs(const char *path, cte_statvfs_t *stbuf) {
  (void)path;
  std::memset(stbuf, 0, sizeof(*stbuf));
  constexpr fsblkcnt_t kBlockSize = 4096;

  clio::run::u64 total_bytes = 0;
  clio::run::u64 remaining_bytes = 0;
  auto *cte = CLIO_CTE_CLIENT;
  if (cte != nullptr) {
    // Broadcast: total + remaining capacity across the whole cluster.
    auto t = cte->AsyncGetCapacity(clio::run::PoolQuery::Broadcast());
    t.Wait();
    if (t->return_code_ == 0) {
      total_bytes = t->total_capacity_;
      remaining_bytes = t->remaining_capacity_;
    }
  }
  fsblkcnt_t total_blocks = total_bytes / kBlockSize;
  fsblkcnt_t free_blocks = remaining_bytes / kBlockSize;

  stbuf->f_bsize = kBlockSize;
  stbuf->f_blocks = total_blocks;
  // Report real remaining space as both free and available (no reservation
  // distinction), so df shows used = total - remaining.
  stbuf->f_bfree = free_blocks;
  stbuf->f_bavail = free_blocks;
  stbuf->f_files = static_cast<fsfilcnt_t>(1) << 20;
  stbuf->f_ffree = static_cast<fsfilcnt_t>(1) << 20;
#ifdef __APPLE__
  // Darwin struct statfs has no f_frsize/f_favail/f_namemax; f_iosize is its
  // optimal-transfer-size analogue.
  stbuf->f_iosize = kBlockSize;
#else
  stbuf->f_frsize = kBlockSize;
  stbuf->f_favail = static_cast<fsfilcnt_t>(1) << 20;
  stbuf->f_namemax = 255;
#endif
  return 0;
}

// External linkage (declared in fuse_cte.h): the mount/process-entry glue in
// fuse_cte_main.cc points fuse_main()/fuse_new() at this table. The callbacks
// it references keep internal linkage — their addresses are captured here.
const struct fuse_operations cte_fuse_ops = {
    .getattr = cte_fuse_getattr,
    .readlink = cte_fuse_readlink,
    .mkdir = cte_fuse_mkdir,
    .unlink = cte_fuse_unlink,
    .rmdir = cte_fuse_rmdir,
    .symlink = cte_fuse_symlink,
    .rename = cte_fuse_rename,
    .link = cte_fuse_link,
    .chmod = cte_fuse_chmod,
    .chown = cte_fuse_chown,
    .truncate = cte_fuse_truncate,
    .open = cte_fuse_open,
    .read = cte_fuse_read,
    .write = cte_fuse_write,
    .statfs = cte_fuse_statfs,
    .flush = cte_fuse_flush,
    .release = cte_fuse_release,
    .fsync = cte_fuse_fsync,
#ifdef __APPLE__
    .setxattr = cte_fuse_setxattr_darwin,
    .getxattr = cte_fuse_getxattr_darwin,
#else
    .setxattr = cte_fuse_setxattr,
    .getxattr = cte_fuse_getxattr,
#endif
    .listxattr = cte_fuse_listxattr,
    .removexattr = cte_fuse_removexattr,
    .readdir = cte_fuse_readdir,
    .init = cte_fuse_init,
    .destroy = cte_fuse_destroy,
    .create = cte_fuse_create,
    .utimens = cte_fuse_utimens,
#ifdef __linux__
    .fallocate = cte_fuse_fallocate,
#endif
};
