/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_RUNTIME_INCLUDE_IPC_CPU2CPU_IMPL_H_
#define CLIO_RUNTIME_INCLUDE_IPC_CPU2CPU_IMPL_H_

#include "clio_runtime/ipc/ipc_cpu2cpu.h"

#include <unordered_map>

#include "clio_ctp/thread/thread_model_manager.h"
#include "clio_runtime/config_manager.h"

namespace clio::run {

// NOTE (SHM refactor): the thread-local ShmOutResponseStash that used to demux
// sibling responses off a per-thread ring is gone. Responses now arrive on the
// single per-process ring and are demuxed centrally by RecvShmClientThread,
// which parks each archive in IpcManager::pending_response_archives_ keyed by
// net_key and wakes the owning waiter — exactly like the ZMQ recv thread. Each
// RecvOut simply blocks on its EventManager and claims its own archive.

template <typename TaskT>
Future<TaskT> IpcCpu2Cpu::SendIn(IpcManager *ipc,
                                      const clio::run::shared_ptr<TaskT> &task_ptr) {
#if !CTP_IS_HOST
  // Host-only SHM client path (MPSC server, SystemInfo, mutex). Never invoked
  // from device kernels; provide an inert device definition so the template
  // compiles in the GPU device pass.
  (void)ipc;
  (void)task_ptr;
  return Future<TaskT>();
#else
  if (task_ptr.IsNull()) return Future<TaskT>();

  // #642: the task's virtual address is the response key the worker echoes back
  // so this client thread can match the result to the right Future.
  size_t net_key = reinterpret_cast<size_t>(task_ptr.get());
  task_ptr->task_id_.net_key_ = net_key;

  // The worker routes the result to "clio-<task_id_.pid_>-<task_id_.tid_>", which
  // MUST equal this client thread's MPSC server name. IpcManagerTls names that
  // server with ctp::SystemInfo::GetPid()/GetTid() (the OS tid), but CreateTaskId
  // stamps task_id_.tid_ from the thread model's *logical* id (PthreadModel hands
  // out a TLS counter, not the OS tid) — so without this the response is addressed
  // to a non-existent segment and the client hangs forever. Stamp the routing
  // identity from the same SystemInfo source the server is named with.
  task_ptr->task_id_.pid_ = static_cast<u32>(ctp::SystemInfo::GetPid());
  task_ptr->task_id_.tid_ = static_cast<u32>(ctp::SystemInfo::GetTid());

  // FutureShm now lives in PRIVATE memory owned by the Future's shared_ptr: the
  // worker never touches it; the result returns over this client thread's MPSC
  // server (clio-<pid>-<tid>).
  Future<TaskT> future(task_ptr->pool_id_, task_ptr->method_, task_ptr);
  RunContext *future_shm = future.GetFutureShm().ptr_;
  future_shm->origin_ = ClientOrigin::kClientShm;
  // Ensure this thread's MPSC receive server exists before the response lands.
  ipc->GetTls();
  // The waiter (this client thread) lives on the task's FutureInfo; the response
  // routes by task_id_.net_key_ (set above).
  task_ptr->SetWaiter(static_cast<u32>(ctp::SystemInfo::GetPid()),
                      static_cast<u32>(ctp::SystemInfo::GetTid()));

  // Register for response matching. The raw pointer stays valid as long as the
  // returned Future (or a copy) is alive — the client holds it until Recv.
  {
    std::lock_guard<std::mutex> lock(ipc->pending_futures_mutex_);
    ipc->pending_zmq_futures_[net_key] = {task_ptr.get()};
  }

  // Send the task to the runtime's single inbound ring. Clients no longer
  // load-balance across per-worker rings (worker_tids_); the net_recv worker
  // drains this one ring and fans tasks out to worker lanes, mirroring the ZMQ
  // ROUTER model. The runtime pid (learned via ClientConnect) keys the name.
  ctp::lbm::ShmMpscTransport *conn = ipc->GetOrCreateShmConn(
      "clio-" + std::to_string(ipc->runtime_pid_) + "-shm-in");
  if (conn == nullptr) {
    HLOG(kError, "IpcCpu2Cpu::SendIn: inbound SHM ring unavailable");
    task_ptr->SetComplete();  // unblock the waiter on the error path
    return future;
  }
  // shm_send_transport_ is only used to Expose bulk descriptors while building
  // the archive; conn->Send performs the actual MPSC transfer (metadata+data).
  SaveTaskArchive archive(MsgType::kSerializeIn,
                           ipc->shm_send_transport_.get());
  archive << (*task_ptr);
  int send_rc = conn->Send(archive);
  if (send_rc != 0) {
    // A submit that never reached the daemon must FAIL the future, not hang
    // it (issue #774: every silent drop on this path turns into a client
    // parked forever in Future::Wait).
    HLOG(kError, "IpcCpu2Cpu::SendIn: MPSC send failed rc={} for task {}",
         send_rc, task_ptr->task_id_);
    task_ptr->SetReturnCode(static_cast<clio::run::u32>(-send_rc));
    task_ptr->SetComplete();
  }
  return future;
#endif  // CTP_IS_HOST
}

template <typename TaskT>
bool IpcCpu2Cpu::RecvOut(IpcManager *ipc,
                             Future<TaskT> &future, float max_sec) {
#if !CTP_IS_HOST
  (void)ipc;
  (void)future;
  (void)max_sec;
  return false;
#else
  TaskT *task_ptr = future.get();
  const size_t want_key = task_ptr->task_id_.net_key_;

  // Wait for RecvShmClientThread to mark this task complete and signal us (SHM
  // analogue of IpcCpu2CpuZmq::RecvOut). The dedicated recv thread — not this
  // thread — drains the single response ring, so app threads no longer poll a
  // per-thread ring or demux siblings here.
  ctp::lbm::EventManager *em = &ipc->GetTls()->event_manager_;
  ctp::Timepoint start;
  start.Now();
  const double timeout_us =
      (max_sec > 0) ? static_cast<double>(max_sec) * 1e6 : 0;

  // Phase 1 — spin (issue #784).
  //
  // This is the last hop on the request path that parked unconditionally. The
  // runtime's workers busy-wait first_busy_wait (1ms) before sleeping, and both
  // SHM ring drainers spin kShmSpinBudget iterations before parking, but the
  // waiter itself went straight into epoll. So every single round trip paid a
  // tgkill -> signalfd -> epoll wake plus a scheduler round trip on the
  // response side — tens of microseconds on a path whose actual work is a few
  // microseconds, which is most of the ~5-6us -> 50-70us regression the
  // single-ring refactor (#780) introduced.
  //
  // Spinning here is safe against a lost wakeup: SIGUSR1 is blocked on this
  // thread (the EventManager ctor's AddSignalEvent), so a Signal that lands
  // while we spin stays pending on the signalfd and would make a later Wait()
  // return immediately. The loop condition is IsComplete() either way.
  //
  // The spin escalates rather than burning a core for the whole window:
  //
  //   0 .. kPauseWindowUs   CTP_THREAD_MODEL->Yield(), a CPU pause hint (~20ns)
  //                         that does NOT release the core. See the comment on
  //                         ThreadModel::Yield in thread_model/pthread.h, where
  //                         a 100ns nanosleep on this path was measured to
  //                         actually sleep ~54us and inflate single-RTT SHM
  //                         latency from ~1us to ~58us. Anything that can block
  //                         belongs nowhere near this window.
  //   .. spin_us            sched_yield, which releases the core to any other
  //                         runnable thread but leaves us in the run queue.
  //
  // The escalation matters because the spinner is not alone: on a SHM round
  // trip the runtime's workers are also busy-waiting (first_busy_wait) and both
  // ring drainers are spinning their budget. On an oversubscribed host those
  // plus N spinning clients exceed the core count, and a pure pause loop
  // starves the very workers that must produce our response — the same
  // livelock Worker::SuspendMe documents for the CI runners. kPauseWindowUs
  // sits comfortably above the measured SHM round trip (~28us), so the common
  // case completes in the pause phase and never reaches sched_yield.
  //
  // The clock is only read every kSpinClockMask+1 iterations so the timing
  // check does not dominate the pause phase.
  auto *config = CLIO_CONFIG_MANAGER;
  const double spin_us = static_cast<double>(config->GetClientBusyWait());
  if (spin_us > 0 && !task_ptr->IsComplete()) {
    constexpr size_t kSpinClockMask = 63;
    constexpr double kPauseWindowUs = 50.0;
    ctp::Timepoint now;
    bool pause_phase = true;
    for (size_t i = 0; !task_ptr->IsComplete(); ++i) {
      if (!pause_phase || (i & kSpinClockMask) == kSpinClockMask) {
        now.Now();
        double elapsed = start.GetUsecFromStart(now);
        if (elapsed >= spin_us) break;
        if (timeout_us > 0 && elapsed >= timeout_us) return false;
        pause_phase = elapsed < kPauseWindowUs;
      }
      if (pause_phase) {
        CTP_THREAD_MODEL->Yield();      // pause hint, keeps the core
      } else {
        ctp::SystemInfo::YieldThread();  // sched_yield, releases the core
      }
    }
  }

  // Phase 2 — park. The 100us bounded Wait is a missed-signal / timeout safety
  // re-check; the named auto-reset event latches a Signal that races the Wait.
  while (!task_ptr->IsComplete()) {
    em->Wait(100);
    if (timeout_us > 0) {
      ctp::Timepoint now;
      now.Now();
      if (start.GetUsecFromStart(now) >= timeout_us) {
        return false;
      }
    }
  }

  // Claim our parked response archive and deserialize into the task. Moving it
  // out and letting it destruct here matches the old stack-archive freeing:
  // output buffers are adopted into the task (TASK_DATA_OWNER) and the archive
  // frees only the wire bytes. Erasing the entry means Future::~Future's
  // CleanupResponseArchive is a no-op on this (consumed) path.
  std::atomic_thread_fence(std::memory_order_acquire);
  std::unique_ptr<LoadTaskArchive> archive;
  {
    std::lock_guard<std::mutex> lock(ipc->pending_futures_mutex_);
    auto it = ipc->pending_response_archives_.find(want_key);
    if (it != ipc->pending_response_archives_.end()) {
      archive = std::move(it->second);
      ipc->pending_response_archives_.erase(it);
    }
  }
  if (archive) {
    archive->ResetBulkIndex();
    archive->msg_type_ = MsgType::kSerializeOut;
    *archive >> (*task_ptr);
  }
  return true;
#endif  // CTP_IS_HOST
}

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_IPC_CPU2CPU_IMPL_H_
