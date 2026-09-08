/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file device_chunk.h
 * @brief Submit a DEVICE-RESIDENT chunk, as the in-situ adapters do -- NeuroPress
 *        preprocessing is CUDA-only and refuses host memory outright.
 */

#ifndef CLIO_CTE_COMPRESSOR_TEST_DEVICE_CHUNK_H_
#define CLIO_CTE_COMPRESSOR_TEST_DEVICE_CHUNK_H_

#include <clio_runtime/ipc_manager.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace clio::cte::compressor::test {

/** One device-resident chunk plus the backend registration that names it. */
class DeviceChunk {
 public:
  DeviceChunk() { alloc_id_.SetNull(); }
  ~DeviceChunk() { Reset(); }
  DeviceChunk(const DeviceChunk &) = delete;
  DeviceChunk &operator=(const DeviceChunk &) = delete;

  /** Register a device backend and copy `bytes` from host `src` into it. */
  bool Fill(const void *src, size_t bytes) {
    Reset();
    char *base = nullptr;
    alloc_id_ = CLIO_IPC->AllocateAndRegisterGpuBackend(
        /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem, bytes,
        &base);
    if (alloc_id_.IsNull() || base == nullptr) {
      alloc_id_.SetNull();
      return false;
    }
    if (cudaMemcpy(base, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
      Reset();
      return false;
    }
    // Same convention Compress uses: a GPU backend carries its device pointer
    // in off_, and ToFullPtr resolves it through the registered backend.
    shm_.alloc_id_ = alloc_id_;
    shm_.off_ = reinterpret_cast<clio::run::u64>(base);
    return true;
  }

  /** The pointer to submit. Valid only while this object lives. */
  ctp::ipc::ShmPtr<> shm() const { return shm_; }

  void Reset() {
    // SetNull(): the default ctor gives (0,0) = GetRoot(), a VALID id.
    if (!alloc_id_.IsNull()) {
      CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, alloc_id_);
    }
    alloc_id_.SetNull();
    shm_.SetNull();
  }

 private:
  ctp::ipc::AllocatorId alloc_id_;
  ctp::ipc::ShmPtr<> shm_;
};

}  // namespace clio::cte::compressor::test

#endif  // CLIO_CTE_COMPRESSOR_TEST_DEVICE_CHUNK_H_
