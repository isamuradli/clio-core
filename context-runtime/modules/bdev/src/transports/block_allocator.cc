/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * ...
 */

#include <clio_runtime/bdev/transports/block_allocator.h>

namespace clio::run::bdev {

// The implementation of WorkerBlockMap, GlobalBlockMap, Heap 
// will be moved here from bdev_runtime.cc

const size_t kBlockSizes[] = {
    4096,     // 4KB
    16384,    // 16KB
    32768,    // 32KB
    65536,    // 64KB
    131072,   // 128KB
    1048576   // 1MB
};

WorkerBlockMap::WorkerBlockMap() {
  blocks_.resize(static_cast<size_t>(BlockSizeCategory::kMaxCategories));
}

bool WorkerBlockMap::AllocateBlock(int block_type, Block &block, size_t min_size) {
  if (block_type < 0 ||
      block_type >= static_cast<int>(BlockSizeCategory::kMaxCategories)) {
    return false;
  }
  auto &list = blocks_[block_type];
  if (list.empty()) return false;

  for (auto it = list.begin(); it != list.end(); ++it) {
    if (it->size_ >= min_size) {
      block = *it;
      list.erase(it);
      return true;
    }
  }
  return false;
}

bool WorkerBlockMap::AllocateAnyUpTo(size_t max_bytes, Block &block) {
  // Largest category first so a big request drains few large freed extents
  // rather than many tiny ones. Match on the block's PHYSICAL FOOTPRINT (its
  // 4 KiB-aligned size, which is what it actually occupies), not its logical
  // size_ -- taking a block whose footprint exceeds the remainder would occupy
  // more space than is charged for it. Alignment is 4 KiB everywhere in this
  // allocator (the block-size categories and the heap slab are all 4 KiB
  // multiples), so it is safe to compute here without the allocator's field.
  for (int bt = static_cast<int>(BlockSizeCategory::kMaxCategories) - 1; bt >= 0;
       --bt) {
    auto &list = blocks_[bt];
    for (auto it = list.begin(); it != list.end(); ++it) {
      clio::run::u64 footprint = ((it->size_ + 4095) / 4096) * 4096;
      if (it->size_ > 0 && footprint <= max_bytes) {
        block = *it;
        list.erase(it);
        return true;
      }
    }
  }
  return false;
}

void WorkerBlockMap::FreeBlock(Block block) {
  int block_type = static_cast<int>(block.block_type_);
  if (block_type >= 0 &&
      block_type < static_cast<int>(BlockSizeCategory::kMaxCategories)) {
    blocks_[block_type].push_back(block);
  }
}

GlobalBlockMap::GlobalBlockMap() {}

void GlobalBlockMap::Init(size_t num_workers) {
  worker_maps_.resize(num_workers);
  worker_locks_.resize(num_workers);
}

int GlobalBlockMap::FindBlockType(size_t io_size) {
  for (int i = 0; i < static_cast<int>(BlockSizeCategory::kMaxCategories); ++i) {
    if (io_size <= kBlockSizes[i]) return i;
  }
  return -1;
}

bool GlobalBlockMap::AllocateBlock(int worker, size_t io_size, Block &block) {
  int block_type = FindBlockType(io_size);
  if (block_type == -1) {
    block_type = static_cast<int>(BlockSizeCategory::kMaxCategories) - 1;
  }

  size_t worker_idx = static_cast<size_t>(worker) % worker_maps_.size();

  {
    clio::run::ScopedCoMutex lock(worker_locks_[worker_idx]);
    if (worker_maps_[worker_idx].AllocateBlock(block_type, block, io_size)) {
      return true;
    }
  }

  for (size_t i = 1; i < worker_maps_.size(); ++i) {
    size_t other_worker = (worker_idx + i) % worker_maps_.size();
    clio::run::ScopedCoMutex lock(worker_locks_[other_worker]);
    if (worker_maps_[other_worker].AllocateBlock(block_type, block, io_size)) {
      return true;
    }
  }

  return false;
}

bool GlobalBlockMap::AllocateAnyUpTo(int worker, size_t max_bytes,
                                     Block &block) {
  if (worker_maps_.empty()) return false;
  size_t worker_idx = static_cast<size_t>(worker) % worker_maps_.size();
  {
    clio::run::ScopedCoMutex lock(worker_locks_[worker_idx]);
    if (worker_maps_[worker_idx].AllocateAnyUpTo(max_bytes, block)) {
      return true;
    }
  }
  for (size_t i = 1; i < worker_maps_.size(); ++i) {
    size_t other = (worker_idx + i) % worker_maps_.size();
    clio::run::ScopedCoMutex lock(worker_locks_[other]);
    if (worker_maps_[other].AllocateAnyUpTo(max_bytes, block)) {
      return true;
    }
  }
  return false;
}

bool GlobalBlockMap::FreeBlock(int worker, Block &block) {
  if (worker_maps_.empty()) return false;
  size_t worker_idx = static_cast<size_t>(worker) % worker_maps_.size();
  clio::run::ScopedCoMutex lock(worker_locks_[worker_idx]);
  worker_maps_[worker_idx].FreeBlock(block);
  return true;
}

Heap::Heap() : heap_(0), total_size_(0), alignment_(4096) {}

void Heap::Init(clio::run::u64 total_size, clio::run::u32 alignment) {
  heap_.store(0);
  total_size_ = total_size;
  alignment_ = alignment;
}

bool Heap::Allocate(size_t block_size, int block_type, Block &block) {
  clio::run::u32 alignment = (alignment_ == 0) ? 4096 : alignment_;
  clio::run::u64 aligned_size =
      ((block_size + alignment - 1) / alignment) * alignment;

  clio::run::u64 old_heap = heap_.fetch_add(aligned_size);
  if (total_size_ > 0 && old_heap + aligned_size > total_size_) {
    heap_.fetch_sub(aligned_size);
    return false;
  }

  block.offset_ = old_heap;
  block.size_ = static_cast<clio::run::u64>(block_size);
  block.block_type_ = static_cast<clio::run::u32>(block_type);
  return true;
}

clio::run::u64 Heap::GetRemainingSize() const {
  clio::run::u64 current_heap = heap_.load();
  if (total_size_ > current_heap) {
    return total_size_ - current_heap;
  }
  return 0;
}

bool StandardBlockAllocator::AllocateBlocks(size_t size, int worker_id, std::vector<Block>& blocks) {
  clio::run::u64 total_size = size;
  if (total_size == 0) {
    blocks.clear();
    return true;
  }

  Block block;
  if (global_block_map_.AllocateBlock(worker_id, total_size, block)) {
    blocks.push_back(block);
    allocated_bytes_.fetch_add(block.size_, std::memory_order_relaxed);
    return true;
  }

  // No single free extent fits. FRAGMENT the request across several smaller
  // free-list blocks before touching the heap (issue #820).
  //
  // The free list is bucketed by size category, so a request only ever probes
  // its own category: a 1 MiB ask never sees freed 64 KB blocks. Under churn
  // the free pool is mostly sub-request-size extents, so a large contiguous
  // ask can never be reused and the bump-only heap watermark climbs to the tier
  // capacity -> write EIO. This used to be worked around in the CTE
  // (ExtendBlob capped every block at 64 KB), but fragmentation avoidance is
  // the allocator's job, not the caller's: the blocks_ vector already carries
  // multi-block extents transparently, so a large logical allocation backed by
  // several reused physical blocks is invisible above this layer.
  //
  // Pull freed blocks that FIT the remainder, of whatever size the free list
  // actually holds -- a freed 8 KiB block bucketed under 16 KiB, which the
  // single-block >= match above can never reach.
  //
  // Accounting is in PHYSICAL bytes (AlignSize(size_) -- the footprint the
  // block actually occupies). Each reused block is reported at its full
  // footprint and covers that many bytes of the request, so a big ask consumes
  // one slab per slab of need -- NOT one slab per (possibly sub-slab) freed
  // logical size, which would over-consume space and exhaust the tier. The
  // caller distributes the logical request across these physical blocks.
  clio::run::u64 remaining_phys = AlignSize(total_size);
  const int kCats = static_cast<int>(BlockSizeCategory::kMaxCategories);
  Block fb;
  while (remaining_phys > 0 &&
         global_block_map_.AllocateAnyUpTo(worker_id, remaining_phys, fb)) {
    clio::run::u64 fp = AlignSize(fb.size_);        // true footprint
    if (fp > remaining_phys) fp = remaining_phys;   // last partial block
    fb.size_ = fp;                                  // report the footprint
    blocks.push_back(fb);
    allocated_bytes_.fetch_add(fp, std::memory_order_relaxed);
    remaining_phys -= fp;
  }

  if (remaining_phys == 0) {
    return true;  // fully satisfied from reused space -- no heap growth
  }
  if (!blocks.empty()) {
    // Partially reused; cover the tail from the heap. Roll back everything on
    // failure so the caller sees all-or-nothing (it frees `blocks` only on a
    // true return).
    int tail_type = GlobalBlockMap::FindBlockType(remaining_phys);
    if (tail_type == -1) tail_type = kCats - 1;
    Block tail;
    if (heap_.Allocate(remaining_phys, tail_type, tail)) {
      tail.size_ = remaining_phys;
      blocks.push_back(tail);
      allocated_bytes_.fetch_add(remaining_phys, std::memory_order_relaxed);
      return true;
    }
    for (auto &b : blocks) {
      global_block_map_.FreeBlock(worker_id, b);  // return to the free list
      allocated_bytes_.fetch_sub(AlignSize(b.size_), std::memory_order_relaxed);
    }
    blocks.clear();
    return false;
  }

  clio::run::u64 aligned_total_size = AlignSize(total_size);
  // Classify the fresh allocation by its size category so callers (and the
  // free list it returns to) see the correct block_type_. Previously this
  // hardcoded the largest category, so e.g. a 4KB allocation came back tagged
  // as 1MB (block_type_ != 0).
  int block_type = GlobalBlockMap::FindBlockType(aligned_total_size);
  if (block_type == -1) {
    block_type = static_cast<int>(BlockSizeCategory::kMaxCategories) - 1;
  }
  if (heap_.Allocate(aligned_total_size, block_type, block)) {
    // block.size_ stays the caller's requested size — that is what the
    // caller reads and writes, and what it hands back to FreeBlocks.
    block.size_ = total_size;
    blocks.push_back(block);
    // Charge the ALIGNED size, not the raw request. The heap advanced by
    // aligned_total_size, and FreeBlocks credits AlignSize(block.size_),
    // which is the same aligned value. Charging total_size here made every
    // first allocation of a non-4096-multiple block credit more on free
    // than it charged, so allocated_bytes_ drifted downward by
    // (AlignSize(size) - size) per block and GetRemainingSize() reported
    // progressively more free space than the target actually had. The
    // `std::min` clamp in FreeBlocks hid it by silently absorbing the
    // difference rather than underflowing. See #798.
    allocated_bytes_.fetch_add(aligned_total_size, std::memory_order_relaxed);
    return true;
  }

  return false;
}

void StandardBlockAllocator::FreeBlocks(int worker_id, const std::vector<Block>& blocks) {
  clio::run::u64 freed_bytes = 0;
  for (const auto& block : blocks) {
    Block block_copy = block;
    clio::run::u64 aligned_size = AlignSize(block.size_);
    block_copy.size_ = aligned_size;
    int bt = -1;
    for (int i = 0; i < static_cast<int>(BlockSizeCategory::kMaxCategories); ++i) {
      if (aligned_size <= kBlockSizes[i]) {
        bt = i;
        break;
      }
    }
    if (bt == -1) bt = static_cast<int>(BlockSizeCategory::kMaxCategories) - 1;
    block_copy.block_type_ = static_cast<clio::run::u32>(bt);
    freed_bytes += block_copy.size_;
    global_block_map_.FreeBlock(worker_id, block_copy);
  }

  clio::run::u64 cur = allocated_bytes_.load(std::memory_order_relaxed);
  clio::run::u64 dec = std::min(cur, freed_bytes);
  allocated_bytes_.fetch_sub(dec, std::memory_order_relaxed);
}

clio::run::u64 StandardBlockAllocator::GetRemainingSize() const {
  clio::run::u64 live = allocated_bytes_.load(std::memory_order_relaxed);
  return (capacity_ > live) ? (capacity_ - live) : 0;
}

} // namespace clio::run::bdev
