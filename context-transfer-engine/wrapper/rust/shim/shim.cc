#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/content_transfer_engine.h>

#include <cstring>
#include <stdexcept>

// cxx-generated header: defines CteTagId shared struct
#include "wrp-cte-rs/src/lib.rs.h"

namespace cte_ffi {

namespace {

namespace cte = clio::cte::core;

// Client-side staging buffer in shared memory. The Rust API deals in plain byte
// slices while the CTE client takes ShmPtr, so every blob op stages through
// SHM; the destructor releases the buffer even when an op throws.
class ShmStage {
 public:
  explicit ShmStage(size_t size) : buf_(CLIO_IPC->AllocateBuffer(size)) {
    if (buf_.IsNull()) {
      throw std::runtime_error("Failed to allocate shared memory for blob I/O");
    }
  }
  ~ShmStage() { CLIO_IPC->FreeBuffer(buf_); }
  ShmStage(const ShmStage &) = delete;
  ShmStage &operator=(const ShmStage &) = delete;

  char *data() const { return buf_.ptr_; }
  ctp::ipc::ShmPtr<> shm() const { return ctp::ipc::ShmPtr<>(buf_.shm_); }

 private:
  ctp::ipc::FullPtr<char> buf_;
};

}  // namespace

bool cte_init(rust::Str config_path) {
  std::string path(config_path.data(), config_path.size());
  bool ok = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
  if (!ok) return false;
  return clio::cte::core::CLIO_CTE_CLIENT_INIT(path);
}

CteTag::CteTag(const std::string &name) {
  auto task = CLIO_CTE_CLIENT->AsyncGetOrCreateTag(name);
  task.Wait();
  if (task->GetReturnCode() != 0) {
    throw std::runtime_error("GetOrCreateTag operation failed");
  }
  id = task->tag_id_;
}

std::unique_ptr<CteTag> tag_new(rust::Str tag_name) {
  std::string name(tag_name.data(), tag_name.size());
  return std::make_unique<CteTag>(name);
}

std::unique_ptr<CteTag> tag_from_id(uint32_t major, uint32_t minor) {
  clio::cte::core::TagId tid(major, minor);
  return std::make_unique<CteTag>(tid);
}

void tag_put_blob(const CteTag &tag, rust::Str name,
                  rust::Slice<const uint8_t> data, uint64_t offset,
                  float score) {
  std::string blob_name(name.data(), name.size());
  ShmStage stage(data.size());
  std::memcpy(stage.data(), data.data(), data.size());

  auto task = CLIO_CTE_CLIENT->AsyncPutBlob(
      tag.id, blob_name, static_cast<clio::run::u64>(offset),
      static_cast<clio::run::u64>(data.size()), stage.shm(), score,
      cte::Context(), 0, clio::run::PoolQuery::Dynamic());
  task.Wait();
  if (task->GetReturnCode() != 0) {
    throw std::runtime_error(
        std::string("PutBlob operation failed (rc=") +
        std::to_string(task->GetReturnCode()) + ")");
  }
}

std::unique_ptr<std::vector<uint8_t>> tag_get_blob(const CteTag &tag,
                                                    rust::Str name,
                                                    uint64_t size,
                                                    uint64_t offset) {
  if (size == 0) {
    throw std::invalid_argument("size must be specified for GetBlob");
  }
  std::string blob_name(name.data(), name.size());
  ShmStage stage(size);

  auto task = CLIO_CTE_CLIENT->AsyncGetBlob(
      tag.id, blob_name, static_cast<clio::run::u64>(offset),
      static_cast<clio::run::u64>(size), 0u, stage.shm());
  task.Wait();
  if (task->GetReturnCode() != 0) {
    throw std::runtime_error("GetBlob operation failed");
  }

  auto buf = std::make_unique<std::vector<uint8_t>>(size);
  std::memcpy(buf->data(), stage.data(), size);
  return buf;
}

float tag_get_blob_score(const CteTag &tag, rust::Str name) {
  std::string blob_name(name.data(), name.size());
  auto task = CLIO_CTE_CLIENT->AsyncGetBlobScore(tag.id, blob_name);
  task.Wait();
  return task->score_;
}

uint64_t tag_get_blob_size(const CteTag &tag, rust::Str name) {
  std::string blob_name(name.data(), name.size());
  auto task = CLIO_CTE_CLIENT->AsyncGetBlobSize(tag.id, blob_name);
  task.Wait();
  return task->size_;
}

std::unique_ptr<std::vector<std::string>> tag_get_contained_blobs(
    const CteTag &tag) {
  auto task = CLIO_CTE_CLIENT->AsyncGetContainedBlobs(tag.id);
  task.Wait();
  return std::make_unique<std::vector<std::string>>(
      std::move(task->blob_names_));
}

void tag_reorganize_blob(const CteTag &tag, rust::Str name, float score) {
  std::string blob_name(name.data(), name.size());
  auto task = CLIO_CTE_CLIENT->AsyncReorganizeBlob(tag.id, blob_name, score);
  task.Wait();
  if (task->GetReturnCode() != 0) {
    throw std::runtime_error("ReorganizeBlob operation failed");
  }
}

CteTagId tag_get_id(const CteTag &tag) {
  return CteTagId{tag.id.major_, tag.id.minor_};
}

bool client_register_target(rust::Str target_path, uint64_t size) {
  std::string path(target_path.data(), target_path.size());
  // Create a bdev pool for this target
  clio::run::PoolId bdev_pool_id(800, 0);
  clio::run::bdev::Client bdev_client(bdev_pool_id);
  auto create_task = bdev_client.AsyncCreate(
      clio::run::PoolQuery::Dynamic(), path, bdev_pool_id,
      clio::run::bdev::BdevType::kFile);
  create_task.Wait();
  // Register with CTE
  auto *client = CLIO_CTE_CLIENT;
  auto reg_task = client->AsyncRegisterTarget(
      path, clio::run::bdev::BdevType::kFile, size,
      clio::run::PoolQuery::Local(), bdev_pool_id);
  reg_task.Wait();
  return true;
}

bool client_del_tag(rust::Str name) {
  std::string tag_name(name.data(), name.size());
  auto *client = CLIO_CTE_CLIENT;
  auto task = client->AsyncDelTag(tag_name);
  task.Wait();
  return true;
}

std::unique_ptr<std::vector<std::string>> client_tag_query(rust::Str regex,
                                                            uint32_t max_tags) {
  std::string re(regex.data(), regex.size());
  auto *mgr = CTE_MANAGER;
  auto results = mgr->TagQuery(re, max_tags);
  return std::make_unique<std::vector<std::string>>(std::move(results));
}

std::unique_ptr<std::vector<std::string>> client_blob_query(rust::Str tag_re,
                                                             rust::Str blob_re,
                                                             uint32_t max_results) {
  std::string tre(tag_re.data(), tag_re.size());
  std::string bre(blob_re.data(), blob_re.size());
  auto *mgr = CTE_MANAGER;
  auto pairs = mgr->BlobQuery(tre, bre, max_results);
  auto out = std::make_unique<std::vector<std::string>>();
  out->reserve(pairs.size() * 2);
  for (auto &p : pairs) {
    out->push_back(std::move(p.first));
    out->push_back(std::move(p.second));
  }
  return out;
}

}  // namespace cte_ffi
