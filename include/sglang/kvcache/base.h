#pragma once

#include <memory>

#include <torch/torch.h>

namespace sglang {

// ============================================================
// Size info for prefix cache
// ============================================================
struct SizeInfo {
    int64_t evictable_size = 0;
    int64_t protected_size = 0;

    int64_t total_size() const { return evictable_size + protected_size; }
};

// ============================================================
// Abstract base class for cache handles
// ============================================================
struct BaseCacheHandle {
    int64_t cached_len = 0;

    explicit BaseCacheHandle(int64_t cached_len = 0) : cached_len(cached_len) {}
    virtual ~BaseCacheHandle() = default;

    virtual torch::Tensor get_matched_indices() const = 0;
};

// ============================================================
// Result types
// ============================================================
struct MatchResult {
    std::shared_ptr<BaseCacheHandle> handle;
};

struct InsertResult {
    int64_t cached_len;  // length already in cache before insertion
    std::shared_ptr<BaseCacheHandle> handle;
};

// ============================================================
// Abstract base class for prefix caches (e.g. RadixPrefixCache)
// ============================================================
class BasePrefixCache {
 public:
    virtual ~BasePrefixCache() = default;

    /// Lock or unlock a cache handle.
    /// When locked, the cached prefix cannot be evicted.
    virtual void lock_handle(const std::shared_ptr<BaseCacheHandle>& handle,
                             bool unlock) = 0;

    /// Match the longest prefix of input_ids in the cache.
    virtual MatchResult match_prefix(const torch::Tensor& input_ids) = 0;

    /// Insert a new prefix into the cache.
    virtual InsertResult insert_prefix(const torch::Tensor& input_ids,
                                       const torch::Tensor& indices) = 0;

    /// Evict cached prefixes to free at least `size` slots.
    virtual torch::Tensor evict(int64_t size) = 0;

    /// Reset the cache.
    virtual void reset() = 0;

    /// Get size information.
    virtual SizeInfo size_info() const = 0;

    /// Check cache integrity (raise on corruption).
    virtual void check_integrity() const = 0;
};

// ============================================================
// Abstract base class for KV cache pools
// ============================================================
class BaseKVCachePool {
 public:
    virtual ~BaseKVCachePool() = default;

    virtual torch::Tensor k_cache(int layer_id) const = 0;
    virtual torch::Tensor v_cache(int layer_id) const = 0;
    virtual void store_kv(const torch::Tensor& key, const torch::Tensor& val,
                          const torch::Tensor& out_loc, int layer_id) = 0;

    virtual torch::Device device() const = 0;
    virtual torch::Dtype dtype() const = 0;
    virtual int num_layers() const = 0;
};

}  // namespace sglang
