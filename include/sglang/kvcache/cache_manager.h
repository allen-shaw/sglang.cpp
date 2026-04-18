#pragma once

#include <memory>
#include <vector>

#include "sglang/kvcache/base.h"
#include "sglang/kvcache/radix_tree.h"

namespace sglang {

// ============================================================
// RadixCacheHandle — concrete handle wrapping a RadixTreeNode*
// ============================================================
class RadixCacheHandle : public BaseCacheHandle {
 public:
    RadixCacheHandle(int64_t cached_len, RadixTreeNode* node);

    torch::Tensor get_matched_indices() const override;

    RadixTreeNode* node() const { return node_; }

 private:
    RadixTreeNode* node_;
};

// ============================================================
// RadixPrefixCache — prefix cache manager using Radix Tree
// ============================================================
class RadixPrefixCache : public BasePrefixCache {
 public:
    explicit RadixPrefixCache(int page_size = 1);
    ~RadixPrefixCache() override;

    // Disallow copy
    RadixPrefixCache(const RadixPrefixCache&) = delete;
    RadixPrefixCache& operator=(const RadixPrefixCache&) = delete;

    void lock_handle(const std::shared_ptr<BaseCacheHandle>& handle,
                     bool unlock = false) override;

    MatchResult match_prefix(const torch::Tensor& input_ids) override;

    InsertResult insert_prefix(const torch::Tensor& input_ids,
                               const torch::Tensor& indices) override;

    torch::Tensor evict(int64_t size) override;

    void reset() override;

    SizeInfo size_info() const override;

    void check_integrity() const override;

    // Expose root for testing
    RadixTreeNode* root_node() const { return root_node_; }

 private:
    /// Walk the tree matching input_ids, returning deepest matched node and
    /// prefix length.
    std::pair<RadixTreeNode*, int64_t> tree_walk(
        const torch::Tensor& input_ids);

    /// Collect leaf nodes eligible for eviction (ref_count == 0).
    std::vector<RadixTreeNode*> collect_leave_nodes_for_evict() const;

    /// Recursively delete all nodes in the tree (for destructor).
    void delete_tree(RadixTreeNode* node);

    int page_size_;
    KeyFn key_fn_;
    int64_t evictable_size_ = 0;
    int64_t protected_size_ = 0;
    RadixTreeNode* root_node_;

    /// Track all allocated nodes for cleanup.
    std::vector<std::unique_ptr<RadixTreeNode>> owned_nodes_;
};

}  // namespace sglang
