#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <unordered_map>

#include <torch/torch.h>

namespace sglang {

/// Key function type: given a tensor slice, returns a hash key for children
/// lookup.  For page_size == 1, this returns the first token value directly.
using KeyFn = std::function<int64_t(const torch::Tensor&)>;

/// Get a default key function for the given page size.
/// page_size == 1 => first element value.
/// page_size > 1  => simple hash of the first `page_size` elements.
KeyFn get_key_fn(int page_size);

/// A node in the Radix Tree used for prefix caching.
/// Ported from minisgl/kvcache/radix_cache.py::RadixTreeNode.
class RadixTreeNode {
 public:
    /// Global counter for assigning unique IDs.
    static int64_t counter;

    explicit RadixTreeNode(KeyFn key_fn, int64_t timestamp = 0);

    /// Set the key (token ids) and value (cache indices) for this node.
    void set_key_value(torch::Tensor key, torch::Tensor value);

    /// Set the parent of this node and register as child.
    void set_parent(RadixTreeNode* parent);

    // --- Accessors ---
    int64_t length() const { return length_; }
    bool is_root() const { return parent_ == nullptr; }
    bool is_leaf() const { return children.empty(); }
    RadixTreeNode* parent() const { return parent_; }
    const torch::Tensor& key() const { return key_; }
    const torch::Tensor& value() const { return value_; }

    /// Compare key with input_ids and return the match length.
    int64_t get_match_len(const torch::Tensor& input_ids) const;

    /// Split this node at position `pos`, creating a new parent node.
    /// Returns the newly-created intermediate node.
    RadixTreeNode* split_at(int64_t pos);

    /// Comparison for min-heap (eviction order by timestamp).
    bool operator<(const RadixTreeNode& other) const {
        return timestamp < other.timestamp;
    }

    // --- Public fields ---
    std::unordered_map<int64_t, RadixTreeNode*> children;
    int32_t ref_count = 0;
    int64_t uuid;
    int64_t timestamp;

 private:
    KeyFn key_fn_;
    RadixTreeNode* parent_ = nullptr;
    torch::Tensor key_;
    torch::Tensor value_;
    int64_t length_ = 0;
};

}  // namespace sglang
