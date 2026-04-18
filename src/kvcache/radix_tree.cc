#include "sglang/kvcache/radix_tree.h"

#include "sglang/kernels/radix.h"

namespace sglang {

// ============================================================
// Static counter
// ============================================================
int64_t RadixTreeNode::counter = 0;

// ============================================================
// Key functions
// ============================================================

KeyFn get_key_fn(int page_size) {
    if (page_size == 1) {
        return [](const torch::Tensor& t) -> int64_t {
            return t[0].item<int64_t>();
        };
    }
    // For page_size > 1, hash the first `page_size` elements.
    return [page_size](const torch::Tensor& t) -> int64_t {
        int64_t h = 0;
        const int64_t len = std::min(static_cast<int64_t>(page_size), t.size(0));
        auto acc = t.accessor<int32_t, 1>();
        for (int64_t i = 0; i < len; ++i) {
            // Simple hash combine (boost-style).
            h ^= static_cast<int64_t>(acc[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    };
}

// ============================================================
// RadixTreeNode
// ============================================================

RadixTreeNode::RadixTreeNode(KeyFn key_fn, int64_t timestamp)
    : key_fn_(std::move(key_fn)),
      uuid(RadixTreeNode::counter++),
      timestamp(timestamp == 0
                    ? std::chrono::steady_clock::now().time_since_epoch().count()
                    : timestamp) {}

void RadixTreeNode::set_key_value(torch::Tensor key, torch::Tensor value) {
    TORCH_CHECK(key.size(0) == value.size(0),
                "RadixTreeNode::set_key_value: key and value must have the "
                "same length");
    key_ = std::move(key);
    value_ = std::move(value);
    length_ = key_.size(0);
}

void RadixTreeNode::set_parent(RadixTreeNode* parent) {
    TORCH_CHECK(parent != nullptr, "set_parent: parent must not be null");
    parent_ = parent;
    parent->children[key_fn_(key_)] = this;
}

int64_t RadixTreeNode::get_match_len(const torch::Tensor& input_ids) const {
    return fast_compare_key(key_, input_ids);
}

RadixTreeNode* RadixTreeNode::split_at(int64_t pos) {
    TORCH_CHECK(pos > 0 && pos < length_,
                "split_at: pos must be in range (0, length)");
    RadixTreeNode* old_parent = parent_;
    TORCH_CHECK(old_parent != nullptr, "split_at: cannot split root node");

    // Create new intermediate node with the prefix [0, pos)
    auto* new_node = new RadixTreeNode(key_fn_, timestamp);
    new_node->set_key_value(key_.slice(0, 0, pos), value_.slice(0, 0, pos));
    new_node->set_parent(old_parent);
    new_node->ref_count = ref_count;

    // Update this node to hold the suffix [pos, end)
    set_key_value(key_.slice(0, pos), value_.slice(0, pos));
    set_parent(new_node);

    return new_node;
}

}  // namespace sglang
