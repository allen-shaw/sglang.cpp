#include "sglang/kvcache/cache_manager.h"

#include <algorithm>
#include <queue>
#include <stack>

#include "sglang/utils/math_utils.h"

namespace sglang {

// ============================================================
// RadixCacheHandle
// ============================================================

RadixCacheHandle::RadixCacheHandle(int64_t cached_len, RadixTreeNode* node)
    : BaseCacheHandle(cached_len), node_(node) {}

torch::Tensor RadixCacheHandle::get_matched_indices() const {
    std::vector<torch::Tensor> value_list;
    auto* n = node_;
    while (!n->is_root()) {
        value_list.push_back(n->value());
        n = n->parent();
    }
    if (value_list.empty()) {
        return torch::empty({0}, torch::kInt32);
    }
    std::reverse(value_list.begin(), value_list.end());
    return torch::cat(value_list);
}

// ============================================================
// RadixPrefixCache
// ============================================================

RadixPrefixCache::RadixPrefixCache(int page_size)
    : page_size_(page_size), key_fn_(get_key_fn(page_size)) {
    auto root = std::make_unique<RadixTreeNode>(key_fn_);
    root_node_ = root.get();
    root_node_->ref_count = 1;  // root is always protected
    owned_nodes_.push_back(std::move(root));
}

RadixPrefixCache::~RadixPrefixCache() {
    // All nodes cleaned up via owned_nodes_ (unique_ptr).
}

void RadixPrefixCache::lock_handle(
    const std::shared_ptr<BaseCacheHandle>& handle, bool unlock) {
    auto* radix_handle = dynamic_cast<RadixCacheHandle*>(handle.get());
    TORCH_CHECK(radix_handle != nullptr,
                "lock_handle: handle must be RadixCacheHandle");
    auto* node = radix_handle->node();

    if (unlock) {
        while (!node->is_root()) {
            node->ref_count -= 1;
            TORCH_CHECK(node->ref_count >= 0,
                        "lock_handle: ref_count went negative");
            if (node->ref_count == 0) {
                evictable_size_ += node->length();
                protected_size_ -= node->length();
            }
            node = node->parent();
        }
    } else {
        while (!node->is_root()) {
            if (node->ref_count == 0) {
                evictable_size_ -= node->length();
                protected_size_ += node->length();
            }
            node->ref_count += 1;
            node = node->parent();
        }
    }
}

MatchResult RadixPrefixCache::match_prefix(const torch::Tensor& input_ids) {
    auto [node, prefix_len] = tree_walk(input_ids);
    auto handle =
        std::make_shared<RadixCacheHandle>(prefix_len, node);
    return MatchResult{handle};
}

InsertResult RadixPrefixCache::insert_prefix(const torch::Tensor& input_ids,
                                             const torch::Tensor& indices) {
    int64_t insert_len = align_down(input_ids.size(0), page_size_);
    auto trimmed_ids = input_ids.slice(0, 0, insert_len);
    auto trimmed_indices = indices.slice(0, 0, insert_len);

    auto [node, prefix_len] = tree_walk(trimmed_ids);

    if (prefix_len != insert_len) {
        // Create new leaf node for the unmatched suffix.
        auto new_node = std::make_unique<RadixTreeNode>(key_fn_);
        new_node->set_key_value(trimmed_ids.slice(0, prefix_len).clone(),
                                trimmed_indices.slice(0, prefix_len).clone());
        new_node->set_parent(node);
        evictable_size_ += new_node->length();
        node = new_node.get();
        owned_nodes_.push_back(std::move(new_node));
    }

    auto handle =
        std::make_shared<RadixCacheHandle>(insert_len, node);
    return InsertResult{prefix_len, handle};
}

torch::Tensor RadixPrefixCache::evict(int64_t size) {
    if (size == 0) {
        return torch::empty({0}, torch::kInt32);
    }
    TORCH_CHECK(size <= evictable_size_,
                "Cannot evict ", size, ", only ", evictable_size_,
                " is evictable");

    // Collect evictable leaves and build a min-heap by timestamp.
    auto leaves = collect_leave_nodes_for_evict();

    // Comparator: greater-than for min-heap (smallest timestamp first).
    auto cmp = [](RadixTreeNode* a, RadixTreeNode* b) {
        return a->timestamp > b->timestamp;
    };
    std::priority_queue<RadixTreeNode*, std::vector<RadixTreeNode*>,
                        decltype(cmp)>
        heap(cmp, std::move(leaves));

    std::vector<torch::Tensor> evicted_indices;
    int64_t evicted_size = 0;

    while (evicted_size < size) {
        TORCH_CHECK(!heap.empty(),
                    "Cannot evict enough cache, need ", size,
                    ", only ", evicted_size, " evicted");
        auto* node = heap.top();
        heap.pop();

        TORCH_CHECK(node->ref_count == 0 && node->is_leaf() &&
                         !node->is_root(),
                     "evict: invalid node state");

        evicted_size += node->length();
        evicted_indices.push_back(node->value());
        evictable_size_ -= node->length();

        auto* par = node->parent();
        par->children.erase(key_fn_(node->key()));

        // If parent became a leaf and is also evictable, push to heap.
        if (par->is_leaf() && par->ref_count == 0 && !par->is_root()) {
            heap.push(par);
        }
    }

    if (evicted_indices.empty()) {
        return torch::empty({0}, torch::kInt32);
    }
    return torch::cat(evicted_indices);
}

void RadixPrefixCache::reset() {
    // Not implemented — matching Python behavior.
    TORCH_CHECK(false, "RadixPrefixCache::reset is not implemented");
}

SizeInfo RadixPrefixCache::size_info() const {
    return SizeInfo{evictable_size_, protected_size_};
}

void RadixPrefixCache::check_integrity() const {
    // Placeholder — can be extended with tree validation.
}

std::pair<RadixTreeNode*, int64_t> RadixPrefixCache::tree_walk(
    const torch::Tensor& input_ids) {
    int64_t prefix_len = 0;
    int64_t total_len = input_ids.size(0);
    auto* node = root_node_;
    int64_t tic = std::chrono::steady_clock::now().time_since_epoch().count();

    while (prefix_len < total_len) {
        auto remaining = input_ids.slice(0, prefix_len);
        int64_t child_key = key_fn_(remaining);
        auto it = node->children.find(child_key);
        if (it == node->children.end()) {
            return {node, prefix_len};
        }
        node = it->second;

        // Match within this node.
        int64_t match_len =
            node->get_match_len(input_ids.slice(0, prefix_len));
        match_len = align_down(match_len, page_size_);
        prefix_len += match_len;

        // If the node is not fully matched, split it.
        if (match_len != node->length()) {
            // split_at allocates a new node internally
            auto* new_node = node->split_at(match_len);
            // We need to track the new_node. But split_at uses `new` internally.
            // We need to own it.
            owned_nodes_.emplace_back(new_node);
            return {new_node, prefix_len};
        }

        // Update timestamp for accessed node.
        node->timestamp = tic;
    }

    return {node, prefix_len};
}

std::vector<RadixTreeNode*> RadixPrefixCache::collect_leave_nodes_for_evict()
    const {
    std::vector<RadixTreeNode*> leaves;
    std::stack<RadixTreeNode*> stack;
    stack.push(root_node_);

    while (!stack.empty()) {
        auto* node = stack.top();
        stack.pop();

        if (node->is_leaf()) {
            if (node->ref_count == 0 && !node->is_root()) {
                leaves.push_back(node);
            }
        } else {
            for (auto& [_, child] : node->children) {
                stack.push(child);
            }
        }
    }
    return leaves;
}

void RadixPrefixCache::delete_tree(RadixTreeNode* node) {
    // All cleanup is handled by owned_nodes_ vector of unique_ptr.
}

}  // namespace sglang
