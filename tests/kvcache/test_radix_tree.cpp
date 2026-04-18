#include <gtest/gtest.h>
#include "sglang/kvcache/radix_tree.h"

namespace sglang {
namespace {

TEST(RadixTreeNodeTest, BasicProperties) {
    auto key_fn = get_key_fn(1);
    RadixTreeNode node(key_fn);

    EXPECT_TRUE(node.is_root());
    EXPECT_TRUE(node.is_leaf());
    EXPECT_EQ(node.ref_count, 0);
    EXPECT_GE(node.uuid, 0);
    EXPECT_GT(node.timestamp, 0);
}

TEST(RadixTreeNodeTest, SetKeyValue) {
    auto key_fn = get_key_fn(1);
    RadixTreeNode node(key_fn);

    auto key = torch::tensor({1, 2, 3}, torch::kInt32);
    auto value = torch::tensor({10, 20, 30}, torch::kInt32);
    node.set_key_value(key, value);

    EXPECT_EQ(node.length(), 3);
    EXPECT_TRUE(torch::equal(node.key(), key));
    EXPECT_TRUE(torch::equal(node.value(), value));
}

TEST(RadixTreeNodeTest, SetParentChildRelation) {
    auto key_fn = get_key_fn(1);
    auto root = std::make_unique<RadixTreeNode>(key_fn);
    root->ref_count = 1;

    auto child = std::make_unique<RadixTreeNode>(key_fn);
    child->set_key_value(torch::tensor({5, 6}, torch::kInt32),
                         torch::tensor({50, 60}, torch::kInt32));
    child->set_parent(root.get());

    EXPECT_FALSE(root->is_leaf());
    EXPECT_TRUE(child->is_leaf());
    EXPECT_FALSE(child->is_root());
    EXPECT_EQ(child->parent(), root.get());
    EXPECT_EQ(root->children.size(), 1u);
    // Key fn for page_size=1 uses first element (5)
    EXPECT_NE(root->children.find(5), root->children.end());
}

TEST(RadixTreeNodeTest, GetMatchLen) {
    auto key_fn = get_key_fn(1);
    RadixTreeNode node(key_fn);
    node.set_key_value(torch::tensor({1, 2, 3, 4}, torch::kInt32),
                       torch::tensor({10, 20, 30, 40}, torch::kInt32));

    // Full match
    auto ids_full = torch::tensor({1, 2, 3, 4}, torch::kInt32);
    EXPECT_EQ(node.get_match_len(ids_full), 4);

    // Partial match
    auto ids_partial = torch::tensor({1, 2, 9, 8}, torch::kInt32);
    EXPECT_EQ(node.get_match_len(ids_partial), 2);

    // No match
    auto ids_none = torch::tensor({9, 8, 7, 6}, torch::kInt32);
    EXPECT_EQ(node.get_match_len(ids_none), 0);

    // Shorter input
    auto ids_short = torch::tensor({1, 2}, torch::kInt32);
    EXPECT_EQ(node.get_match_len(ids_short), 2);
}

TEST(RadixTreeNodeTest, SplitAt) {
    auto key_fn = get_key_fn(1);
    auto root = std::make_unique<RadixTreeNode>(key_fn);
    root->ref_count = 1;

    auto child = std::make_unique<RadixTreeNode>(key_fn);
    child->set_key_value(torch::tensor({1, 2, 3, 4}, torch::kInt32),
                         torch::tensor({10, 20, 30, 40}, torch::kInt32));
    child->set_parent(root.get());
    child->ref_count = 2;

    // Split at position 2: [1,2] becomes new intermediate, [3,4] stays
    auto* new_node = child->split_at(2);

    // New intermediate node
    EXPECT_EQ(new_node->length(), 2);
    EXPECT_TRUE(torch::equal(new_node->key(),
                              torch::tensor({1, 2}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(new_node->value(),
                              torch::tensor({10, 20}, torch::kInt32)));
    EXPECT_EQ(new_node->ref_count, 2);
    EXPECT_EQ(new_node->parent(), root.get());
    EXPECT_FALSE(new_node->is_leaf());

    // Original node now has suffix
    EXPECT_EQ(child->length(), 2);
    EXPECT_TRUE(torch::equal(child->key(),
                              torch::tensor({3, 4}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(child->value(),
                              torch::tensor({30, 40}, torch::kInt32)));
    EXPECT_EQ(child->parent(), new_node);

    // Cleanup the split node (not owned by unique_ptr here)
    delete new_node;
}

TEST(RadixTreeNodeTest, ComparisonByTimestamp) {
    auto key_fn = get_key_fn(1);
    RadixTreeNode a(key_fn, 100);
    RadixTreeNode b(key_fn, 200);

    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);
}

TEST(KeyFnTest, PageSizeOne) {
    auto key_fn = get_key_fn(1);
    auto tensor = torch::tensor({42, 99, 7}, torch::kInt32);
    EXPECT_EQ(key_fn(tensor), 42);
}

TEST(KeyFnTest, PageSizeGreaterThanOne) {
    auto key_fn = get_key_fn(2);
    auto tensor1 = torch::tensor({1, 2, 3}, torch::kInt32);
    auto tensor2 = torch::tensor({1, 3, 3}, torch::kInt32);  // differs at index 1
    auto tensor3 = torch::tensor({1, 2, 9}, torch::kInt32);  // same first 2

    // Same first page_size elements should produce the same hash
    EXPECT_EQ(key_fn(tensor1), key_fn(tensor3));

    // Different first page_size elements should produce different hash
    EXPECT_NE(key_fn(tensor1), key_fn(tensor2));
}

}  // namespace
}  // namespace sglang
