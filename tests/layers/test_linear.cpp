#include <gtest/gtest.h>
#include <torch/torch.h>
#include "sglang/layers/linear.h"

using namespace sglang;

TEST(LinearTest, Replicated) {
    int batch_size = 2;
    int in_features = 128;
    int out_features = 64;
    
    LinearReplicated linear(in_features, out_features, true);
    auto input = torch::randn({batch_size, in_features}, torch::device(torch::kCUDA).dtype(torch::kFloat16));
    
    auto output = linear.forward(input);
    
    EXPECT_EQ(output.size(0), batch_size);
    EXPECT_EQ(output.size(1), out_features);
    EXPECT_EQ(output.scalar_type(), torch::kFloat16);
}

TEST(LinearTest, RowParallel) {
    int batch_size = 2;
    int in_features = 128;
    int out_features = 64;
    
    LinearRowParallel linear(in_features, out_features, false);
    auto input = torch::randn({batch_size, in_features}, torch::device(torch::kCUDA).dtype(torch::kFloat16));
    
    auto output = linear.forward(input);
    
    EXPECT_EQ(output.size(0), batch_size);
    EXPECT_EQ(output.size(1), out_features);
}

TEST(LinearTest, ColParallelMerged) {
    int batch_size = 2;
    int in_features = 128;
    std::vector<int> out_features = {64, 32};
    
    LinearColParallelMerged linear(in_features, out_features, true);
    auto input = torch::randn({batch_size, in_features}, torch::device(torch::kCUDA).dtype(torch::kFloat16));
    
    auto output = linear.forward(input);
    
    EXPECT_EQ(output.size(0), batch_size);
    EXPECT_EQ(output.size(1), 64 + 32);
}

TEST(LinearTest, QKVMerged) {
    int batch_size = 2;
    int hidden_size = 128;
    int head_dim = 64;
    int num_qo_heads = 4;
    int num_kv_heads = 2;
    
    LinearQKVMerged linear(hidden_size, head_dim, num_qo_heads, num_kv_heads, true);
    auto input = torch::randn({batch_size, hidden_size}, torch::device(torch::kCUDA).dtype(torch::kFloat16));
    
    auto output = linear.forward(input);
    
    EXPECT_EQ(output.size(0), batch_size);
    EXPECT_EQ(output.size(1), (num_qo_heads + 2 * num_kv_heads) * head_dim);
}

TEST(LinearTest, OProj) {
    int batch_size = 2;
    int in_features = 128;
    int out_features = 64;
    
    LinearOProj linear(in_features, out_features, true);
    auto input = torch::randn({batch_size, in_features}, torch::device(torch::kCUDA).dtype(torch::kFloat16));
    
    auto output = linear.forward(input);
    
    EXPECT_EQ(output.size(0), batch_size);
    EXPECT_EQ(output.size(1), out_features);
}
