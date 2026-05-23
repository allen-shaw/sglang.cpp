#include <gtest/gtest.h>
#include <torch/torch.h>
#include "sglang/distributed/distributed.h"
#include "sglang/layers/linear.h"

using namespace sglang;

TEST(LinearTest, Replicated) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required for Linear layer tests";
    }

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
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required for Linear layer tests";
    }

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
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required for Linear layer tests";
    }

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
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required for Linear layer tests";
    }

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
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required for Linear layer tests";
    }

    int batch_size = 2;
    int in_features = 128;
    int out_features = 64;
    
    LinearOProj linear(in_features, out_features, true);
    auto input = torch::randn({batch_size, in_features}, torch::device(torch::kCUDA).dtype(torch::kFloat16));
    
    auto output = linear.forward(input);
    
    EXPECT_EQ(output.size(0), batch_size);
    EXPECT_EQ(output.size(1), out_features);
}

TEST(LinearTest, TensorParallelShapes) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required because Linear weights are CUDA tensors";
    }

    set_tp_info_for_test(/*rank=*/1, /*size=*/2);

    LinearColParallelMerged col(/*input_size=*/128, std::vector<int>{64, 32}, /*has_bias=*/false);
    EXPECT_EQ(col.weight.sizes().vec(), std::vector<int64_t>({48, 128}));

    LinearRowParallel row(/*input_size=*/128, /*output_size=*/64, /*has_bias=*/false);
    EXPECT_EQ(row.weight.sizes().vec(), std::vector<int64_t>({64, 64}));

    LinearQKVMerged qkv(/*hidden_size=*/128, /*head_dim=*/64, /*num_qo_heads=*/4, /*num_kv_heads=*/2, false);
    EXPECT_EQ(qkv.weight.sizes().vec(), std::vector<int64_t>({256, 128}));
    EXPECT_EQ(qkv.local_num_qo_heads(), 2);
    EXPECT_EQ(qkv.local_num_kv_heads(), 1);

    LinearOProj o_proj(/*input_size=*/256, /*output_size=*/128, /*has_bias=*/false);
    EXPECT_EQ(o_proj.weight.sizes().vec(), std::vector<int64_t>({128, 128}));

    reset_tp_info_for_test();
}
