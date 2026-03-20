#include <gtest/gtest.h>
#include <torch/torch.h>
#include "sglang/layers/embedding.h"

using namespace sglang;

TEST(EmbeddingTest, VocabParallelEmbedding) {
    int num_embeddings = 1000;
    int embedding_dim = 128;
    int batch_size = 4;
    
    VocabParallelEmbedding emb(num_embeddings, embedding_dim);
    auto input = torch::randint(0, num_embeddings, {batch_size}, torch::device(torch::kCUDA).dtype(torch::kLong));
    
    auto output = emb.forward(input);
    
    EXPECT_EQ(output.size(0), batch_size);
    EXPECT_EQ(output.size(1), embedding_dim);
    EXPECT_EQ(output.scalar_type(), torch::kFloat16);
}

TEST(EmbeddingTest, ParallelLMHead) {
    int num_embeddings = 1000;
    int embedding_dim = 128;
    int batch_size = 4;
    
    ParallelLMHead lm_head(num_embeddings, embedding_dim, false, nullptr);
    auto input = torch::randn({batch_size, embedding_dim}, torch::device(torch::kCUDA).dtype(torch::kFloat16));
    
    auto output = lm_head.forward(input);
    
    EXPECT_EQ(output.size(0), batch_size);
    EXPECT_EQ(output.size(1), num_embeddings);
    EXPECT_EQ(output.scalar_type(), torch::kFloat16);
}
