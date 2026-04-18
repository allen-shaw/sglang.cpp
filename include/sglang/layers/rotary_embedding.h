#pragma once
#include <torch/torch.h>

namespace sglang {

class RotaryEmbedding {
 public:
    RotaryEmbedding(int head_size, int rotary_dim, int max_position_embeddings, float base);
    
    // Original version: caller provides cos/sin cache
    void forward_inplace(
        torch::Tensor& positions, 
        torch::Tensor& query, 
        torch::Tensor& key, 
        const torch::Tensor& cos_cache,
        const torch::Tensor& sin_cache);

    // New version: uses internal precomputed cos/sin cache (matches Python API)
    void forward_inplace(
        torch::Tensor& positions, 
        torch::Tensor& query, 
        torch::Tensor& key);

    const torch::Tensor& cos_cache() const { return cos_cache_; }
    const torch::Tensor& sin_cache() const { return sin_cache_; }

 private:
    int head_size_;
    int rotary_dim_;
    int max_position_embeddings_;
    float base_;
    torch::Tensor cos_cache_;  // [max_pos, rotary_dim/2]
    torch::Tensor sin_cache_;  // [max_pos, rotary_dim/2]
};

}  // namespace sglang
