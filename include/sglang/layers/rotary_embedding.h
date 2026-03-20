#pragma once
#include <torch/torch.h>

namespace sglang {

class RotaryEmbedding {
 public:
    RotaryEmbedding(int head_size, int rotary_dim, int max_position_embeddings, float base);
    
    void forward_inplace(
        torch::Tensor& positions, 
        torch::Tensor& query, 
        torch::Tensor& key, 
        const torch::Tensor& cos_cache,
        const torch::Tensor& sin_cache);

 private:
    int head_size_;
    int rotary_dim_;
    int max_position_embeddings_;
    float base_;
};

}  // namespace sglang
