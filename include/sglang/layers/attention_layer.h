#pragma once

#include <torch/torch.h>
#include "sglang/layers/normalization.h"
#include "sglang/layers/rotary_embedding.h"

namespace sglang {

class AttentionLayer {
 public:
    AttentionLayer(int layer_id, int num_qo_heads, int num_kv_heads, int head_dim,
                   RotaryEmbedding* rotary, RMSNorm* q_norm = nullptr, RMSNorm* k_norm = nullptr);
                   
    torch::Tensor forward(const torch::Tensor& qkv, const torch::Tensor& positions);

 private:
    int layer_id_;
    int head_dim_;
    int num_qo_heads_;
    int num_kv_heads_;
    int qo_attn_dim_;
    int kv_attn_dim_;
    
    RotaryEmbedding* rotary_;
    RMSNorm* q_norm_;
    RMSNorm* k_norm_;
};

}  // namespace sglang
