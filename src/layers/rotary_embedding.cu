#include "sglang/layers/rotary_embedding.h"
#include <flashinfer/pos_enc.cuh>
#include <ATen/cuda/CUDAContext.h>

namespace sglang {

RotaryEmbedding::RotaryEmbedding(int head_size, int rotary_dim, int max_position_embeddings, float base)
    : head_size_(head_size), rotary_dim_(rotary_dim), max_position_embeddings_(max_position_embeddings), base_(base) {
    
    // Precompute cos/sin cache (matching Python rotary.py)
    // inv_freq = 1.0 / (base ^ (arange(0, rotary_dim, 2) / rotary_dim))
    auto inv_freq = 1.0 / torch::pow(
        base,
        torch::arange(0, rotary_dim, 2, torch::kFloat32) / static_cast<float>(rotary_dim)
    );
    
    // t = arange(max_position_embeddings)
    auto t = torch::arange(max_position_embeddings, torch::kFloat32);
    
    // freqs = outer(t, inv_freq) -> [max_pos, rotary_dim/2]
    auto freqs = torch::einsum("i,j->ij", {t, inv_freq});
    
    // cos/sin cache stored on CPU first, will be moved to CUDA when needed
    cos_cache_ = freqs.cos().contiguous();   // [max_pos, rotary_dim/2]
    sin_cache_ = freqs.sin().contiguous();   // [max_pos, rotary_dim/2]
}

void RotaryEmbedding::forward_inplace(
    torch::Tensor& positions, torch::Tensor& query, torch::Tensor& key, 
    const torch::Tensor& cos_cache, const torch::Tensor& sin_cache) {
    
    TORCH_CHECK(query.is_cuda() && key.is_cuda() && positions.is_cuda(), "Inputs must be CUDA tensors");

    uint32_t nnz = positions.size(0);
    uint32_t num_qo_heads = query.size(1);
    uint32_t num_kv_heads = key.size(1);
    uint32_t head_dim = query.size(2);
    
    size_t q_stride_n = query.stride(0);
    size_t q_stride_h = query.stride(1);
    size_t k_stride_n = key.stride(0);
    size_t k_stride_h = key.stride(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (query.scalar_type() == torch::kFloat16) {
        flashinfer::BatchQKApplyRotaryPosIdsCosSinCache<half, int32_t>(
            reinterpret_cast<half*>(query.data_ptr<at::Half>()),
            reinterpret_cast<half*>(key.data_ptr<at::Half>()),
            reinterpret_cast<half*>(query.data_ptr<at::Half>()),
            reinterpret_cast<half*>(key.data_ptr<at::Half>()),
            cos_cache.data_ptr<float>(),
            sin_cache.data_ptr<float>(),
            positions.data_ptr<int32_t>(),
            nnz, num_qo_heads, num_kv_heads, rotary_dim_, head_dim,
            q_stride_n, q_stride_h, k_stride_n, k_stride_h,
            q_stride_n, q_stride_h, k_stride_n, k_stride_h,
            false, stream
        );
    } else {
        flashinfer::BatchQKApplyRotaryPosIdsCosSinCache<nv_bfloat16, int32_t>(
            reinterpret_cast<nv_bfloat16*>(query.data_ptr<at::BFloat16>()),
            reinterpret_cast<nv_bfloat16*>(key.data_ptr<at::BFloat16>()),
            reinterpret_cast<nv_bfloat16*>(query.data_ptr<at::BFloat16>()),
            reinterpret_cast<nv_bfloat16*>(key.data_ptr<at::BFloat16>()),
            cos_cache.data_ptr<float>(),
            sin_cache.data_ptr<float>(),
            positions.data_ptr<int32_t>(),
            nnz, num_qo_heads, num_kv_heads, rotary_dim_, head_dim,
            q_stride_n, q_stride_h, k_stride_n, k_stride_h,
            q_stride_n, q_stride_h, k_stride_n, k_stride_h,
            false, stream
        );
    }
}

void RotaryEmbedding::forward_inplace(
    torch::Tensor& positions, torch::Tensor& query, torch::Tensor& key) {

    // Move cos/sin cache to the same device as query if needed (lazy)
    if (!cos_cache_.is_cuda() && query.is_cuda()) {
        cos_cache_ = cos_cache_.to(query.device());
        sin_cache_ = sin_cache_.to(query.device());
    }
    
    forward_inplace(positions, query, key, cos_cache_, sin_cache_);
}

}  // namespace sglang
