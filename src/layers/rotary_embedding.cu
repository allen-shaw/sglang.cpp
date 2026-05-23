#include "sglang/layers/rotary_embedding.h"
#include <flashinfer/pos_enc.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace sglang {
namespace {

template <typename scalar_t>
__global__ void qk_apply_rotary_pos_ids_cos_sin_cache_kernel(
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    const int32_t* __restrict__ positions,
    int64_t nnz,
    int64_t num_qo_heads,
    int64_t num_kv_heads,
    int64_t rotary_dim,
    int64_t head_dim,
    int64_t cos_cache_width,
    int64_t sin_cache_width,
    int64_t q_stride_n,
    int64_t q_stride_h,
    int64_t k_stride_n,
    int64_t k_stride_h) {
    const int64_t total_rows = nnz * (num_qo_heads + num_kv_heads);
    const int64_t row = blockIdx.x;
    if (row >= total_rows) {
        return;
    }

    scalar_t* row_ptr = nullptr;
    if (row < nnz * num_qo_heads) {
        const int64_t token = row / num_qo_heads;
        const int64_t head = row - token * num_qo_heads;
        row_ptr = query + token * q_stride_n + head * q_stride_h;
    } else {
        const int64_t k_row = row - nnz * num_qo_heads;
        const int64_t token = k_row / num_kv_heads;
        const int64_t head = k_row - token * num_kv_heads;
        row_ptr = key + token * k_stride_n + head * k_stride_h;
    }

    const int64_t token = row < nnz * num_qo_heads
                              ? row / num_qo_heads
                              : (row - nnz * num_qo_heads) / num_kv_heads;
    const int64_t pos = positions[token];
    const int64_t half_rotary_dim = rotary_dim / 2;

    for (int64_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        if (i >= rotary_dim) {
            continue;
        }
        const int64_t pair_idx = i < half_rotary_dim ? i : i - half_rotary_dim;
        const int64_t other_idx = i < half_rotary_dim ? i + half_rotary_dim : i - half_rotary_dim;
        const int64_t cache_idx = cos_cache_width == rotary_dim ? i : pair_idx;
        const int64_t sin_idx = sin_cache_width == rotary_dim ? i : pair_idx;

        const float val = static_cast<float>(row_ptr[i]);
        const float other = static_cast<float>(row_ptr[other_idx]);
        const float cos = cos_cache[pos * cos_cache_width + cache_idx];
        const float sin = sin_cache[pos * sin_cache_width + sin_idx];
        const float rotated = val * cos + (i < half_rotary_dim ? -other : other) * sin;
        row_ptr[i] = static_cast<scalar_t>(rotated);
    }
}

bool is_flashinfer_supported_head_dim(uint32_t head_dim) {
    return head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512;
}

}  // namespace

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

    if (!is_flashinfer_supported_head_dim(head_dim)) {
        TORCH_CHECK(rotary_dim_ % 2 == 0, "rotary_dim must be even");
        TORCH_CHECK(cos_cache.dim() == 2 && sin_cache.dim() == 2, "cos/sin cache must be rank-2");
        TORCH_CHECK(cos_cache.size(1) == rotary_dim_ || cos_cache.size(1) == rotary_dim_ / 2,
                    "cos cache width must be rotary_dim or rotary_dim / 2");
        TORCH_CHECK(sin_cache.size(1) == rotary_dim_ || sin_cache.size(1) == rotary_dim_ / 2,
                    "sin cache width must be rotary_dim or rotary_dim / 2");
        const int64_t total_rows = static_cast<int64_t>(nnz) * (num_qo_heads + num_kv_heads);
        dim3 blocks(total_rows);
        dim3 threads(128);
        if (query.scalar_type() == torch::kFloat16) {
            qk_apply_rotary_pos_ids_cos_sin_cache_kernel<half><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<half*>(query.data_ptr<at::Half>()),
                reinterpret_cast<half*>(key.data_ptr<at::Half>()),
                cos_cache.data_ptr<float>(),
                sin_cache.data_ptr<float>(),
                positions.data_ptr<int32_t>(),
                nnz, num_qo_heads, num_kv_heads, rotary_dim_, head_dim,
                cos_cache.size(1), sin_cache.size(1),
                q_stride_n, q_stride_h, k_stride_n, k_stride_h);
        } else {
            qk_apply_rotary_pos_ids_cos_sin_cache_kernel<nv_bfloat16><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<nv_bfloat16*>(query.data_ptr<at::BFloat16>()),
                reinterpret_cast<nv_bfloat16*>(key.data_ptr<at::BFloat16>()),
                cos_cache.data_ptr<float>(),
                sin_cache.data_ptr<float>(),
                positions.data_ptr<int32_t>(),
                nnz, num_qo_heads, num_kv_heads, rotary_dim_, head_dim,
                cos_cache.size(1), sin_cache.size(1),
                q_stride_n, q_stride_h, k_stride_n, k_stride_h);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }

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
