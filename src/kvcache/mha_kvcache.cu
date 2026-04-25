#include "sglang/kvcache/mha_kvcache.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace sglang {
namespace {

template <typename scalar_t, typename index_t>
__global__ void store_kv_cache_kernel(scalar_t* __restrict__ k_cache,
                                      scalar_t* __restrict__ v_cache,
                                      const scalar_t* __restrict__ k,
                                      const scalar_t* __restrict__ v,
                                      const index_t* __restrict__ out_loc,
                                      int64_t num_tokens,
                                      int64_t elems_per_token,
                                      int64_t k_stride_token,
                                      int64_t v_stride_token) {
    constexpr int kWarpSize = 32;
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int warps_per_block = blockDim.x / kWarpSize;
    const int64_t token_idx = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_id;
    if (token_idx >= num_tokens) {
        return;
    }

    const int64_t cache_pos = static_cast<int64_t>(out_loc[token_idx]);
    scalar_t* dst_k = k_cache + cache_pos * elems_per_token;
    scalar_t* dst_v = v_cache + cache_pos * elems_per_token;
    const scalar_t* src_k = k + token_idx * k_stride_token;
    const scalar_t* src_v = v + token_idx * v_stride_token;

    for (int64_t elem = lane; elem < elems_per_token; elem += kWarpSize) {
        dst_k[elem] = src_k[elem];
        dst_v[elem] = src_v[elem];
    }
}

template <typename scalar_t>
void launch_store_kv_cache(torch::Tensor& k_buf,
                           torch::Tensor& v_buf,
                           const torch::Tensor& k,
                           const torch::Tensor& v,
                           const torch::Tensor& out_loc,
                           int64_t num_tokens,
                           int64_t elems_per_token,
                           int64_t k_stride_token,
                           int64_t v_stride_token) {
    constexpr int threads = 128;
    constexpr int warps_per_block = threads / 32;
    const dim3 blocks((num_tokens + warps_per_block - 1) / warps_per_block);
    auto stream = at::cuda::getCurrentCUDAStream();

    if (out_loc.scalar_type() == torch::kInt32) {
        store_kv_cache_kernel<scalar_t, int32_t><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<scalar_t*>(k_buf.data_ptr()),
            reinterpret_cast<scalar_t*>(v_buf.data_ptr()),
            reinterpret_cast<const scalar_t*>(k.data_ptr()),
            reinterpret_cast<const scalar_t*>(v.data_ptr()),
            out_loc.data_ptr<int32_t>(),
            num_tokens,
            elems_per_token,
            k_stride_token,
            v_stride_token);
    } else {
        TORCH_CHECK(out_loc.scalar_type() == torch::kInt64,
                    "out_loc must be int32 or int64");
        store_kv_cache_kernel<scalar_t, int64_t><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<scalar_t*>(k_buf.data_ptr()),
            reinterpret_cast<scalar_t*>(v_buf.data_ptr()),
            reinterpret_cast<const scalar_t*>(k.data_ptr()),
            reinterpret_cast<const scalar_t*>(v.data_ptr()),
            out_loc.data_ptr<int64_t>(),
            num_tokens,
            elems_per_token,
            k_stride_token,
            v_stride_token);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

MHAKVCache::MHAKVCache(int num_kv_heads, int num_layers, int head_dim,
                       int num_pages, int page_size,
                       torch::Dtype dtype, torch::Device device)
    : device_(device), num_layers_(num_layers),
      num_pages_(num_pages), page_size_(page_size),
      kv_heads_(num_kv_heads), head_dim_(head_dim) {
    int64_t total_slots = static_cast<int64_t>(num_pages) * page_size;
    kv_buffer_ = torch::empty(
        {2, num_layers, total_slots, num_kv_heads, head_dim},
        torch::TensorOptions().dtype(dtype).device(device)
    );
    k_buffer_ = kv_buffer_[0];
    v_buffer_ = kv_buffer_[1];
}

torch::Tensor MHAKVCache::k_cache(int layer_id) const {
    return k_buffer_[layer_id];
}

torch::Tensor MHAKVCache::v_cache(int layer_id) const {
    return v_buffer_[layer_id];
}

void MHAKVCache::store_kv(const torch::Tensor& key, const torch::Tensor& val,
                          const torch::Tensor& out_loc, int layer_id) {
    TORCH_CHECK(key.is_cuda() && val.is_cuda() && out_loc.is_cuda(),
                "key, val and out_loc must be CUDA tensors");
    TORCH_CHECK(key.scalar_type() == val.scalar_type(),
                "key and val dtype must match");
    TORCH_CHECK(key.stride(-1) == 1 && val.stride(-1) == 1,
                "key and val last dimension must be contiguous");
    TORCH_CHECK(out_loc.dim() == 1, "out_loc must be rank 1");

    auto k = key.dim() == 2 ? key.view({-1, kv_heads_, head_dim_}) : key;
    auto v = val.dim() == 2 ? val.view({-1, kv_heads_, head_dim_}) : val;
    TORCH_CHECK(k.dim() == 3 && v.dim() == 3, "key and val must be rank 2 or 3");
    TORCH_CHECK(k.size(0) == v.size(0) && k.size(0) == out_loc.size(0),
                "key, val and out_loc token counts must match");
    TORCH_CHECK(k.size(1) == kv_heads_ && v.size(1) == kv_heads_ &&
                k.size(2) == head_dim_ && v.size(2) == head_dim_,
                "key and val shape must match KV cache layout");

    auto k_buf = k_buffer_[layer_id];
    auto v_buf = v_buffer_[layer_id];
    TORCH_CHECK(k_buf.is_contiguous() && v_buf.is_contiguous(),
                "KV cache buffers must be contiguous");

    const int64_t num_tokens = k.size(0);
    const int64_t elems_per_token = static_cast<int64_t>(kv_heads_) * head_dim_;
    const int64_t k_stride_token = k.stride(0);
    const int64_t v_stride_token = v.stride(0);
    if (num_tokens == 0) {
        return;
    }

    if (k.scalar_type() == torch::kFloat16) {
        launch_store_kv_cache<half>(
            k_buf, v_buf, k, v, out_loc, num_tokens, elems_per_token,
            k_stride_token, v_stride_token);
    } else {
        TORCH_CHECK(k.scalar_type() == torch::kBFloat16,
                    "KV cache store supports FP16/BF16 tensors");
        launch_store_kv_cache<nv_bfloat16>(
            k_buf, v_buf, k, v, out_loc, num_tokens, elems_per_token,
            k_stride_token, v_stride_token);
    }
}

}  // namespace sglang
