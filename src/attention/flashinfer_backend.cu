#include "sglang/attention/flashinfer_backend.h"
#include "sglang/core/batch.h"
#include "sglang/core/context.h"
#include "sglang/kvcache/base.h"

#include <algorithm>
#include <flashinfer/attention/decode_params.cuh>
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <c10/cuda/CUDAStream.h>

using namespace flashinfer;

namespace sglang {

torch::Tensor FlashInferAttnMetadata::get_last_indices(int bs) const {
    return indptr.slice(0, 1, bs + 1) - 1;
}

FlashInferBackend::FlashInferBackend(std::shared_ptr<BaseKVCachePool> kv_cache,
                                     int num_qo_heads, int num_kv_heads, int head_dim)
    : kv_cache_(kv_cache), num_qo_heads_(num_qo_heads),
      num_kv_heads_(num_kv_heads), head_dim_(head_dim) {
    auto device = kv_cache_->device();
    float_workspace_ = torch::empty({128 * 1024 * 1024}, torch::TensorOptions().dtype(torch::kByte).device(device));
    int_workspace_ = torch::empty({8 * 1024 * 1024}, torch::TensorOptions().dtype(torch::kByte).device(device));
    pinned_int_workspace_ = torch::empty({8 * 1024 * 1024}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCPU).pinned_memory(true));
}

void FlashInferBackend::prepare_metadata(Batch& batch) {
    auto metadata = std::make_shared<FlashInferAttnMetadata>();
    metadata->is_prefill = batch.is_prefill();

    int batch_size = batch.padded_reqs.size();
    auto device = kv_cache_->device();
    auto page_table = get_global_ctx()->page_table;

    std::vector<int32_t> qo_lens;
    std::vector<int32_t> kv_lens;
    std::vector<torch::Tensor> page_table_slices;
    page_table_slices.reserve(batch_size);

    for (const auto& req : batch.padded_reqs) {
        qo_lens.push_back(std::max(1, req->extend_len()));
        kv_lens.push_back(req->device_len());
        metadata->seq_lens.push_back(req->device_len());
        metadata->cached_lens.push_back(req->cached_len);
        page_table_slices.push_back(page_table[req->table_idx].slice(0, 0, req->device_len()));
    }

    if (!page_table_slices.empty()) {
        metadata->indices = torch::cat(page_table_slices).to(torch::kInt32).contiguous();
    } else {
        metadata->indices =
            torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    }

    const bool all_no_cache_hit = std::all_of(
        metadata->cached_lens.begin(), metadata->cached_lens.end(), [](int cached_len) {
            return cached_len == 0;
        });

    metadata->qo_indptr_host = {0};
    metadata->kv_indptr_host = {0};
    for (int len : kv_lens) {
        metadata->kv_indptr_host.push_back(metadata->kv_indptr_host.back() + len);
    }
    if (batch.is_decode()) {
        for (int i = 0; i < batch_size; ++i) {
            metadata->qo_indptr_host.push_back(metadata->qo_indptr_host.back() + 1);
        }
    } else if (all_no_cache_hit) {
        metadata->qo_indptr_host = metadata->kv_indptr_host;
    } else {
        for (int len : qo_lens) {
            metadata->qo_indptr_host.push_back(metadata->qo_indptr_host.back() + len);
        }
    }

    metadata->last_page_len_host.assign(batch_size, 1);

    metadata->indptr = torch::tensor(metadata->is_prefill ? metadata->qo_indptr_host : metadata->kv_indptr_host,
                                     torch::TensorOptions().dtype(torch::kInt32).device(device));
    if (metadata->is_prefill) {
        metadata->paged_kv_indptr = torch::tensor(metadata->kv_indptr_host, torch::TensorOptions().dtype(torch::kInt32).device(device));
    }
    metadata->paged_kv_last_page_len = torch::tensor(metadata->last_page_len_host, torch::TensorOptions().dtype(torch::kInt32).device(device));

    batch.attn_metadata = metadata;
}

torch::Tensor FlashInferBackend::forward(const torch::Tensor& q,
                                         const torch::Tensor& k,
                                         const torch::Tensor& v, int layer_id,
                                         Batch& batch) {
    auto meta = std::dynamic_pointer_cast<FlashInferAttnMetadata>(batch.attn_metadata);
    TORCH_CHECK(meta, "FlashInferBackend requires FlashInferAttnMetadata");

    auto k_3d = k.dim() == 2 ? k.view({-1, num_kv_heads_, head_dim_}) : k;
    auto v_3d = v.dim() == 2 ? v.view({-1, num_kv_heads_, head_dim_}) : v;
    kv_cache_->store_kv(k_3d, v_3d, batch.out_loc, layer_id);

    auto q_view = q.view({-1, num_qo_heads_, head_dim_});
    auto o = torch::empty_like(q_view);

    auto k_cache = kv_cache_->k_cache(layer_id);
    auto v_cache = kv_cache_->v_cache(layer_id);
    auto k_cache_view = k_cache.view({-1, num_kv_heads_, head_dim_});
    auto v_cache_view = v_cache.view({-1, num_kv_heads_, head_dim_});

    using DTypeQ = nv_bfloat16;
    using DTypeKV = nv_bfloat16;
    using DTypeO = nv_bfloat16;
    using IdType = int32_t;
    constexpr uint32_t HEAD_DIM = 128;
    constexpr auto POS_ENCODING_MODE = PosEncodingMode::kNone;

    int batch_size = batch.padded_reqs.size();
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr uint32_t GROUP_SIZE = 2;

    if (!meta->is_prefill) {
        using DecodeParamsT = BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType>;
        using DecodeAttentionVariant = ComposedAttention<DecodeParamsT, get_variant_code(false, true, false, false)>;

        DecodePlanInfo plan_info;
        auto work_estimation_func = BatchDecodeWithPagedKVCacheWorkEstimationDispatched<GROUP_SIZE, HEAD_DIM, POS_ENCODING_MODE, DecodeAttentionVariant>;

        cudaError_t status = DecodePlan<HEAD_DIM, POS_ENCODING_MODE, DecodeAttentionVariant>(
            float_workspace_.data_ptr(), float_workspace_.nbytes(),
            int_workspace_.data_ptr(), pinned_int_workspace_.data_ptr(), int_workspace_.nbytes(),
            plan_info, meta->kv_indptr_host.data(), batch_size, num_qo_heads_, 1, false, stream, work_estimation_func
        );
        TORCH_CHECK(status == cudaSuccess, "FlashInfer DecodePlan failed");

        const int64_t page_stride = k_cache_view.stride(0);
        const int64_t head_stride = k_cache_view.stride(1);
        // Our cache is stored as [num_pages, num_heads, head_dim] with page_size == 1.
        // FlashInfer's NHD view is [num_pages, page_size, num_heads, head_dim], so the
        // degenerate page entry dim shares the page stride while the head dim keeps stride(1).
        int64_t kv_strides[3] = {page_stride, page_stride, head_stride};
        paged_kv_t<DTypeKV, IdType> paged_kv(
            num_kv_heads_, 1, HEAD_DIM, batch_size, QKVLayout::kNHD,
            reinterpret_cast<DTypeKV*>(k_cache_view.data_ptr()),
            reinterpret_cast<DTypeKV*>(v_cache_view.data_ptr()),
            kv_strides,
            meta->indices.data_ptr<IdType>(),
            meta->indptr.data_ptr<IdType>(),
            meta->paged_kv_last_page_len.data_ptr<IdType>()
        );

        DecodeParamsT params(
            reinterpret_cast<DTypeQ*>(q_view.data_ptr()), nullptr, paged_kv,
            reinterpret_cast<DTypeO*>(o.data_ptr()), nullptr, nullptr,
            num_qo_heads_, num_qo_heads_ * head_dim_, head_dim_,
            -1, 0.0f, 1.0f / sqrt(head_dim_), 1.0f, 10000.0f
        );

        params.request_indices = GetPtrFromBaseOffset<IdType>(int_workspace_.data_ptr(), plan_info.request_indices_offset);
        params.kv_tile_indices = GetPtrFromBaseOffset<IdType>(int_workspace_.data_ptr(), plan_info.kv_tile_indices_offset);
        params.o_indptr = GetPtrFromBaseOffset<IdType>(int_workspace_.data_ptr(), plan_info.o_indptr_offset);
        params.kv_chunk_size_ptr = GetPtrFromBaseOffset<IdType>(int_workspace_.data_ptr(), plan_info.kv_chunk_size_ptr_offset);
        params.padded_batch_size = plan_info.padded_batch_size;

        DTypeO* tmp_v = plan_info.split_kv ? GetPtrFromBaseOffset<DTypeO>(float_workspace_.data_ptr(), plan_info.v_offset) : nullptr;
        float* tmp_s = plan_info.split_kv ? GetPtrFromBaseOffset<float>(float_workspace_.data_ptr(), plan_info.s_offset) : nullptr;

        status = BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM, POS_ENCODING_MODE, DecodeAttentionVariant>(
            params, tmp_v, tmp_s, stream);
        TORCH_CHECK(status == cudaSuccess, "FlashInfer Decode Run failed");
    } else {
        using PrefillParamsT = BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, IdType>;
        using PrefillAttentionVariant = ComposedAttention<PrefillParamsT, get_variant_code(false, true, false, false)>;

        PrefillPlanInfo plan_info;
        uint32_t total_num_rows = meta->qo_indptr_host[batch_size];

        cudaError_t status = PrefillPlan<IdType>(
            float_workspace_.data_ptr(), float_workspace_.nbytes(),
            int_workspace_.data_ptr(), pinned_int_workspace_.data_ptr(), int_workspace_.nbytes(),
            plan_info, meta->qo_indptr_host.data(), meta->kv_indptr_host.data(),
            total_num_rows, batch_size, num_qo_heads_, num_kv_heads_, head_dim_, 1, false, sizeof(DTypeO), stream
        );
        TORCH_CHECK(status == cudaSuccess, "FlashInfer PrefillPlan failed");

        const int64_t page_stride = k_cache_view.stride(0);
        const int64_t head_stride = k_cache_view.stride(1);
        // Our cache is stored as [num_pages, num_heads, head_dim] with page_size == 1.
        // FlashInfer's NHD view is [num_pages, page_size, num_heads, head_dim], so the
        // degenerate page entry dim shares the page stride while the head dim keeps stride(1).
        int64_t kv_strides[3] = {page_stride, page_stride, head_stride};
        paged_kv_t<DTypeKV, IdType> paged_kv(
            num_kv_heads_, 1, HEAD_DIM, batch_size, QKVLayout::kNHD,
            reinterpret_cast<DTypeKV*>(k_cache_view.data_ptr()),
            reinterpret_cast<DTypeKV*>(v_cache_view.data_ptr()),
            kv_strides,
            meta->indices.data_ptr<IdType>(),
            meta->paged_kv_indptr.data_ptr<IdType>(),
            meta->paged_kv_last_page_len.data_ptr<IdType>()
        );

        PrefillParamsT params(
            reinterpret_cast<DTypeQ*>(q_view.data_ptr()), paged_kv, nullptr,
            meta->indptr.data_ptr<IdType>(), meta->paged_kv_indptr.data_ptr<IdType>(), nullptr,
            reinterpret_cast<DTypeO*>(o.data_ptr()), nullptr, nullptr,
            num_qo_heads_, num_qo_heads_ * head_dim_, head_dim_,
            -1, 0.0f, 1.0f / sqrt(head_dim_), 1.0f, 10000.0f
        );

        params.request_indices = GetPtrFromBaseOffset<IdType>(int_workspace_.data_ptr(), plan_info.request_indices_offset);
        params.qo_tile_indices = GetPtrFromBaseOffset<IdType>(int_workspace_.data_ptr(), plan_info.qo_tile_indices_offset);
        params.kv_tile_indices = GetPtrFromBaseOffset<IdType>(int_workspace_.data_ptr(), plan_info.kv_tile_indices_offset);
        params.merge_indptr = GetPtrFromBaseOffset<IdType>(int_workspace_.data_ptr(), plan_info.merge_indptr_offset);
        params.o_indptr = GetPtrFromBaseOffset<IdType>(int_workspace_.data_ptr(), plan_info.o_indptr_offset);
        params.kv_chunk_size_ptr = GetPtrFromBaseOffset<IdType>(int_workspace_.data_ptr(), plan_info.kv_chunk_size_ptr_offset);
        params.block_valid_mask = plan_info.split_kv ? GetPtrFromBaseOffset<bool>(int_workspace_.data_ptr(), plan_info.block_valid_mask_offset) : nullptr;
        params.padded_batch_size = plan_info.padded_batch_size;
        params.max_total_num_rows = total_num_rows;
        params.total_num_rows = plan_info.enable_cuda_graph ? GetPtrFromBaseOffset<uint32_t>(int_workspace_.data_ptr(), plan_info.total_num_rows_offset) : nullptr;

        DTypeO* tmp_v = nullptr;
        float* tmp_s = nullptr;
        if (plan_info.split_kv) {
            tmp_v = GetPtrFromBaseOffset<DTypeO>(float_workspace_.data_ptr(), plan_info.v_offset);
            tmp_s = GetPtrFromBaseOffset<float>(float_workspace_.data_ptr(), plan_info.s_offset);
        }

#define DISPATCH_CTA_TILE(cta_tile, CTA_TILE, ...) \
    if (cta_tile == 128) { \
        constexpr uint32_t CTA_TILE = 128; \
        __VA_ARGS__ \
    } else if (cta_tile == 64) { \
        constexpr uint32_t CTA_TILE = 64; \
        __VA_ARGS__ \
    } else if (cta_tile == 16) { \
        constexpr uint32_t CTA_TILE = 16; \
        __VA_ARGS__ \
    } else { \
        TORCH_CHECK(false, "Unsupported cta_tile_q"); \
    }

        DISPATCH_CTA_TILE(plan_info.cta_tile_q, CTA_TILE_Q, {
            status = BatchPrefillWithPagedKVCacheDispatched<CTA_TILE_Q, HEAD_DIM, POS_ENCODING_MODE, false, MaskMode::kCausal, PrefillAttentionVariant>(
                params, tmp_v, tmp_s, stream);
        });
#undef DISPATCH_CTA_TILE

        TORCH_CHECK(status == cudaSuccess, "FlashInfer Prefill Run failed");
    }

    return o.view({-1, num_qo_heads_ * head_dim_});
}

} // namespace sglang
