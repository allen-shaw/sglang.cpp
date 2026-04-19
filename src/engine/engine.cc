#include "sglang/engine/engine.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

#include <ATen/ops/cumsum.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <cuda_runtime_api.h>
#include <torch/cuda.h>

#include "sglang/attention/flashinfer_backend.h"
#include "sglang/kvcache/mha_kvcache.h"
#include "sglang/models/llama.h"
#include "sglang/models/mistral.h"
#include "sglang/models/qwen.h"
#include "sglang/models/qwen3.h"
#include "sglang/models/qwen3_moe.h"
#include "sglang/models/weight_loader.h"
#include "sglang/utils/math_utils.h"

namespace sglang {

namespace {

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool contains_token(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

torch::Tensor make_tensor(const std::vector<float>& data, torch::Device device) {
    return torch::tensor(data, torch::TensorOptions().dtype(torch::kFloat32).device(device));
}

torch::Tensor make_tensor(const std::vector<int32_t>& data, torch::Device device) {
    return torch::tensor(data, torch::TensorOptions().dtype(torch::kInt32).device(device));
}

torch::Tensor sample_row(torch::Tensor logits,
                         float temperature,
                         int32_t top_k,
                         float top_p) {
    logits = logits / std::max(temperature, 1e-6F);
    if (top_k > 0 && top_k < logits.size(0)) {
        auto topk = torch::topk(logits, top_k);
        auto filtered =
            torch::full_like(logits, -std::numeric_limits<float>::infinity());
        filtered.index_put_({std::get<1>(topk)}, std::get<0>(topk));
        logits = filtered;
    }

    if (top_p < 1.0F) {
        auto sorted = torch::sort(logits, -1, /*descending=*/true);
        auto sorted_logits = std::get<0>(sorted);
        auto sorted_indices = std::get<1>(sorted);
        auto probs = torch::softmax(sorted_logits, -1);
        auto cumulative = torch::cumsum(probs, -1);
        auto cutoff = cumulative > top_p;
        if (cutoff.numel() > 0) {
            cutoff.index_put_({0}, false);
            sorted_logits.masked_fill_(cutoff, -std::numeric_limits<float>::infinity());
            auto filtered =
                torch::full_like(logits, -std::numeric_limits<float>::infinity());
            filtered.index_put_({sorted_indices}, sorted_logits);
            logits = filtered;
        }
    }

    auto probs = torch::softmax(logits, -1);
    return torch::multinomial(probs, 1);
}

torch::Tensor select_sampling_logits(const Batch& batch, const torch::Tensor& logits) {
    if (!batch.is_prefill()) {
        return logits.slice(0, 0, batch.size());
    }

    std::vector<int64_t> last_indices;
    last_indices.reserve(batch.reqs.size());
    int64_t offset = 0;
    for (size_t i = 0; i < batch.reqs.size(); ++i) {
        const auto& req = batch.reqs[i];
        const int64_t extend_len = req->extend_len();
        TORCH_CHECK(extend_len > 0, "Prefill req must have positive extend_len");
        last_indices.push_back(offset + extend_len - 1);
        offset += extend_len;
    }

    return logits.index_select(
        0,
        torch::tensor(last_indices, torch::TensorOptions().dtype(torch::kInt64).device(logits.device())));
}

}  // namespace

void ForwardOutput::synchronize() const {
    if (copy_done_event) {
        copy_done_event->synchronize();
    }
}

Sampler::Sampler(torch::Device device, int vocab_size)
    : device_(device), vocab_size_(vocab_size) {}

BatchSamplingArgs Sampler::prepare(const Batch& batch) const {
    std::vector<float> temperatures;
    std::vector<int32_t> top_ks;
    std::vector<float> top_ps;
    temperatures.reserve(batch.reqs.size());
    top_ks.reserve(batch.reqs.size());
    top_ps.reserve(batch.reqs.size());

    bool all_greedy = true;
    bool any_top_k = false;
    bool any_top_p = false;
    for (const auto& req : batch.reqs) {
        const auto& params = req->sampling_params;
        all_greedy = all_greedy && params.is_greedy();
        temperatures.push_back(std::max(params.is_greedy() ? 0.0F : params.temperature, 1e-6F));
        int32_t top_k = params.top_k >= 1 ? params.top_k : vocab_size_;
        float top_p = std::min(std::max(params.top_p, 1e-6F), 1.0F);
        any_top_k = any_top_k || top_k != vocab_size_;
        any_top_p = any_top_p || top_p < 1.0F;
        top_ks.push_back(top_k);
        top_ps.push_back(top_p);
    }

    BatchSamplingArgs args;
    if (all_greedy) {
        return args;
    }

    args.temperatures = make_tensor(temperatures, device_);
    if (any_top_k) {
        args.top_k = make_tensor(top_ks, device_);
    }
    if (any_top_p) {
        args.top_p = make_tensor(top_ps, device_);
    }
    return args;
}

torch::Tensor Sampler::sample(const torch::Tensor& logits,
                              const BatchSamplingArgs& args) const {
    if (!args.temperatures.defined()) {
        return torch::argmax(logits, -1);
    }

    auto logits_fp32 = logits.to(torch::kFloat32);
    std::vector<torch::Tensor> rows;
    rows.reserve(logits_fp32.size(0));
    for (int64_t i = 0; i < logits_fp32.size(0); ++i) {
        const float temperature = args.temperatures[i].item<float>();
        const int32_t top_k =
            args.top_k.defined() ? args.top_k[i].item<int32_t>() : vocab_size_;
        const float top_p = args.top_p.defined() ? args.top_p[i].item<float>() : 1.0F;
        rows.push_back(sample_row(logits_fp32[i], temperature, top_k, top_p));
    }
    return torch::cat(rows, 0);
}

Engine::ModelRunner Engine::create_model_runner(const ModelConfig& model_config) {
    auto choose_name = [&]() {
        if (!model_config.architectures.empty()) {
            return to_lower(model_config.architectures.front());
        }
        return to_lower(model_config.model_type);
    };

    const std::string name = choose_name();
    if (contains_token(name, "qwen3moe") || (model_config.is_moe() && contains_token(name, "qwen3"))) {
        auto model = std::make_shared<Qwen3MoEForCausalLM>(model_config);
        return {model, [model](const torch::Tensor& input_ids, const torch::Tensor& positions) {
                    return model->forward(input_ids, positions);
                }};
    }
    if (contains_token(name, "qwen3")) {
        auto model = std::make_shared<Qwen3ForCausalLM>(model_config);
        return {model, [model](const torch::Tensor& input_ids, const torch::Tensor& positions) {
                    return model->forward(input_ids, positions);
                }};
    }
    if (contains_token(name, "qwen")) {
        auto model = std::make_shared<Qwen2ForCausalLM>(model_config);
        return {model, [model](const torch::Tensor& input_ids, const torch::Tensor& positions) {
                    return model->forward(input_ids, positions);
                }};
    }
    if (contains_token(name, "mistral")) {
        auto model = std::make_shared<MistralForCausalLM>(model_config);
        return {model, [model](const torch::Tensor& input_ids, const torch::Tensor& positions) {
                    return model->forward(input_ids, positions);
                }};
    }

    auto model = std::make_shared<LlamaForCausalLM>(model_config);
    return {model, [model](const torch::Tensor& input_ids, const torch::Tensor& positions) {
                return model->forward(input_ids, positions);
            }};
}

void Engine::initialize_dummy_weights(torch::nn::Module& module,
                                      torch::Device device,
                                      torch::Dtype dtype) {
    torch::NoGradGuard no_grad;
    module.to(device, dtype);
    for (auto& entry : module.named_parameters(/*recurse=*/true)) {
        entry.value().copy_(torch::randn_like(entry.value()));
    }
}

int Engine::determine_num_pages(const EngineConfig& config,
                                const ModelConfig& model_config) {
    if (config.num_pages_override.has_value()) {
        return *config.num_pages_override;
    }

    size_t free_memory = 0;
    size_t total_memory = 0;
    cudaError_t status = cudaMemGetInfo(&free_memory, &total_memory);
    TORCH_CHECK(status == cudaSuccess, "cudaMemGetInfo failed");

    const size_t cache_per_page =
        2ULL * model_config.head_dim *
        div_even(model_config.num_kv_heads, 1, /*allow_replicate=*/true) *
        config.page_size * c10::elementSize(config.dtype) * model_config.num_layers;
    const size_t budget = static_cast<size_t>(free_memory * config.memory_ratio);
    int num_pages = static_cast<int>(budget / std::max<size_t>(cache_per_page, 1ULL));
    num_pages = std::max(num_pages, 2);
    return num_pages;
}

Engine::Engine(const EngineConfig& config)
    : config_(config),
      model_config_(config.load_model_config()),
      device_(config.device),
      dtype_(config.dtype),
      sampler_(config.device, model_config_.vocab_size),
      stream_(c10::cuda::getStreamFromPool(/*isHighPriority=*/false, device_.index())) {
    TORCH_CHECK(torch::cuda::is_available(), "CUDA is required for Engine");
    TORCH_CHECK(device_.is_cuda(), "Engine only supports CUDA devices");
    TORCH_CHECK(config_.page_size == 1,
                "Current Engine implementation requires page_size == 1");

    c10::cuda::CUDAGuard device_guard(device_);
    torch::manual_seed(config_.random_seed);
    model_config_ = config_.load_model_config();
    model_runner_ = create_model_runner(model_config_);

    if (config_.use_dummy_weight) {
        initialize_dummy_weights(*model_runner_.module, device_, dtype_);
    } else {
        TORCH_CHECK(!config_.model_path.empty(),
                    "model_path is required when use_dummy_weight=false");
        WeightLoader::load_weights(*model_runner_.module, config_.model_path, dtype_, device_);
    }

    num_pages_ = determine_num_pages(config_, model_config_);
    max_seq_len_ = std::min(config_.max_seq_len(), num_pages_ * config_.page_size);
    page_table_ = torch::zeros(
        {config_.max_running_req + 1, align_ceil(max_seq_len_, 32)},
        torch::TensorOptions().dtype(torch::kInt32).device(device_));

    kv_cache_ = std::make_shared<MHAKVCache>(
        model_config_.num_kv_heads,
        model_config_.num_layers,
        model_config_.head_dim,
        num_pages_ + 1,
        config_.page_size,
        dtype_,
        device_);
    attn_backend_ = std::make_shared<FlashInferBackend>(
        kv_cache_,
        model_config_.num_qo_heads,
        model_config_.num_kv_heads,
        model_config_.head_dim);

    ctx_ = std::make_shared<Context>(config_.page_size, attn_backend_);
    ctx_->page_table = page_table_;
    ctx_->kv_cache = kv_cache_;
    reset_global_ctx();
    set_global_ctx(ctx_);

    dummy_req_ = std::make_shared<Req>();
    dummy_req_->req_id = std::numeric_limits<uint64_t>::max();
    dummy_req_->input_ids =
        torch::tensor({0}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    dummy_req_->table_idx = config_.max_running_req;
    dummy_req_->cached_len = 0;
    dummy_req_->output_len = 1;
    dummy_req_->sampling_params = SamplingParams{};
    dummy_req_->initialize_runtime_state();
    page_table_[dummy_req_->table_idx].fill_(num_pages_ * config_.page_size);

    const int graph_max_bs =
        config_.cuda_graph_max_batch_size.value_or(config_.max_running_req);
    graph_runner_ = std::make_unique<GraphRunner>(
        device_,
        config_.enable_cuda_graph,
        config_.cuda_graph_batch_sizes,
        graph_max_bs,
        page_table_.size(1),
        model_config_.vocab_size,
        ctx_,
        attn_backend_,
        model_runner_.forward,
        dummy_req_,
        stream_);
}

Engine::~Engine() {
    shutdown();
}

ForwardOutput Engine::forward_batch(Batch& batch, const BatchSamplingArgs& args) {
    c10::InferenceMode inference_guard(true);
    c10::cuda::CUDAStreamGuard stream_guard(stream_);
    std::shared_ptr<Batch> batch_alias(&batch, [](Batch*) {});
    BatchGuard guard(ctx_, batch_alias);

    torch::Tensor logits;
    if (graph_runner_->can_use_cuda_graph(batch)) {
        logits = graph_runner_->replay(batch);
    } else {
        auto model_input_ids = batch.input_ids.to(torch::kInt64);
        logits = model_runner_.forward(model_input_ids, batch.positions);
    }

    auto sampling_logits = select_sampling_logits(batch, logits);

    for (const auto& req : batch.reqs) {
        req->complete_one();
    }

    auto sampled = sampler_.sample(sampling_logits, args).to(torch::kInt32);
    auto next_tokens_cpu = torch::empty(
        sampled.sizes(),
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    next_tokens_cpu.copy_(sampled, /*non_blocking=*/true);
    auto event = std::make_shared<at::cuda::CUDAEvent>();
    event->record(stream_);
    return ForwardOutput{sampled, next_tokens_cpu, event};
}

BatchSamplingArgs Engine::prepare_sampling_args(const Batch& batch) const {
    return sampler_.prepare(batch);
}

void Engine::shutdown() {
    if (graph_runner_) {
        graph_runner_->destroy_cuda_graphs();
    }
    reset_global_ctx();
}

void Engine::prepare_attention_metadata(Batch& batch) {
    TORCH_CHECK(attn_backend_, "Engine attention backend is not initialized");
    attn_backend_->prepare_metadata(batch);
}

void Engine::pad_batch(Batch& batch) {
    graph_runner_->pad_batch(batch);
}

GraphRunner::GraphRunner(torch::Device device,
                         bool enable_cuda_graph,
                         std::vector<int> batch_sizes,
                         int max_batch_size,
                         int max_seq_len,
                         int vocab_size,
                         std::shared_ptr<Context> ctx,
                         std::shared_ptr<BaseAttnBackend> attn_backend,
                         std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&)> model_forward,
                         std::shared_ptr<Req> dummy_req,
                         c10::cuda::CUDAStream stream)
    : device_(device),
      enable_cuda_graph_(enable_cuda_graph),
      graph_batch_sizes_(determine_batch_sizes(enable_cuda_graph, std::move(batch_sizes), max_batch_size)),
      max_batch_size_(graph_batch_sizes_.empty() ? 0 : graph_batch_sizes_.back()),
      ctx_(std::move(ctx)),
      attn_backend_(std::move(attn_backend)),
      model_forward_(std::move(model_forward)),
      dummy_req_(std::move(dummy_req)),
      stream_(stream) {
    capture_graphs(max_seq_len, vocab_size);
}

std::vector<int> GraphRunner::determine_batch_sizes(bool enable_cuda_graph,
                                                    std::vector<int> batch_sizes,
                                                    int max_batch_size) {
    if (!enable_cuda_graph || max_batch_size < 1) {
        return {};
    }

    if (batch_sizes.empty()) {
        batch_sizes = {1, 2, 4};
        for (int bs = 8; bs <= max_batch_size; bs += 8) {
            batch_sizes.push_back(bs);
        }
    }

    std::sort(batch_sizes.begin(), batch_sizes.end());
    batch_sizes.erase(
        std::remove_if(batch_sizes.begin(), batch_sizes.end(),
                       [&](int bs) { return bs < 1 || bs > max_batch_size; }),
        batch_sizes.end());
    batch_sizes.erase(std::unique(batch_sizes.begin(), batch_sizes.end()), batch_sizes.end());
    return batch_sizes;
}

bool GraphRunner::can_use_cuda_graph(const Batch& batch) const {
    return enable_cuda_graph_ && batch.is_decode() &&
           !graph_batch_sizes_.empty() &&
           batch.size() <= max_batch_size_;
}

void GraphRunner::pad_batch(Batch& batch) const {
    if (!can_use_cuda_graph(batch)) {
        batch.padded_reqs = batch.reqs;
        return;
    }

    const auto it =
        std::lower_bound(graph_batch_sizes_.begin(), graph_batch_sizes_.end(), batch.size());
    if (it == graph_batch_sizes_.end()) {
        batch.padded_reqs = batch.reqs;
        return;
    }

    batch.padded_reqs = batch.reqs;
    batch.padded_reqs.insert(batch.padded_reqs.end(),
                             static_cast<size_t>(*it - batch.size()),
                             dummy_req_);
}

GraphRunner::GraphCaptureBuffer GraphRunner::GraphCaptureBuffer::init(int max_batch_size,
                                                                      torch::Device device) {
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
    return GraphCaptureBuffer{
        torch::zeros({max_batch_size}, options),
        torch::zeros({max_batch_size}, options),
        torch::zeros({max_batch_size}, options),
    };
}

void GraphRunner::GraphCaptureBuffer::set_batch(Batch& batch, int batch_size) const {
    batch.input_ids = input_ids.slice(0, 0, batch_size);
    batch.out_loc = out_loc.slice(0, 0, batch_size);
    batch.positions = positions.slice(0, 0, batch_size);
}

void GraphRunner::GraphCaptureBuffer::copy_from(const Batch& batch) const {
    const auto count = batch.padded_size();
    input_ids.slice(0, 0, count).copy_(batch.input_ids);
    out_loc.slice(0, 0, count).copy_(batch.out_loc);
    positions.slice(0, 0, count).copy_(batch.positions);
}

void GraphRunner::capture_graphs(int max_seq_len, int vocab_size) {
    graph_map_.clear();
    if (graph_batch_sizes_.empty()) {
        return;
    }

    c10::InferenceMode inference_guard(true);
    attn_backend_->init_capture_graph(max_seq_len, graph_batch_sizes_);

    c10::cuda::CUDAGuard device_guard(device_);
    c10::cuda::CUDAStreamGuard stream_guard(stream_);
    torch::cuda::synchronize(device_.index());

    std::optional<at::cuda::MempoolId_t> pool;
    for (auto it = graph_batch_sizes_.rbegin(); it != graph_batch_sizes_.rend(); ++it) {
        const int batch_size = *it;
        GraphCaptureState state;
        state.batch_size = batch_size;
        state.batch = std::make_shared<Batch>();
        state.batch->phase = BatchPhase::Decode;
        state.batch->reqs.assign(batch_size, dummy_req_);
        state.batch->padded_reqs = state.batch->reqs;
        state.buffer = GraphCaptureBuffer::init(batch_size, device_);
        state.buffer.set_batch(*state.batch, batch_size);

        attn_backend_->prepare_for_capture(*state.batch);
        state.buffer.out_loc.copy_(
            get_global_ctx()->page_table[dummy_req_->table_idx].slice(0, 0, batch_size));

        {
            BatchGuard guard(ctx_, state.batch);
            state.logits = model_forward_(state.batch->input_ids.to(torch::kInt64),
                                          state.batch->positions);

            state.graph = std::make_unique<at::cuda::CUDAGraph>();
            if (pool.has_value()) {
                state.graph->capture_begin(*pool);
            } else {
                state.graph->capture_begin();
            }
            state.logits = model_forward_(state.batch->input_ids.to(torch::kInt64),
                                          state.batch->positions);
            state.graph->capture_end();
        }

        if (!pool.has_value()) {
            pool = state.graph->pool();
        }
        graph_map_.emplace(batch_size, std::move(state));
    }
}

torch::Tensor GraphRunner::replay(Batch& batch) {
    TORCH_CHECK(can_use_cuda_graph(batch), "CUDA graph replay is not available for this batch");
    const auto it = graph_map_.find(batch.padded_size());
    TORCH_CHECK(it != graph_map_.end(), "No CUDA graph captured for batch size ",
                batch.padded_size());

    auto& state = it->second;
    state.buffer.copy_from(batch);
    attn_backend_->prepare_for_replay(batch);
    state.graph->replay();
    return state.logits.slice(0, 0, batch.size());
}

void GraphRunner::destroy_cuda_graphs() {
    graph_map_.clear();
}

}  // namespace sglang
