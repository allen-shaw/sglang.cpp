#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <torch/torch.h>

#include "sglang/core/batch.h"
#include "sglang/core/context.h"
#include "sglang/engine/config.h"
#include "sglang/kvcache/base.h"

namespace sglang {

class GraphRunner;

struct BatchSamplingArgs {
    torch::Tensor temperatures;
    torch::Tensor top_k;
    torch::Tensor top_p;
};

struct ForwardOutput {
    torch::Tensor next_tokens_gpu;
    torch::Tensor next_tokens_cpu;
    std::shared_ptr<at::cuda::CUDAEvent> copy_done_event;

    void synchronize() const;
};

class Sampler {
 public:
    Sampler(torch::Device device, int vocab_size);

    BatchSamplingArgs prepare(const Batch& batch) const;
    torch::Tensor sample(const torch::Tensor& logits, const BatchSamplingArgs& args) const;

 private:
    torch::Device device_;
    int vocab_size_;
};

class Engine {
 public:
    explicit Engine(const EngineConfig& config);
    ~Engine();

    ForwardOutput forward_batch(Batch& batch, const BatchSamplingArgs& args);
    BatchSamplingArgs prepare_sampling_args(const Batch& batch) const;
    void shutdown();

    const torch::Device& device() const { return device_; }
    const torch::Tensor& page_table() const { return page_table_; }
    int max_seq_len() const { return max_seq_len_; }
    int num_pages() const { return num_pages_; }
    int eos_token_id() const { return model_config_.eos_token_id; }
    const ModelConfig& model_config() const { return model_config_; }

    void prepare_attention_metadata(Batch& batch);
    void pad_batch(Batch& batch);

 private:
    struct ModelRunner {
        std::shared_ptr<torch::nn::Module> module;
        std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&)> forward;
    };

    static ModelRunner create_model_runner(const ModelConfig& model_config);
    static void initialize_dummy_weights(torch::nn::Module& module, torch::Device device, torch::Dtype dtype);
    static int determine_num_pages(const EngineConfig& config, const ModelConfig& model_config);

    EngineConfig config_;
    ModelConfig model_config_;
    torch::Device device_;
    torch::Dtype dtype_;
    int max_seq_len_ = 0;
    int num_pages_ = 0;

    ModelRunner model_runner_;
    std::shared_ptr<Context> ctx_;
    std::shared_ptr<BaseKVCachePool> kv_cache_;
    std::shared_ptr<BaseAttnBackend> attn_backend_;
    std::unique_ptr<GraphRunner> graph_runner_;
    Sampler sampler_;
    c10::cuda::CUDAStream stream_;
    torch::Tensor page_table_;
    std::shared_ptr<Req> dummy_req_;
};

class GraphRunner {
 public:
    GraphRunner(torch::Device device,
                bool enable_cuda_graph,
                std::vector<int> batch_sizes,
                int max_batch_size,
                int max_seq_len,
                int vocab_size,
                std::shared_ptr<Context> ctx,
                std::shared_ptr<BaseAttnBackend> attn_backend,
                std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&)> model_forward,
                std::shared_ptr<Req> dummy_req,
                c10::cuda::CUDAStream stream);

    bool can_use_cuda_graph(const Batch& batch) const;
    void pad_batch(Batch& batch) const;
    torch::Tensor replay(Batch& batch);
    void destroy_cuda_graphs();

 private:
    struct GraphCaptureBuffer {
        torch::Tensor input_ids;
        torch::Tensor out_loc;
        torch::Tensor positions;

        static GraphCaptureBuffer init(int max_batch_size, torch::Device device);
        void set_batch(Batch& batch, int batch_size) const;
        void copy_from(const Batch& batch) const;
    };

    struct GraphCaptureState {
        int batch_size = 0;
        std::shared_ptr<Batch> batch;
        GraphCaptureBuffer buffer;
        torch::Tensor logits;
        std::unique_ptr<at::cuda::CUDAGraph> graph;
    };

    void capture_graphs(int max_seq_len, int vocab_size);
    static std::vector<int> determine_batch_sizes(bool enable_cuda_graph,
                                                  std::vector<int> batch_sizes,
                                                  int max_batch_size);

    torch::Device device_;
    bool enable_cuda_graph_ = false;
    std::vector<int> graph_batch_sizes_;
    int max_batch_size_ = 0;
    std::shared_ptr<Context> ctx_;
    std::shared_ptr<BaseAttnBackend> attn_backend_;
    std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&)> model_forward_;
    std::shared_ptr<Req> dummy_req_;
    c10::cuda::CUDAStream stream_;
    std::unordered_map<int, GraphCaptureState> graph_map_;
};

}  // namespace sglang
