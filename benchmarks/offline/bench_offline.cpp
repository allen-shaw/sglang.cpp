#include <benchmark/benchmark.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "sglang/llm/llm.h"

namespace sglang {
namespace {

struct BenchOptions {
  std::string model_path;
  torch::Dtype dtype = torch::kBFloat16;
  bool use_dummy_weight = false;
  int max_running_req = 256;
  int max_extend_tokens = 16384;
  int page_size = 1;
  float memory_ratio = 0.2F;
  int seed = 0;
  int warmup_tokens = 8;
  std::optional<int> max_seq_len_override = 4096;
  std::optional<int> num_pages_override;
  bool enable_cuda_graph = false;
  std::optional<int> cuda_graph_max_batch_size;
  std::vector<int> cuda_graph_batch_sizes;
  int cuda_graph_capture_max_seq_len = 4096;
  std::vector<int> batch_sizes = {1, 8, 32, 128};
  std::vector<int> input_lengths = {128, 512, 1024};
  std::vector<int> output_lengths = {128, 512};
};

struct BatchInputs {
  std::vector<torch::Tensor> prompt_token_ids;
  std::vector<SamplingParams> sampling_params;
  int64_t prompt_tokens = 0;
  int64_t requested_output_tokens = 0;
};

BenchOptions g_options;
std::unique_ptr<LLM> g_llm;
bool g_show_help = false;

torch::Dtype parse_dtype(const std::string& value) {
  if (value == "float16" || value == "fp16") {
    return torch::kFloat16;
  }
  if (value == "bfloat16" || value == "bf16") {
    return torch::kBFloat16;
  }
  if (value == "float32" || value == "fp32") {
    return torch::kFloat32;
  }
  throw std::invalid_argument("Unsupported dtype: " + value);
}

std::string require_value(int argc, char** argv, int* index) {
  if (*index + 1 >= argc) {
    throw std::invalid_argument("Missing value for argument: " + std::string(argv[*index]));
  }
  ++(*index);
  return argv[*index];
}

std::vector<int> parse_csv_ints(const std::string& value) {
  std::vector<int> result;
  size_t start = 0;
  while (start < value.size()) {
    size_t end = value.find(',', start);
    if (end == std::string::npos) {
      end = value.size();
    }
    result.push_back(std::stoi(value.substr(start, end - start)));
    start = end + 1;
  }
  if (result.empty()) {
    throw std::invalid_argument("Expected non-empty comma separated integer list");
  }
  return result;
}

void print_usage(const char* binary) {
  std::cout
      << "Usage: " << binary
      << " --model-path PATH [options] [--benchmark_* Google Benchmark flags]\n"
      << "Options:\n"
      << "  --dtype {float16|bfloat16|float32}\n"
      << "  --dummy-weight\n"
      << "  --max-running-requests N\n"
      << "  --max-prefill-length N\n"
      << "  --page-size N\n"
      << "  --memory-ratio FLOAT\n"
      << "  --seed N\n"
      << "  --warmup-tokens N\n"
      << "  --max-seq-len-override N\n"
      << "  --num-pages N\n"
      << "  --graph N | --cuda-graph-max-bs N\n"
      << "  --cuda-graph-batch-sizes 1,2,4,8\n"
      << "  --cuda-graph-capture-max-seq-len N\n"
      << "  --batch-sizes 1,8,32\n"
      << "  --input-lens 128,512,1024\n"
      << "  --output-lens 128,512\n";
}

void parse_custom_args(int* argc, char** argv) {
  int write_index = 1;
  for (int i = 1; i < *argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      g_show_help = true;
      print_usage(argv[0]);
      continue;
    }
    if (arg == "--model-path" || arg == "--model") {
      g_options.model_path = require_value(*argc, argv, &i);
    } else if (arg == "--dtype") {
      g_options.dtype = parse_dtype(require_value(*argc, argv, &i));
    } else if (arg == "--dummy-weight") {
      g_options.use_dummy_weight = true;
    } else if (arg == "--max-running-requests") {
      g_options.max_running_req = std::stoi(require_value(*argc, argv, &i));
    } else if (arg == "--max-prefill-length" || arg == "--max-extend-length") {
      g_options.max_extend_tokens = std::stoi(require_value(*argc, argv, &i));
    } else if (arg == "--page-size") {
      g_options.page_size = std::stoi(require_value(*argc, argv, &i));
    } else if (arg == "--memory-ratio") {
      g_options.memory_ratio = std::stof(require_value(*argc, argv, &i));
    } else if (arg == "--seed") {
      g_options.seed = std::stoi(require_value(*argc, argv, &i));
    } else if (arg == "--warmup-tokens") {
      g_options.warmup_tokens = std::stoi(require_value(*argc, argv, &i));
    } else if (arg == "--max-seq-len-override") {
      g_options.max_seq_len_override = std::stoi(require_value(*argc, argv, &i));
    } else if (arg == "--num-pages") {
      g_options.num_pages_override = std::stoi(require_value(*argc, argv, &i));
    } else if (arg == "--graph" || arg == "--cuda-graph-max-bs") {
      g_options.enable_cuda_graph = true;
      g_options.cuda_graph_max_batch_size = std::stoi(require_value(*argc, argv, &i));
    } else if (arg == "--cuda-graph-batch-sizes") {
      g_options.enable_cuda_graph = true;
      g_options.cuda_graph_batch_sizes = parse_csv_ints(require_value(*argc, argv, &i));
    } else if (arg == "--cuda-graph-capture-max-seq-len") {
      g_options.cuda_graph_capture_max_seq_len = std::stoi(require_value(*argc, argv, &i));
    } else if (arg == "--batch-sizes") {
      g_options.batch_sizes = parse_csv_ints(require_value(*argc, argv, &i));
    } else if (arg == "--input-lens") {
      g_options.input_lengths = parse_csv_ints(require_value(*argc, argv, &i));
    } else if (arg == "--output-lens") {
      g_options.output_lengths = parse_csv_ints(require_value(*argc, argv, &i));
    } else {
      argv[write_index++] = argv[i];
    }
  }
  *argc = write_index;

  if (g_show_help) {
    return;
  }

  if (g_options.model_path.empty()) {
    throw std::invalid_argument("--model-path is required");
  }
}

ServerArgs make_server_args() {
  ServerArgs args;
  args.model_path = g_options.model_path;
  args.dtype = g_options.dtype;
  args.max_running_req = g_options.max_running_req;
  args.max_extend_tokens = g_options.max_extend_tokens;
  args.memory_ratio = g_options.memory_ratio;
  args.page_size = g_options.page_size;
  args.num_pages_override = g_options.num_pages_override;
  args.max_seq_len_override = g_options.max_seq_len_override;
  args.use_dummy_weight = g_options.use_dummy_weight;
  args.enable_cuda_graph = g_options.enable_cuda_graph;
  args.cuda_graph_max_batch_size = g_options.cuda_graph_max_batch_size;
  args.cuda_graph_batch_sizes = g_options.cuda_graph_batch_sizes;
  args.cuda_graph_capture_max_seq_len = g_options.cuda_graph_capture_max_seq_len;
  args.num_tokenizer_threads = 0;
  return args;
}

LLM& get_llm() {
  if (!g_llm) {
    g_llm = std::make_unique<LLM>(make_server_args());
    SamplingParams warmup;
    warmup.temperature = 0.1F;
    warmup.top_p = 1.0F;
    warmup.top_k = -1;
    warmup.ignore_eos = true;
    warmup.max_new_tokens = g_options.warmup_tokens;
    (void)g_llm->generate(
        torch::tensor({1, 2, 3, 4}, torch::TensorOptions().dtype(torch::kInt32)), warmup);
  }
  return *g_llm;
}

BatchInputs make_batch_inputs(int batch_size, int input_len, int output_len) {
  std::mt19937 rng(static_cast<uint32_t>(g_options.seed + batch_size * 17 + input_len * 31 + output_len * 13));
  std::uniform_int_distribution<int32_t> token_dist(0, 10000);

  BatchInputs batch;
  batch.prompt_token_ids.reserve(batch_size);
  batch.sampling_params.reserve(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    std::vector<int32_t> ids(input_len);
    for (auto& token : ids) {
      token = token_dist(rng);
    }
    batch.prompt_token_ids.push_back(
        torch::tensor(ids, torch::TensorOptions().dtype(torch::kInt32)));

    SamplingParams params;
    params.temperature = 0.6F;
    params.top_p = 1.0F;
    params.top_k = -1;
    params.ignore_eos = true;
    params.max_new_tokens = output_len;
    batch.sampling_params.push_back(params);
  }
  batch.prompt_tokens = static_cast<int64_t>(batch_size) * input_len;
  batch.requested_output_tokens = static_cast<int64_t>(batch_size) * output_len;
  return batch;
}

void run_offline_batch(benchmark::State& state) {
  const int batch_size = static_cast<int>(state.range(0));
  const int input_len = static_cast<int>(state.range(1));
  const int output_len = static_cast<int>(state.range(2));

  auto batch = make_batch_inputs(batch_size, input_len, output_len);
  auto& llm = get_llm();

  int64_t generated_tokens = 0;
  for (auto _ : state) {
    auto results = llm.generate(batch.prompt_token_ids, batch.sampling_params);
    for (const auto& result : results) {
      generated_tokens += static_cast<int64_t>(result.token_ids.size());
    }
    benchmark::DoNotOptimize(results);
  }

  state.counters["batch_size"] = batch_size;
  state.counters["input_len"] = input_len;
  state.counters["output_len"] = output_len;
  state.counters["prompt_tok/s"] =
      benchmark::Counter(static_cast<double>(batch.prompt_tokens) * state.iterations(),
                         benchmark::Counter::kIsRate);
  state.counters["gen_tok/s"] =
      benchmark::Counter(static_cast<double>(generated_tokens), benchmark::Counter::kIsRate);
  state.counters["req/s"] =
      benchmark::Counter(static_cast<double>(batch_size) * state.iterations(),
                         benchmark::Counter::kIsRate);
}

void register_benchmarks() {
  for (int batch_size : g_options.batch_sizes) {
    for (int input_len : g_options.input_lengths) {
      for (int output_len : g_options.output_lengths) {
        std::string name = "offline_generate/bs_" + std::to_string(batch_size) +
                           "/in_" + std::to_string(input_len) +
                           "/out_" + std::to_string(output_len);
    benchmark::RegisterBenchmark(name.c_str(), &run_offline_batch)
            ->Args({batch_size, input_len, output_len})
            ->Unit(benchmark::kMillisecond)
            ->UseRealTime();
      }
    }
  }
}

}  // namespace
}  // namespace sglang

int main(int argc, char** argv) {
  try {
    sglang::parse_custom_args(&argc, argv);
    if (sglang::g_show_help) {
      return 0;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
      return 1;
    }
    sglang::register_benchmarks();
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    sglang::g_llm.reset();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "offline benchmark failed: " << error.what() << std::endl;
    return 1;
  }
}
