#include "sglang/server/args.h"

#include <cstdlib>
#include <stdexcept>
#include <unordered_map>

namespace sglang {

namespace {

torch::Dtype parse_dtype(const std::string& value) {
  static const std::unordered_map<std::string, torch::Dtype> kMap = {
      {"float16", torch::kFloat16},
      {"fp16", torch::kFloat16},
      {"bfloat16", torch::kBFloat16},
      {"bf16", torch::kBFloat16},
      {"float32", torch::kFloat32},
      {"fp32", torch::kFloat32},
  };
  auto it = kMap.find(value);
  if (it == kMap.end()) {
    throw std::invalid_argument("Unsupported dtype: " + value);
  }
  return it->second;
}

std::string require_value(const std::vector<std::string>& args, size_t* index) {
  if (*index + 1 >= args.size()) {
    throw std::invalid_argument("Missing value for argument: " + args[*index]);
  }
  ++(*index);
  return args[*index];
}

int parse_int(const std::string& value) {
  return std::stoi(value);
}

float parse_float(const std::string& value) {
  return std::stof(value);
}

std::vector<int> parse_int_list(const std::string& value) {
  std::vector<int> result;
  size_t start = 0;
  while (start <= value.size()) {
    const auto end = value.find(',', start);
    const auto token =
        value.substr(start, end == std::string::npos ? std::string::npos : end - start);
    if (!token.empty()) {
      result.push_back(parse_int(token));
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return result;
}

std::string expand_home(std::string path) {
  if (!path.empty() && path[0] == '~') {
    if (const char* home = std::getenv("HOME")) {
      return std::string(home) + path.substr(1);
    }
  }
  return path;
}

}  // namespace

SchedulerConfig ServerArgs::to_scheduler_config() const {
  SchedulerConfig config;
  config.model_path = model_path;
  config.dtype = dtype;
  config.max_running_req = max_running_req;
  config.page_size = page_size;
  config.memory_ratio = memory_ratio;
  config.use_dummy_weight = use_dummy_weight;
  config.enable_cuda_graph = enable_cuda_graph;
  config.cuda_graph_max_batch_size = cuda_graph_max_batch_size;
  config.cuda_graph_batch_sizes = cuda_graph_batch_sizes;
  config.cuda_graph_capture_max_seq_len = cuda_graph_capture_max_seq_len;
  config.max_seq_len_override = max_seq_len_override;
  config.num_pages_override = num_pages_override;
  config.max_extend_tokens = max_extend_tokens;
  config.cache_type = cache_type;
  config.enable_overlap_scheduling = enable_overlap_scheduling;
  return config;
}

ServerArgs ServerArgsParser::parse(int argc, char** argv) {
  std::vector<std::string> args;
  args.reserve(static_cast<size_t>(argc > 0 ? argc - 1 : 0));
  for (int i = 1; i < argc; ++i) {
    args.emplace_back(argv[i]);
  }
  return parse(args);
}

ServerArgs ServerArgsParser::parse(const std::vector<std::string>& args) {
  ServerArgs result;
  for (size_t i = 0; i < args.size(); ++i) {
    const auto& arg = args[i];
    if (arg == "--model-path" || arg == "--model") {
      result.model_path = expand_home(require_value(args, &i));
    } else if (arg == "--dtype") {
      const auto value = require_value(args, &i);
      if (value != "auto") {
        result.dtype = parse_dtype(value);
      }
    } else if (arg == "--host") {
      result.server_host = require_value(args, &i);
    } else if (arg == "--port") {
      result.server_port = parse_int(require_value(args, &i));
    } else if (arg == "--num-tokenizer-threads" || arg == "--num-tokenizer" ||
               arg == "--tokenizer-count") {
      result.num_tokenizer_threads = parse_int(require_value(args, &i));
    } else if (arg == "--max-running-requests") {
      result.max_running_req = parse_int(require_value(args, &i));
    } else if (arg == "--max-prefill-length" || arg == "--max-extend-length") {
      result.max_extend_tokens = parse_int(require_value(args, &i));
    } else if (arg == "--memory-ratio") {
      result.memory_ratio = parse_float(require_value(args, &i));
    } else if (arg == "--page-size") {
      result.page_size = parse_int(require_value(args, &i));
    } else if (arg == "--num-pages") {
      result.num_pages_override = parse_int(require_value(args, &i));
    } else if (arg == "--max-seq-len-override") {
      result.max_seq_len_override = parse_int(require_value(args, &i));
    } else if (arg == "--dummy-weight") {
      result.use_dummy_weight = true;
    } else if (arg == "--disable-graph" || arg == "--disable-cuda-graph") {
      result.enable_cuda_graph = false;
    } else if (arg == "--graph" || arg == "--cuda-graph-max-bs") {
      result.enable_cuda_graph = true;
      result.cuda_graph_max_batch_size = parse_int(require_value(args, &i));
    } else if (arg == "--cuda-graph-batch-sizes") {
      result.enable_cuda_graph = true;
      result.cuda_graph_batch_sizes = parse_int_list(require_value(args, &i));
    } else if (arg == "--cuda-graph-capture-max-seq-len") {
      result.cuda_graph_capture_max_seq_len = parse_int(require_value(args, &i));
    } else if (arg == "--cache-type") {
      result.cache_type = require_value(args, &i);
    } else if (arg == "--shell-mode") {
      result.shell_mode = true;
    } else if (arg == "--silent-output") {
      result.silent_output = true;
    } else if (arg == "--enable-overlap-scheduling") {
      result.enable_overlap_scheduling = true;
    } else if (arg == "--disable-overlap-scheduling") {
      result.enable_overlap_scheduling = false;
    } else {
      throw std::invalid_argument("Unknown argument: " + arg);
    }
  }

  if (result.model_path.empty()) {
    throw std::invalid_argument("--model-path is required");
  }

  if (result.shell_mode) {
    result.max_running_req = 1;
    result.silent_output = true;
  }

  return result;
}

}  // namespace sglang
