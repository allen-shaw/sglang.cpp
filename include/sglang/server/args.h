#pragma once

#include <optional>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "sglang/scheduler/config.h"

namespace sglang {

struct ServerArgs {
  std::string model_path;
  torch::Dtype dtype = torch::kBFloat16;
  std::string server_host = "127.0.0.1";
  int server_port = 1919;
  int num_tokenizer_threads = 0;
  bool silent_output = false;
  int max_running_req = SchedulerConfig{}.max_running_req;
  int max_extend_tokens = SchedulerConfig{}.max_extend_tokens;
  float memory_ratio = SchedulerConfig{}.memory_ratio;
  int page_size = SchedulerConfig{}.page_size;
  std::optional<int> num_pages_override;
  std::optional<int> max_seq_len_override;
  std::optional<int> cuda_graph_max_batch_size;
  std::vector<int> cuda_graph_batch_sizes;
  int cuda_graph_capture_max_seq_len = SchedulerConfig{}.cuda_graph_capture_max_seq_len;
  bool use_dummy_weight = false;
  bool enable_cuda_graph = false;
  bool shell_mode = false;
  std::string cache_type = SchedulerConfig{}.cache_type;
  bool enable_overlap_scheduling = SchedulerConfig{}.enable_overlap_scheduling;

  bool share_tokenizer() const { return num_tokenizer_threads == 0; }
  SchedulerConfig to_scheduler_config() const;
};

class ServerArgsParser {
 public:
  static ServerArgs parse(int argc, char** argv);
  static ServerArgs parse(const std::vector<std::string>& args);
};

}  // namespace sglang
