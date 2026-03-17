#pragma once

#include <string>
#include <vector>

namespace sglang {

struct SamplingParams {
  int n = 1;
  int best_of = 1;
  std::vector<std::string> stop;
  std::vector<int> stop_token_ids;
  bool ignore_eos = false;
  int max_new_tokens = 16;
  int min_new_tokens = 0;
  
  float temperature = 1.0F;
  float top_p = 1.0F;
  int top_k = -1;
  float min_p = 0.0F;
  float presence_penalty = 0.0F;
  float frequency_penalty = 0.0F;
  float repetition_penalty = 1.0F;
  
  bool use_beam_search = false;
  bool skip_special_tokens = true;
  bool spaces_between_special_tokens = true;

  // Helper function to check if sampling params indicate greedy decoding
  bool is_greedy() const;
};

} // namespace sglang
