#pragma once

#include <string>
#include <vector>

#include "sglang/core/sampling_params.h"
#include "sglang/server/args.h"
#include "sglang/server/frontend.h"

namespace sglang {

class LLM {
 public:
  explicit LLM(const ServerArgs& args);
  ~LLM();

  GenerateTextResult generate(const std::string& prompt, const SamplingParams& sampling_params);
  std::vector<GenerateTextResult> generate(
      const std::vector<std::string>& prompts,
      const std::vector<SamplingParams>& sampling_params);
  void shutdown();

 private:
  ServerArgs args_;
  std::unique_ptr<TokenizerWorkerPool> tokenizer_pool_;
  std::unique_ptr<SchedulerRunner> scheduler_runner_;
  std::unique_ptr<FrontendManager> frontend_;
  bool shutdown_ = false;
};

}  // namespace sglang
