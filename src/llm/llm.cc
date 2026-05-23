#include "sglang/llm/llm.h"

#include <stdexcept>

namespace sglang {

LLM::LLM(const ServerArgs& args) : args_(args) {
  const auto tokenizer_path = args_.model_path + "/tokenizer.json";
  tokenizer_pool_ =
      std::make_unique<TokenizerWorkerPool>(tokenizer_path, args_.num_tokenizer_threads);
  scheduler_runner_ = std::make_unique<SchedulerRunner>(
      args_.to_scheduler_config(),
      [this](std::vector<DetokenizeMsg> replies) {
        frontend_->handle_detokenize(std::move(replies));
      });
  frontend_ =
      std::make_unique<FrontendManager>(args_, *tokenizer_pool_, *scheduler_runner_);
  scheduler_runner_->start();
}

LLM::~LLM() {
  shutdown();
}

GenerateTextResult LLM::generate(const std::string& prompt,
                                 const SamplingParams& sampling_params) {
  const auto uid = frontend_->submit_text_request(std::string(prompt), sampling_params);
  return frontend_->wait_result(uid);
}

GenerateTextResult LLM::generate(const torch::Tensor& prompt_token_ids,
                                 const SamplingParams& sampling_params) {
  const auto uid = frontend_->submit_tokenized_request(prompt_token_ids, sampling_params);
  return frontend_->wait_result(uid);
}

std::vector<GenerateTextResult> LLM::generate(
    const std::vector<std::string>& prompts,
    const std::vector<SamplingParams>& sampling_params) {
  if (prompts.size() != sampling_params.size()) {
    throw std::invalid_argument("prompts and sampling_params must have the same size");
  }

  std::vector<uint64_t> uids;
  uids.reserve(prompts.size());
  for (size_t i = 0; i < prompts.size(); ++i) {
    uids.push_back(frontend_->submit_text_request(prompts[i], sampling_params[i]));
  }

  std::vector<GenerateTextResult> results;
  results.reserve(prompts.size());
  for (auto uid : uids) {
    results.push_back(frontend_->wait_result(uid));
  }
  return results;
}

std::vector<GenerateTextResult> LLM::generate(
    const std::vector<torch::Tensor>& prompt_token_ids,
    const std::vector<SamplingParams>& sampling_params) {
  if (prompt_token_ids.size() != sampling_params.size()) {
    throw std::invalid_argument("prompt_token_ids and sampling_params must have the same size");
  }

  std::vector<uint64_t> uids;
  uids.reserve(prompt_token_ids.size());
  for (size_t i = 0; i < prompt_token_ids.size(); ++i) {
    uids.push_back(
        frontend_->submit_tokenized_request(prompt_token_ids[i], sampling_params[i]));
  }

  std::vector<GenerateTextResult> results;
  results.reserve(prompt_token_ids.size());
  for (auto uid : uids) {
    results.push_back(frontend_->wait_result(uid));
  }
  return results;
}

void LLM::shutdown() {
  if (shutdown_) {
    return;
  }
  shutdown_ = true;
  if (scheduler_runner_) {
    scheduler_runner_->stop();
  }
  tokenizer_pool_.reset();
  frontend_.reset();
  scheduler_runner_.reset();
}

}  // namespace sglang
