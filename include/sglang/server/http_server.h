#pragma once

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <async_simple/coro/Lazy.h>
#include <nlohmann/json.hpp>

#include "cinatra/coro_http_server.hpp"
#include "sglang/core/sampling_params.h"
#include "sglang/message/tokenizer_msg.h"
#include "sglang/server/args.h"
#include "sglang/server/frontend.h"

namespace sglang {

struct GenerateHttpRequest {
  std::string prompt;
  int max_tokens = 16;
  bool ignore_eos = false;
  bool stream = false;
};

struct OpenAICompletionRequest {
  std::string model;
  std::optional<std::string> prompt;
  std::optional<std::vector<ChatMessage>> messages;
  int max_tokens = 16;
  float temperature = 1.0F;
  int top_k = -1;
  float top_p = 1.0F;
  int n = 1;
  bool stream = false;
  std::vector<std::string> stop;
  float presence_penalty = 0.0F;
  float frequency_penalty = 0.0F;
  bool ignore_eos = false;
};

struct ModelCard {
  std::string id;
  std::string object = "model";
  int64_t created = 0;
  std::string owned_by = "sglang.cpp";
  std::string root;
};

struct ModelList {
  std::string object = "list";
  std::vector<ModelCard> data;
};

void from_json(const nlohmann::json& json, GenerateHttpRequest& request);
void from_json(const nlohmann::json& json, ChatMessage& message);
void from_json(const nlohmann::json& json, OpenAICompletionRequest& request);
void to_json(nlohmann::json& json, const ModelCard& card);
void to_json(nlohmann::json& json, const ModelList& list);

class ApiServer {
 public:
  explicit ApiServer(const ServerArgs& args);
  ~ApiServer();

  void run();
  void stop();

  GenerateTextResult handle_generate(const GenerateHttpRequest& request);
  std::string handle_generate_stream(const GenerateHttpRequest& request);
  nlohmann::json handle_chat_completions(const OpenAICompletionRequest& request);
  std::string handle_chat_stream(const OpenAICompletionRequest& request);
  ModelList handle_models() const;

 private:
  async_simple::coro::Lazy<void> stream_generate_http(
      cinatra::coro_http_request& request, cinatra::coro_http_response& response);
  async_simple::coro::Lazy<void> stream_chat_http(
      cinatra::coro_http_request& request, cinatra::coro_http_response& response);
  SamplingParams make_sampling_params(const GenerateHttpRequest& request) const;
  SamplingParams make_sampling_params(const OpenAICompletionRequest& request) const;
  uint64_t submit_chat_request(const OpenAICompletionRequest& request,
                               const SamplingParams& sampling_params);
  std::string collect_generate_sse(uint64_t uid);
  std::string collect_chat_sse(uint64_t uid);
  void register_routes();
  void wait_until_stopped();
  void notify_stop();

  ServerArgs args_;
  std::unique_ptr<TokenizerWorkerPool> tokenizer_pool_;
  std::unique_ptr<SchedulerRunner> scheduler_runner_;
  std::unique_ptr<FrontendManager> frontend_;
  cinatra::coro_http_server server_;
  std::mutex stop_mutex_;
  std::condition_variable stop_cv_;
  bool stopped_ = false;
};

}  // namespace sglang
