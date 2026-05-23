#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iostream>
#include <optional>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "cinatra/coro_http_client.hpp"
#include "sglang/server/http_server.h"

namespace sglang {
namespace {

std::string find_qwen3_model_path() {
  if (const char* env_path = std::getenv("QWEN3_MODEL_PATH")) {
    return std::string(env_path);
  }

  const char* home = std::getenv("HOME");
  if (!home) {
    return "";
  }

  const std::string hf_dir =
      std::string(home) + "/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots";
  if (!std::filesystem::exists(hf_dir)) {
    return "";
  }

  for (const auto& entry : std::filesystem::directory_iterator(hf_dir)) {
    if (entry.is_directory()) {
      return entry.path().string();
    }
  }
  return "";
}

bool contains_paris(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text.find("paris") != std::string::npos || text.find("巴黎") != std::string::npos;
}

bool contains_berlin(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text.find("berlin") != std::string::npos || text.find("柏林") != std::string::npos;
}

bool is_cuda_oom(const std::exception& error) {
  const std::string message = error.what();
  return message.find("out of memory") != std::string::npos ||
         message.find("CUDA error: out of memory") != std::string::npos ||
         message.find("CUDA out of memory") != std::string::npos;
}

nlohmann::json parse_json_or_fail(const std::string& body) {
  try {
    return nlohmann::json::parse(body);
  } catch (const std::exception& e) {
    ADD_FAILURE() << "Failed to parse JSON body: " << e.what() << "\nRaw body:\n" << body;
    throw;
  }
}

bool is_expected_shutdown_exception(const std::exception_ptr& exception) {
  if (!exception) {
    return false;
  }
  try {
    std::rethrow_exception(exception);
  } catch (const std::runtime_error& error) {
    return std::string(error.what()).find("Operation aborted") != std::string::npos;
  } catch (...) {
    return false;
  }
}

struct OwnedHttpResponse {
  std::error_code net_err;
  int status = 0;
  std::string body;
};

OwnedHttpResponse post_json_owned(const std::string& url, const nlohmann::json& payload) {
  cinatra::coro_http_client client;
  auto response = client.post(url, payload.dump(), cinatra::req_content_type::json);
  return OwnedHttpResponse{
      .net_err = response.net_err,
      .status = response.status,
      .body = std::string(response.resp_body),
  };
}

class ScopedApiServer {
 public:
  explicit ScopedApiServer(ServerArgs args)
      : args_(std::move(args)), server_(std::make_unique<ApiServer>(args_)) {}

  ~ScopedApiServer() {
    stop();
  }

  void start() {
    thread_ = std::thread([this]() {
      try {
        server_->run();
      } catch (...) {
        exception_ = std::current_exception();
      }
    });
    wait_until_ready();
  }

  void stop() {
    if (!server_) {
      return;
    }
    server_->stop();
    if (thread_.joinable()) {
      thread_.join();
    }
    if (exception_ && !is_expected_shutdown_exception(exception_)) {
      std::rethrow_exception(exception_);
    }
    server_.reset();
  }

 private:
  void wait_until_ready() {
    cinatra::coro_http_client client;
    for (int i = 0; i < 100; ++i) {
      auto resp = client.get("http://" + args_.server_host + ":" +
                             std::to_string(args_.server_port) + "/v1/models");
      if (!resp.net_err && resp.status == 200) {
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    throw std::runtime_error("HTTP server did not become ready in time");
  }

  ServerArgs args_;
  std::unique_ptr<ApiServer> server_;
  std::thread thread_;
  std::exception_ptr exception_;
};

std::optional<ServerArgs> make_real_server_args(int port) {
  if (!torch::cuda::is_available()) {
    return std::nullopt;
  }

  const auto model_path = find_qwen3_model_path();
  if (model_path.empty()) {
    return std::nullopt;
  }

  ServerArgs args;
  args.model_path = model_path;
  args.dtype = torch::kBFloat16;
  args.server_host = "127.0.0.1";
  args.server_port = port;
  args.num_tokenizer_threads = 1;
  args.max_running_req = 2;
  args.memory_ratio = 0.2F;
  args.page_size = 1;
  args.max_extend_tokens = 64;
  return args;
}

std::optional<ServerArgs> make_dummy_server_args(int port, int concurrency) {
  auto maybe_args = make_real_server_args(port);
  if (!maybe_args.has_value()) {
    return std::nullopt;
  }

  maybe_args->num_tokenizer_threads = std::min(concurrency, 8);
  maybe_args->max_running_req = concurrency;
  maybe_args->memory_ratio = 0.01F;
  maybe_args->max_extend_tokens = std::max(128, concurrency * 8);
  maybe_args->num_pages_override = std::max(64, concurrency * 32);
  maybe_args->max_seq_len_override = 64;
  maybe_args->use_dummy_weight = true;
  return maybe_args;
}

TEST(HttpE2ETest, GenerateEndpointAnswersFranceCapitalQuestion) {
  auto maybe_args = make_real_server_args(/*port=*/18081);
  if (!maybe_args.has_value()) {
    GTEST_SKIP() << "CUDA or Qwen3 model path is unavailable for HTTP E2E test";
  }

  ScopedApiServer server(*maybe_args);
  server.start();

  nlohmann::json payload = {
      {"prompt",
       "Question: What is the capital of France?\nAnswer in one short sentence."},
      {"max_tokens", 24},
      {"ignore_eos", false},
      {"stream", false},
  };

  cinatra::coro_http_client client;
  auto response = client.post("http://127.0.0.1:18081/generate", payload.dump(),
                              cinatra::req_content_type::json);

  ASSERT_FALSE(response.net_err) << response.net_err.message();
  ASSERT_EQ(response.status, 200);

  const std::string body(response.resp_body);
  auto json_body = nlohmann::json::parse(body);
  ASSERT_TRUE(json_body.contains("text"));
  const auto text = json_body.at("text").get<std::string>();
  EXPECT_TRUE(contains_paris(text)) << "Expected Paris/巴黎 in: " << text;

  server.stop();
}

TEST(HttpE2ETest, StreamingGenerateEndpointEndsWithDoneAndMentionsParis) {
  auto maybe_args = make_real_server_args(/*port=*/18082);
  if (!maybe_args.has_value()) {
    GTEST_SKIP() << "CUDA or Qwen3 model path is unavailable for HTTP E2E test";
  }

  ScopedApiServer server(*maybe_args);
  server.start();

  nlohmann::json payload = {
      {"prompt",
       "Question: What is the capital of France?\nAnswer in one short sentence."},
      {"max_tokens", 24},
      {"ignore_eos", false},
      {"stream", true},
  };

  cinatra::coro_http_client client;
  auto response = client.post("http://127.0.0.1:18082/generate", payload.dump(),
                              cinatra::req_content_type::json);

  ASSERT_FALSE(response.net_err) << response.net_err.message();
  ASSERT_EQ(response.status, 200);

  const std::string body(response.resp_body);
  EXPECT_NE(body.find("data: [DONE]"), std::string::npos);
  EXPECT_TRUE(contains_paris(body)) << "Expected Paris/巴黎 in SSE body: " << body;

  server.stop();
}

TEST(HttpE2ETest, ChatCompletionsMessagesEndpointAnswersFranceCapitalQuestion) {
  auto maybe_args = make_real_server_args(/*port=*/18083);
  if (!maybe_args.has_value()) {
    GTEST_SKIP() << "CUDA or Qwen3 model path is unavailable for HTTP E2E test";
  }

  ScopedApiServer server(*maybe_args);
  server.start();

  nlohmann::json payload = {
      {"model", maybe_args->model_path},
      {"messages",
       nlohmann::json::array({
           nlohmann::json{{"role", "system"}, {"content", "You are a concise assistant."}},
           nlohmann::json{{"role", "user"},
                          {"content",
                           "What is the capital of France? Answer in one short sentence."}},
       })},
      {"max_tokens", 24},
      {"temperature", 0.0},
      {"top_k", -1},
      {"top_p", 1.0},
      {"stream", false},
  };

  cinatra::coro_http_client client;
  auto response =
      client.post("http://127.0.0.1:18083/v1/chat/completions", payload.dump(),
                  cinatra::req_content_type::json);

  ASSERT_FALSE(response.net_err) << response.net_err.message();
  ASSERT_EQ(response.status, 200);

  const std::string body(response.resp_body);
  auto json_body = nlohmann::json::parse(body);
  ASSERT_TRUE(json_body.contains("choices"));
  ASSERT_TRUE(json_body["choices"].is_array());
  ASSERT_FALSE(json_body["choices"].empty());
  ASSERT_TRUE(json_body["choices"][0].contains("message"));
  ASSERT_TRUE(json_body["choices"][0]["message"].contains("content"));
  const auto text = json_body["choices"][0]["message"]["content"].get<std::string>();
  EXPECT_TRUE(contains_paris(text)) << "Expected Paris/巴黎 in: " << text;

  server.stop();
}

TEST(HttpE2ETest, StreamingChatCompletionsMessagesEndsWithDoneAndMentionsParis) {
  auto maybe_args = make_real_server_args(/*port=*/18084);
  if (!maybe_args.has_value()) {
    GTEST_SKIP() << "CUDA or Qwen3 model path is unavailable for HTTP E2E test";
  }

  ScopedApiServer server(*maybe_args);
  server.start();

  nlohmann::json payload = {
      {"model", maybe_args->model_path},
      {"messages",
       nlohmann::json::array({
           nlohmann::json{{"role", "system"}, {"content", "You are a concise assistant."}},
           nlohmann::json{{"role", "user"},
                          {"content",
                           "What is the capital of France? Answer in one short sentence."}},
       })},
      {"max_tokens", 24},
      {"temperature", 0.0},
      {"top_k", -1},
      {"top_p", 1.0},
      {"stream", true},
  };

  cinatra::coro_http_client client;
  auto response =
      client.post("http://127.0.0.1:18084/v1/chat/completions", payload.dump(),
                  cinatra::req_content_type::json);

  ASSERT_FALSE(response.net_err) << response.net_err.message();
  ASSERT_EQ(response.status, 200);

  const std::string body(response.resp_body);
  EXPECT_NE(body.find("data: [DONE]"), std::string::npos);
  EXPECT_TRUE(contains_paris(body)) << "Expected Paris/巴黎 in SSE body: " << body;

  server.stop();
}

TEST(HttpE2ETest, ConcurrentGenerateRequestsReturnIndependentAnswers) {
  auto maybe_args = make_real_server_args(/*port=*/18085);
  if (!maybe_args.has_value()) {
    GTEST_SKIP() << "CUDA or Qwen3 model path is unavailable for HTTP E2E test";
  }

  maybe_args->max_running_req = 4;
  ScopedApiServer server(*maybe_args);
  server.start();

  auto france_future = std::async(std::launch::async, []() {
    nlohmann::json payload = {
        {"prompt",
         "Question: What is the capital of France?\nAnswer in one short sentence."},
        {"max_tokens", 24},
        {"ignore_eos", false},
        {"stream", false},
    };

    return post_json_owned("http://127.0.0.1:18085/generate", payload);
  });

  auto germany_future = std::async(std::launch::async, []() {
    nlohmann::json payload = {
        {"prompt",
         "Question: What is the capital of Germany?\nAnswer in one short sentence."},
        {"max_tokens", 24},
        {"ignore_eos", false},
        {"stream", false},
    };

    return post_json_owned("http://127.0.0.1:18085/generate", payload);
  });

  auto france_response = france_future.get();
  auto germany_response = germany_future.get();

  ASSERT_FALSE(france_response.net_err) << france_response.net_err.message();
  ASSERT_FALSE(germany_response.net_err) << germany_response.net_err.message();
  ASSERT_EQ(france_response.status, 200);
  ASSERT_EQ(germany_response.status, 200);

  const auto france_body = parse_json_or_fail(france_response.body);
  const auto germany_body = parse_json_or_fail(germany_response.body);
  const auto france_text = france_body.at("text").get<std::string>();
  const auto germany_text = germany_body.at("text").get<std::string>();

  EXPECT_TRUE(contains_paris(france_text)) << france_text;
  EXPECT_TRUE(contains_berlin(germany_text)) << germany_text;

  server.stop();
}

TEST(HttpE2ETest, ConcurrentChatCompletionHandlersReturnIndependentAnswers) {
  auto maybe_args = make_real_server_args(/*port=*/18086);
  if (!maybe_args.has_value()) {
    GTEST_SKIP() << "CUDA or Qwen3 model path is unavailable for HTTP E2E test";
  }

  maybe_args->max_running_req = 4;
  ApiServer server(*maybe_args);

  auto france_future = std::async(std::launch::async, [&]() {
    OpenAICompletionRequest request;
    request.model = maybe_args->model_path;
    request.messages = std::vector<ChatMessage>{
        ChatMessage{"system", "You are a concise assistant."},
        ChatMessage{"user", "What is the capital of France? Answer in one short sentence."},
    };
    request.max_tokens = 24;
    request.temperature = 0.0F;
    request.top_k = -1;
    request.top_p = 1.0F;
    request.stream = false;
    return server.handle_chat_completions(request);
  });

  auto germany_future = std::async(std::launch::async, [&]() {
    OpenAICompletionRequest request;
    request.model = maybe_args->model_path;
    request.messages = std::vector<ChatMessage>{
        ChatMessage{"system", "You are a concise assistant."},
        ChatMessage{"user", "What is the capital of Germany? Answer in one short sentence."},
    };
    request.max_tokens = 24;
    request.temperature = 0.0F;
    request.top_k = -1;
    request.top_p = 1.0F;
    request.stream = false;
    return server.handle_chat_completions(request);
  });

  auto france_response = france_future.get();
  auto germany_response = germany_future.get();

  const auto france_body = france_response;
  const auto germany_body = germany_response;
  const auto france_text =
      france_body["choices"][0]["message"]["content"].get<std::string>();
  const auto germany_text =
      germany_body["choices"][0]["message"]["content"].get<std::string>();

  EXPECT_TRUE(contains_paris(france_text)) << france_text;
  EXPECT_TRUE(contains_berlin(germany_text)) << germany_text;
}

TEST(HttpE2ETest, ConcurrentGenerateSweepUpToOom) {
  const std::vector<int> concurrency_levels = {8, 16, 64, 128};
  int max_completed_concurrency = 0;

  for (int concurrency : concurrency_levels) {
    SCOPED_TRACE("http_generate_concurrency=" + std::to_string(concurrency));
    auto maybe_args = make_dummy_server_args(/*port=*/18100 + concurrency, concurrency);
    if (!maybe_args.has_value()) {
      GTEST_SKIP() << "CUDA or Qwen3 tokenizer/model path is unavailable for HTTP E2E test";
    }

    try {
      ScopedApiServer server(*maybe_args);
      server.start();

      std::vector<std::future<OwnedHttpResponse>> futures;
      futures.reserve(concurrency);
      for (int i = 0; i < concurrency; ++i) {
        futures.push_back(std::async(std::launch::async, [port = maybe_args->server_port, i]() {
          nlohmann::json payload = {
              {"prompt", "Hello #" + std::to_string(i)},
              {"max_tokens", 2},
              {"ignore_eos", true},
              {"stream", false},
          };

          return post_json_owned("http://127.0.0.1:" + std::to_string(port) + "/generate",
                                 payload);
        }));
      }

      for (auto& future : futures) {
        auto response = future.get();
        ASSERT_FALSE(response.net_err) << response.net_err.message();
        ASSERT_EQ(response.status, 200);
        const auto body = parse_json_or_fail(response.body);
        EXPECT_TRUE(body.at("finished").get<bool>());
        EXPECT_GE(body.at("token_ids").size(), 1);
        EXPECT_LE(body.at("token_ids").size(), 2);
      }

      server.stop();
      max_completed_concurrency = concurrency;
      std::cout << "[http sweep] completed concurrency=" << concurrency << std::endl;
    } catch (const c10::Error& error) {
      if (is_cuda_oom(error)) {
        std::cout << "[http sweep] hit OOM at concurrency=" << concurrency << std::endl;
        break;
      }
      throw;
    } catch (const std::runtime_error& error) {
      if (is_cuda_oom(error)) {
        std::cout << "[http sweep] hit OOM at concurrency=" << concurrency << std::endl;
        break;
      }
      throw;
    }
  }

  EXPECT_GE(max_completed_concurrency, 8);
}

}  // namespace
}  // namespace sglang
