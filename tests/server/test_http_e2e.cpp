#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
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

}  // namespace
}  // namespace sglang
