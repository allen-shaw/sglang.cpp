#include "sglang/server/http_server.h"

#include <chrono>
#include <ctime>
#include <sstream>
#include <stdexcept>
#include <string>

#include "cinatra/define.h"

namespace sglang {

namespace {

using json = nlohmann::json;

json make_error_response(std::string message) {
  return json{{"error", std::move(message)}};
}

std::string make_sse_event(std::string_view payload) {
  std::ostringstream stream;
  size_t line_start = 0;
  while (line_start <= payload.size()) {
    const auto next_newline = payload.find('\n', line_start);
    if (next_newline == std::string_view::npos) {
      stream << "data: " << payload.substr(line_start) << "\n";
      break;
    }
    stream << "data: " << payload.substr(line_start, next_newline - line_start) << "\n";
    line_start = next_newline + 1;
    if (line_start == payload.size()) {
      stream << "data: \n";
      break;
    }
  }
  stream << "\n";
  return stream.str();
}

json make_generate_response(const GenerateTextResult& result) {
  return json{
      {"id", result.uid},
      {"text", result.text},
      {"token_ids", result.token_ids},
      {"finished", result.finished},
  };
}

json make_chat_response(uint64_t uid, const std::string& text) {
  return json{
      {"id", "cmpl-" + std::to_string(uid)},
      {"object", "text_completion"},
      {"choices",
       json::array({json{{"index", 0},
                         {"message", json{{"role", "assistant"}, {"content", text}}},
                         {"finish_reason", "stop"}}})},
  };
}

json parse_request_json(cinatra::coro_http_request& request) {
  auto body = request.get_body();
  if (body.empty()) {
    return json::object();
  }
  return json::parse(body.begin(), body.end());
}

}  // namespace

void from_json(const json& json_obj, GenerateHttpRequest& request) {
  json_obj.at("prompt").get_to(request.prompt);
  if (json_obj.contains("max_tokens")) {
    json_obj.at("max_tokens").get_to(request.max_tokens);
  }
  if (json_obj.contains("ignore_eos")) {
    json_obj.at("ignore_eos").get_to(request.ignore_eos);
  }
  if (json_obj.contains("stream")) {
    json_obj.at("stream").get_to(request.stream);
  }
}

void from_json(const json& json_obj, ChatMessage& message) {
  json_obj.at("role").get_to(message.role);
  json_obj.at("content").get_to(message.content);
}

void from_json(const json& json_obj, OpenAICompletionRequest& request) {
  if (json_obj.contains("model")) {
    json_obj.at("model").get_to(request.model);
  }
  if (json_obj.contains("prompt") && !json_obj.at("prompt").is_null()) {
    request.prompt = json_obj.at("prompt").get<std::string>();
  }
  if (json_obj.contains("messages") && !json_obj.at("messages").is_null()) {
    request.messages = json_obj.at("messages").get<std::vector<ChatMessage>>();
  }
  if (json_obj.contains("max_tokens")) {
    json_obj.at("max_tokens").get_to(request.max_tokens);
  }
  if (json_obj.contains("temperature")) {
    json_obj.at("temperature").get_to(request.temperature);
  }
  if (json_obj.contains("top_k")) {
    json_obj.at("top_k").get_to(request.top_k);
  }
  if (json_obj.contains("top_p")) {
    json_obj.at("top_p").get_to(request.top_p);
  }
  if (json_obj.contains("n")) {
    json_obj.at("n").get_to(request.n);
  }
  if (json_obj.contains("stream")) {
    json_obj.at("stream").get_to(request.stream);
  }
  if (json_obj.contains("stop") && !json_obj.at("stop").is_null()) {
    request.stop = json_obj.at("stop").get<std::vector<std::string>>();
  }
  if (json_obj.contains("presence_penalty")) {
    json_obj.at("presence_penalty").get_to(request.presence_penalty);
  }
  if (json_obj.contains("frequency_penalty")) {
    json_obj.at("frequency_penalty").get_to(request.frequency_penalty);
  }
  if (json_obj.contains("ignore_eos")) {
    json_obj.at("ignore_eos").get_to(request.ignore_eos);
  }
}

void to_json(json& json_obj, const ModelCard& card) {
  json_obj = json{{"id", card.id},
                  {"object", card.object},
                  {"created", card.created},
                  {"owned_by", card.owned_by},
                  {"root", card.root}};
}

void to_json(json& json_obj, const ModelList& list) {
  json_obj = json{{"object", list.object}, {"data", list.data}};
}

ApiServer::ApiServer(const ServerArgs& args)
    : args_(args),
      tokenizer_pool_(std::make_unique<TokenizerWorkerPool>(
          args.model_path + "/tokenizer.json", args.num_tokenizer_threads)),
      scheduler_runner_(std::make_unique<SchedulerRunner>(
          args.to_scheduler_config(),
          [this](std::vector<DetokenizeMsg> replies) {
            frontend_->handle_detokenize(std::move(replies));
          })),
      frontend_(std::make_unique<FrontendManager>(args_, *tokenizer_pool_, *scheduler_runner_)),
      server_(std::max(1U, std::thread::hardware_concurrency()),
              static_cast<unsigned short>(args.server_port), args.server_host) {
  scheduler_runner_->start();
  register_routes();
}

ApiServer::~ApiServer() {
  stop();
}

void ApiServer::run() {
  auto error = server_.sync_start();
  if (error) {
    throw std::runtime_error("Failed to start HTTP server: " + error.message());
  }
  wait_until_stopped();
}

void ApiServer::stop() {
  notify_stop();
  server_.stop();
  if (scheduler_runner_) {
    scheduler_runner_->stop();
  }
  tokenizer_pool_.reset();
  frontend_.reset();
  scheduler_runner_.reset();
}

GenerateTextResult ApiServer::handle_generate(const GenerateHttpRequest& request) {
  const auto uid = frontend_->submit_text_request(
      request.prompt, make_sampling_params(request));
  return frontend_->wait_result(uid);
}

std::string ApiServer::handle_generate_stream(const GenerateHttpRequest& request) {
  const auto uid = frontend_->submit_text_request(
      request.prompt, make_sampling_params(request));
  return collect_generate_sse(uid);
}

json ApiServer::handle_chat_completions(const OpenAICompletionRequest& request) {
  const auto sampling_params = make_sampling_params(request);
  const auto uid = submit_chat_request(request, sampling_params);
  const auto result = frontend_->wait_result(uid);
  return make_chat_response(result.uid, result.text);
}

std::string ApiServer::handle_chat_stream(const OpenAICompletionRequest& request) {
  const auto uid = submit_chat_request(request, make_sampling_params(request));
  return collect_chat_sse(uid);
}

ModelList ApiServer::handle_models() const {
  ModelList list;
  list.data.push_back(ModelCard{
      .id = args_.model_path,
      .object = "model",
      .created = static_cast<int64_t>(std::time(nullptr)),
      .owned_by = "sglang.cpp",
      .root = args_.model_path,
  });
  return list;
}

async_simple::coro::Lazy<void> ApiServer::stream_generate_http(
    cinatra::coro_http_request& request, cinatra::coro_http_response& response) {
  uint64_t uid = 0;
  bool submitted = false;
  try {
    const auto json_obj = parse_request_json(request);
    const auto parsed = json_obj.get<GenerateHttpRequest>();
    if (!parsed.stream) {
      response.add_header("Content-Type", "application/json");
      response.set_status_and_content(cinatra::status_type::ok,
                                      make_generate_response(handle_generate(parsed)).dump());
      co_return;
    }
    uid = frontend_->submit_text_request(parsed.prompt, make_sampling_params(parsed));
    submitted = true;

    response.add_header("Content-Type", "text/event-stream");
    response.add_header("Cache-Control", "no-cache");
    response.add_header("X-Accel-Buffering", "no");
    response.set_format_type(cinatra::format_type::chunked);

    auto* connection = response.get_conn();
    if (!co_await connection->begin_chunked()) {
      frontend_->abort(uid);
      (void)frontend_->wait_result(uid);
      co_return;
    }

    GenerateResponse chunk;
    while (frontend_->wait_next_chunk(uid, chunk)) {
      if (!co_await connection->write_chunked(make_sse_event(chunk.incremental_output))) {
        frontend_->abort(uid);
        break;
      }
      if (chunk.finished) {
        break;
      }
    }

    (void)frontend_->wait_result(uid);
    (void)co_await connection->write_chunked(make_sse_event("[DONE]"));
    (void)co_await connection->end_chunked();
  } catch (const std::exception& e) {
    if (submitted) {
      frontend_->abort(uid);
      (void)frontend_->wait_result(uid);
    }
    response.add_header("Content-Type", "application/json");
    response.set_status_and_content(cinatra::status_type::bad_request,
                                    make_error_response(e.what()).dump());
  }
}

async_simple::coro::Lazy<void> ApiServer::stream_chat_http(
    cinatra::coro_http_request& request, cinatra::coro_http_response& response) {
  uint64_t uid = 0;
  bool submitted = false;
  try {
    const auto json_obj = parse_request_json(request);
    const auto parsed = json_obj.get<OpenAICompletionRequest>();
    if (!parsed.stream) {
      response.add_header("Content-Type", "application/json");
      response.set_status_and_content(cinatra::status_type::ok,
                                      handle_chat_completions(parsed).dump());
      co_return;
    }
    uid = submit_chat_request(parsed, make_sampling_params(parsed));
    submitted = true;

    response.add_header("Content-Type", "text/event-stream");
    response.add_header("Cache-Control", "no-cache");
    response.add_header("X-Accel-Buffering", "no");
    response.set_format_type(cinatra::format_type::chunked);

    auto* connection = response.get_conn();
    if (!co_await connection->begin_chunked()) {
      frontend_->abort(uid);
      (void)frontend_->wait_result(uid);
      co_return;
    }

    bool first_chunk = true;
    GenerateResponse chunk;
    while (frontend_->wait_next_chunk(uid, chunk)) {
      json delta = json::object();
      if (first_chunk) {
        delta["role"] = "assistant";
        first_chunk = false;
      }
      if (!chunk.incremental_output.empty()) {
        delta["content"] = chunk.incremental_output;
      }
      json payload = {{"id", "cmpl-" + std::to_string(uid)},
                      {"object", "text_completion.chunk"},
                      {"choices", json::array({json{{"delta", delta},
                                                   {"index", 0},
                                                   {"finish_reason", nullptr}}})}};
      if (!co_await connection->write_chunked(make_sse_event(payload.dump()))) {
        frontend_->abort(uid);
        break;
      }
      if (chunk.finished) {
        break;
      }
    }

    (void)frontend_->wait_result(uid);

    json end_payload = {{"id", "cmpl-" + std::to_string(uid)},
                        {"object", "text_completion.chunk"},
                        {"choices", json::array({json{{"delta", json::object()},
                                                     {"index", 0},
                                                     {"finish_reason", "stop"}}})}};
    (void)co_await connection->write_chunked(make_sse_event(end_payload.dump()));
    (void)co_await connection->write_chunked(make_sse_event("[DONE]"));
    (void)co_await connection->end_chunked();
  } catch (const std::exception& e) {
    if (submitted) {
      frontend_->abort(uid);
      (void)frontend_->wait_result(uid);
    }
    response.add_header("Content-Type", "application/json");
    response.set_status_and_content(cinatra::status_type::bad_request,
                                    make_error_response(e.what()).dump());
  }
}

SamplingParams ApiServer::make_sampling_params(const GenerateHttpRequest& request) const {
  SamplingParams params;
  params.ignore_eos = request.ignore_eos;
  params.max_new_tokens = request.max_tokens;
  params.temperature = 0.0F;
  params.top_p = 1.0F;
  params.top_k = -1;
  return params;
}

SamplingParams ApiServer::make_sampling_params(const OpenAICompletionRequest& request) const {
  SamplingParams params;
  params.n = request.n;
  params.stop = request.stop;
  params.ignore_eos = request.ignore_eos;
  params.max_new_tokens = request.max_tokens;
  params.temperature = request.temperature;
  params.top_k = request.top_k;
  params.top_p = request.top_p;
  params.presence_penalty = request.presence_penalty;
  params.frequency_penalty = request.frequency_penalty;
  return params;
}

uint64_t ApiServer::submit_chat_request(const OpenAICompletionRequest& request,
                                        const SamplingParams& sampling_params) {
  if (request.messages.has_value()) {
    return frontend_->submit_text_request(*request.messages, sampling_params);
  }
  if (request.prompt.has_value()) {
    return frontend_->submit_text_request(*request.prompt, sampling_params);
  }
  throw std::invalid_argument("Either prompt or messages must be provided");
}

std::string ApiServer::collect_generate_sse(uint64_t uid) {
  std::ostringstream stream;
  GenerateResponse response;
  while (frontend_->wait_next_chunk(uid, response)) {
    stream << "data: " << response.incremental_output << "\n";
    if (response.finished) {
      break;
    }
  }
  stream << "data: [DONE]\n";
  (void)frontend_->wait_result(uid);
  return stream.str();
}

std::string ApiServer::collect_chat_sse(uint64_t uid) {
  std::ostringstream stream;
  bool first_chunk = true;
  GenerateResponse response;
  while (frontend_->wait_next_chunk(uid, response)) {
    json delta = json::object();
    if (first_chunk) {
      delta["role"] = "assistant";
      first_chunk = false;
    }
    if (!response.incremental_output.empty()) {
      delta["content"] = response.incremental_output;
    }
    json chunk = {{"id", "cmpl-" + std::to_string(uid)},
                  {"object", "text_completion.chunk"},
                  {"choices", json::array({json{{"delta", delta},
                                               {"index", 0},
                                               {"finish_reason", nullptr}}})}};
    stream << "data: " << chunk.dump() << "\n\n";
    if (response.finished) {
      break;
    }
  }

  json end_chunk = {{"id", "cmpl-" + std::to_string(uid)},
                    {"object", "text_completion.chunk"},
                    {"choices", json::array({json{{"delta", json::object()},
                                                 {"index", 0},
                                                 {"finish_reason", "stop"}}})}};
  stream << "data: " << end_chunk.dump() << "\n\n";
  stream << "data: [DONE]\n\n";
  (void)frontend_->wait_result(uid);
  return stream.str();
}

void ApiServer::register_routes() {
  server_.set_http_handler<cinatra::POST>("/generate", &ApiServer::stream_generate_http, *this);

  server_.set_http_handler<cinatra::GET, cinatra::POST, cinatra::HEAD, cinatra::OPTIONS>(
      "/v1",
      [](cinatra::coro_http_request&, cinatra::coro_http_response& response) {
        response.add_header("Content-Type", "application/json");
        response.set_status_and_content(cinatra::status_type::ok, json{{"status", "ok"}}.dump());
      });

  server_.set_http_handler<cinatra::POST>("/v1/chat/completions",
                                          &ApiServer::stream_chat_http, *this);

  server_.set_http_handler<cinatra::GET>(
      "/v1/models",
      [this](cinatra::coro_http_request&, cinatra::coro_http_response& response) {
        response.add_header("Content-Type", "application/json");
        response.set_status_and_content(cinatra::status_type::ok,
                                        json(handle_models()).dump());
      });
}

void ApiServer::wait_until_stopped() {
  std::unique_lock<std::mutex> lock(stop_mutex_);
  stop_cv_.wait(lock, [this]() { return stopped_; });
}

void ApiServer::notify_stop() {
  {
    std::lock_guard<std::mutex> lock(stop_mutex_);
    if (stopped_) {
      return;
    }
    stopped_ = true;
  }
  stop_cv_.notify_all();
}

}  // namespace sglang
