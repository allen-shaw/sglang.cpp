#include "sglang/server/frontend.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace sglang {

TokenizerWorkerPool::TokenizerWorkerPool(const std::string& tokenizer_json_path,
                                         int num_encode_threads)
    : tokenize_manager_(tokenizer_json_path),
      detokenize_manager_(tokenizer_json_path),
      num_encode_threads_(std::max(1, num_encode_threads)) {
  encode_threads_.reserve(num_encode_threads_);
  for (int i = 0; i < num_encode_threads_; ++i) {
    encode_threads_.emplace_back([this]() { encode_loop(); });
  }
  decode_thread_ = std::thread([this]() { decode_loop(); });
}

TokenizerWorkerPool::~TokenizerWorkerPool() {
  {
    std::lock_guard<std::mutex> encode_lock(encode_mutex_);
    std::lock_guard<std::mutex> decode_lock(decode_mutex_);
    stop_ = true;
  }
  encode_cv_.notify_all();
  decode_cv_.notify_all();
  for (auto& thread : encode_threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  if (decode_thread_.joinable()) {
    decode_thread_.join();
  }
}

std::future<torch::Tensor> TokenizerWorkerPool::tokenize_async(
    TokenizeInput input, SamplingParams sampling_params) {
  return enqueue_encode(
      [this, input = std::move(input), sampling_params]() mutable -> torch::Tensor {
        TokenizeMsg msg;
        msg.uid = 0;
        msg.text = std::move(input);
        msg.sampling_params = sampling_params;
        auto result = tokenize_manager_.tokenize({msg});
        return std::move(result.front());
      });
}

std::future<std::vector<std::string>> TokenizerWorkerPool::detokenize_async(
    std::vector<DetokenizeMsg> msgs) {
  return enqueue_decode([this, msgs = std::move(msgs)]() mutable {
    return detokenize_manager_.detokenize(msgs);
  });
}

void TokenizerWorkerPool::encode_loop() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(encode_mutex_);
      encode_cv_.wait(lock, [this]() { return stop_ || !encode_tasks_.empty(); });
      if (stop_ && encode_tasks_.empty()) {
        return;
      }
      task = std::move(encode_tasks_.front());
      encode_tasks_.pop();
    }
    task();
  }
}

void TokenizerWorkerPool::decode_loop() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(decode_mutex_);
      decode_cv_.wait(lock, [this]() { return stop_ || !decode_tasks_.empty(); });
      if (stop_ && decode_tasks_.empty()) {
        return;
      }
      task = std::move(decode_tasks_.front());
      decode_tasks_.pop();
    }
    task();
  }
}

SchedulerRunner::SchedulerRunner(const SchedulerConfig& config,
                                 ReplyCallback reply_callback)
    : scheduler_(config),
      reply_callback_(std::move(reply_callback)) {}

SchedulerRunner::~SchedulerRunner() {
  stop();
}

void SchedulerRunner::start() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (running_) {
    return;
  }
  stop_requested_ = false;
  worker_ = std::thread([this]() { run_loop(); });
  running_ = true;
}

void SchedulerRunner::stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) {
      return;
    }
    stop_requested_ = true;
  }
  cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
  running_ = false;
}

void SchedulerRunner::submit(GenerateRequest request) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    commands_.push(Command{CommandType::kSubmit, std::move(request), 0});
  }
  cv_.notify_one();
}

void SchedulerRunner::abort(uint64_t uid) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    commands_.push(Command{CommandType::kAbort, GenerateRequest{}, uid});
  }
  cv_.notify_one();
}

void SchedulerRunner::run_loop() {
  while (true) {
    std::queue<Command> local_commands;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (!stop_requested_ && commands_.empty() && !scheduler_.has_work()) {
        cv_.wait(lock, [this]() {
          return stop_requested_ || !commands_.empty() || scheduler_.has_work();
        });
      }
      if (stop_requested_ && commands_.empty() && !scheduler_.has_work()) {
        return;
      }
      std::swap(local_commands, commands_);
    }

    while (!local_commands.empty()) {
      auto command = std::move(local_commands.front());
      local_commands.pop();
      if (command.type == CommandType::kSubmit) {
        scheduler_.submit(command.request);
      } else {
        scheduler_.abort(command.uid);
      }
    }

    if (scheduler_.has_work()) {
      auto replies = scheduler_.step();
      if (!replies.empty()) {
        reply_callback_(std::move(replies));
      }
    }
  }
}

FrontendManager::FrontendManager(const ServerArgs& args,
                                 TokenizerWorkerPool& tokenizer_pool,
                                 SchedulerRunner& scheduler_runner)
    : args_(args),
      tokenizer_pool_(tokenizer_pool),
      scheduler_runner_(scheduler_runner) {}

uint64_t FrontendManager::new_request() {
  const uint64_t uid = next_uid_.fetch_add(1);
  std::lock_guard<std::mutex> lock(contexts_mutex_);
  contexts_.emplace(uid, std::make_shared<RequestContext>(uid));
  return uid;
}

uint64_t FrontendManager::submit_tokenized_request(torch::Tensor input_ids,
                                                   SamplingParams sampling_params) {
  const uint64_t uid = new_request();
  auto context = find_context(uid);
  if (!context) {
    return uid;
  }

  try {
    GenerateRequest request;
    request.uid = uid;
    request.input_ids = input_ids.to(torch::kCPU).to(torch::kInt32).contiguous();
    request.sampling_params = sampling_params;
    scheduler_runner_.submit(std::move(request));
  } catch (...) {
    std::lock_guard<std::mutex> lock(context->mutex);
    context->finished = true;
    context->cv.notify_all();
  }

  return uid;
}

uint64_t FrontendManager::submit_text_request(TokenizeInput input,
                                              SamplingParams sampling_params) {
  const uint64_t uid = new_request();
  auto future = tokenizer_pool_.tokenize_async(std::move(input), sampling_params);
  std::thread([this, uid, sampling_params, future = std::move(future)]() mutable {
    try {
      auto input_ids = future.get();
      auto context = find_context(uid);
      if (!context) {
        return;
      }
      {
        std::lock_guard<std::mutex> lock(context->mutex);
        if (context->aborted) {
          return;
        }
      }
      GenerateRequest request;
      request.uid = uid;
      request.input_ids = input_ids.to(torch::kCPU).to(torch::kInt32).contiguous();
      request.sampling_params = sampling_params;
      scheduler_runner_.submit(std::move(request));
    } catch (...) {
      auto context = find_context(uid);
      if (context) {
        std::lock_guard<std::mutex> lock(context->mutex);
        context->finished = true;
        context->cv.notify_all();
      }
    }
  }).detach();
  return uid;
}

void FrontendManager::abort(uint64_t uid) {
  auto context = find_context(uid);
  if (!context) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(context->mutex);
    context->aborted = true;
    context->finished = true;
  }
  context->cv.notify_all();
  scheduler_runner_.abort(uid);
}

GenerateTextResult FrontendManager::wait_result(uint64_t uid) {
  auto context = get_or_create_context(uid);
  std::unique_lock<std::mutex> lock(context->mutex);
  context->cv.wait(lock, [&]() { return context->finished; });
  GenerateTextResult result;
  result.uid = uid;
  result.text = context->text;
  result.token_ids = context->token_ids;
  result.finished = context->finished;
  lock.unlock();

  std::lock_guard<std::mutex> map_lock(contexts_mutex_);
  contexts_.erase(uid);
  return result;
}

bool FrontendManager::wait_next_chunk(uint64_t uid, GenerateResponse& response) {
  auto context = get_or_create_context(uid);
  std::unique_lock<std::mutex> lock(context->mutex);
  context->cv.wait(lock, [&]() {
    return !context->pending_chunks.empty() || context->finished;
  });
  if (!context->pending_chunks.empty()) {
    response = std::move(context->pending_chunks.front());
    context->pending_chunks.pop_front();
    return true;
  }
  return false;
}

void FrontendManager::handle_detokenize(std::vector<DetokenizeMsg> msgs) {
  auto future = tokenizer_pool_.detokenize_async(msgs);
  std::thread([this, msgs = std::move(msgs), future = std::move(future)]() mutable {
    auto chunks = future.get();
    for (size_t i = 0; i < msgs.size(); ++i) {
      auto context = find_context(msgs[i].uid);
      if (!context) {
        continue;
      }
      GenerateResponse response;
      response.uid = msgs[i].uid;
      response.incremental_output = chunks[i];
      response.finished = msgs[i].finished;

      {
        std::lock_guard<std::mutex> lock(context->mutex);
        context->token_ids.push_back(msgs[i].next_token);
        context->text += response.incremental_output;
        context->pending_chunks.push_back(response);
        if (msgs[i].finished) {
          context->finished = true;
        }
      }
      context->cv.notify_all();
    }
  }).detach();
}

std::shared_ptr<RequestContext> FrontendManager::get_or_create_context(uint64_t uid) {
  std::lock_guard<std::mutex> lock(contexts_mutex_);
  auto it = contexts_.find(uid);
  if (it != contexts_.end()) {
    return it->second;
  }
  auto context = std::make_shared<RequestContext>(uid);
  contexts_.emplace(uid, context);
  return context;
}

std::shared_ptr<RequestContext> FrontendManager::find_context(uint64_t uid) {
  std::lock_guard<std::mutex> lock(contexts_mutex_);
  auto it = contexts_.find(uid);
  return it == contexts_.end() ? nullptr : it->second;
}

void FrontendManager::complete_request(const std::shared_ptr<RequestContext>& context) {
  {
    std::lock_guard<std::mutex> lock(context->mutex);
    context->finished = true;
  }
  context->cv.notify_all();
}

}  // namespace sglang
