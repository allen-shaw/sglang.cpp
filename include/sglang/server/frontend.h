#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sglang/message/message.h"
#include "sglang/message/tokenizer_msg.h"
#include "sglang/scheduler/scheduler.h"
#include "sglang/server/args.h"
#include "sglang/tokenizer/hf_tokenizer.h"

namespace sglang {

struct GenerateTextResult {
  uint64_t uid = 0;
  std::string text;
  std::vector<int32_t> token_ids;
  bool finished = false;
};

struct RequestContext {
  explicit RequestContext(uint64_t request_uid = 0) : uid(request_uid) {}

  uint64_t uid = 0;
  std::vector<int32_t> token_ids;
  std::string text;
  std::deque<GenerateResponse> pending_chunks;
  bool finished = false;
  bool aborted = false;
  std::mutex mutex;
  std::condition_variable cv;
};

class TokenizerWorkerPool {
 public:
  using TokenizeDoneCallback = std::function<void(torch::Tensor)>;
  using DetokenizeDoneCallback = std::function<void(std::vector<std::string>)>;
  using ErrorCallback = std::function<void(std::exception_ptr)>;

  TokenizerWorkerPool(const std::string& tokenizer_json_path, int num_encode_threads);
  ~TokenizerWorkerPool();

  std::future<torch::Tensor> tokenize_async(TokenizeInput input,
                                            SamplingParams sampling_params);
  std::future<std::vector<std::string>> detokenize_async(std::vector<DetokenizeMsg> msgs);
  void tokenize_dispatch(TokenizeInput input, SamplingParams sampling_params,
                         TokenizeDoneCallback on_done,
                         ErrorCallback on_error = {});
  void detokenize_dispatch(std::vector<DetokenizeMsg> msgs,
                           DetokenizeDoneCallback on_done,
                           ErrorCallback on_error = {});

 private:
  template <typename Fn>
  auto enqueue_encode(Fn&& fn)
      -> std::future<typename std::invoke_result_t<Fn>>;
  template <typename Fn>
  auto enqueue_decode(Fn&& fn)
      -> std::future<typename std::invoke_result_t<Fn>>;

  void encode_loop();
  void decode_loop();

  HFTokenizeManager tokenize_manager_;
  HFDetokenizeManager detokenize_manager_;
  int num_encode_threads_ = 0;
  std::vector<std::thread> encode_threads_;
  std::thread decode_thread_;

  std::queue<std::function<void()>> encode_tasks_;
  std::queue<std::function<void()>> decode_tasks_;
  std::mutex encode_mutex_;
  std::mutex decode_mutex_;
  std::condition_variable encode_cv_;
  std::condition_variable decode_cv_;
  bool stop_ = false;
};

class SchedulerRunner {
 public:
  using ReplyCallback = std::function<void(std::vector<DetokenizeMsg>)>;

  SchedulerRunner(const SchedulerConfig& config, ReplyCallback reply_callback);
  ~SchedulerRunner();

  void start();
  void stop();
  void submit(GenerateRequest request);
  void abort(uint64_t uid);

 private:
  enum class CommandType {
    kSubmit,
    kAbort,
  };

  struct Command {
    CommandType type;
    GenerateRequest request;
    uint64_t uid = 0;
  };

  void run_loop();

  Scheduler scheduler_;
  ReplyCallback reply_callback_;
  std::thread worker_;
  std::queue<Command> commands_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool running_ = false;
  bool stop_requested_ = false;
};

class FrontendManager {
 public:
  FrontendManager(const ServerArgs& args,
                  TokenizerWorkerPool& tokenizer_pool,
                  SchedulerRunner& scheduler_runner);
  ~FrontendManager() = default;

  uint64_t new_request();
  uint64_t submit_tokenized_request(torch::Tensor input_ids, SamplingParams sampling_params);
  uint64_t submit_text_request(TokenizeInput input, SamplingParams sampling_params);
  void abort(uint64_t uid);
  GenerateTextResult wait_result(uint64_t uid);
  bool wait_next_chunk(uint64_t uid, GenerateResponse& response);
  void handle_detokenize(std::vector<DetokenizeMsg> msgs);

 private:
  std::shared_ptr<RequestContext> get_or_create_context(uint64_t uid);
  std::shared_ptr<RequestContext> find_context(uint64_t uid);
  void complete_request(const std::shared_ptr<RequestContext>& context);

  ServerArgs args_;
  TokenizerWorkerPool& tokenizer_pool_;
  SchedulerRunner& scheduler_runner_;
  std::atomic<uint64_t> next_uid_{0};
  std::mutex contexts_mutex_;
  std::unordered_map<uint64_t, std::shared_ptr<RequestContext>> contexts_;
};

template <typename Fn>
auto TokenizerWorkerPool::enqueue_encode(Fn&& fn)
    -> std::future<typename std::invoke_result_t<Fn>> {
  using ReturnT = typename std::invoke_result_t<Fn>;
  auto task = std::make_shared<std::packaged_task<ReturnT()>>(std::forward<Fn>(fn));
  auto future = task->get_future();
  {
    std::lock_guard<std::mutex> lock(encode_mutex_);
    encode_tasks_.push([task]() { (*task)(); });
  }
  encode_cv_.notify_one();
  return future;
}

template <typename Fn>
auto TokenizerWorkerPool::enqueue_decode(Fn&& fn)
    -> std::future<typename std::invoke_result_t<Fn>> {
  using ReturnT = typename std::invoke_result_t<Fn>;
  auto task = std::make_shared<std::packaged_task<ReturnT()>>(std::forward<Fn>(fn));
  auto future = task->get_future();
  {
    std::lock_guard<std::mutex> lock(decode_mutex_);
    decode_tasks_.push([task]() { (*task)(); });
  }
  decode_cv_.notify_one();
  return future;
}

}  // namespace sglang
