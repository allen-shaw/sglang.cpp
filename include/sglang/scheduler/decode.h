#pragma once

#include <memory>
#include <vector>

#include "sglang/core/batch.h"

namespace sglang {

class DecodeManager {
 public:
    explicit DecodeManager(int page_size);

    void filter_reqs(const std::vector<std::shared_ptr<Req>>& reqs);
    void remove_req(const std::shared_ptr<Req>& req);
    std::shared_ptr<Req> abort_req(uint64_t uid);
    int inflight_tokens() const;
    std::shared_ptr<Batch> schedule_next_batch() const;
    bool runnable() const;

 private:
    int page_size_;
    std::vector<std::shared_ptr<Req>> running_reqs_;
};

}  // namespace sglang
