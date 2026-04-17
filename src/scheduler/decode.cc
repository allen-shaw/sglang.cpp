#include "sglang/scheduler/decode.h"

#include <algorithm>
#include <unordered_set>

namespace sglang {

DecodeManager::DecodeManager(int page_size) : page_size_(page_size) {}

void DecodeManager::filter_reqs(const std::vector<std::shared_ptr<Req>>& reqs) {
    std::vector<std::shared_ptr<Req>> merged;
    merged.reserve(running_reqs_.size() + reqs.size());
    std::unordered_set<uint64_t> seen;

    for (const auto& req : running_reqs_) {
        if (req->can_decode() && seen.insert(req->req_id).second) {
            merged.push_back(req);
        }
    }
    for (const auto& req : reqs) {
        if (req->can_decode() && seen.insert(req->req_id).second) {
            merged.push_back(req);
        }
    }

    running_reqs_ = std::move(merged);
}

void DecodeManager::remove_req(const std::shared_ptr<Req>& req) {
    running_reqs_.erase(
        std::remove_if(running_reqs_.begin(), running_reqs_.end(),
                       [&](const std::shared_ptr<Req>& running_req) {
                           return running_req->req_id == req->req_id;
                       }),
        running_reqs_.end());
}

std::shared_ptr<Req> DecodeManager::abort_req(uint64_t uid) {
    auto it = std::find_if(running_reqs_.begin(), running_reqs_.end(),
                           [&](const std::shared_ptr<Req>& req) {
                               return req->req_id == uid;
                           });
    if (it == running_reqs_.end()) {
        return nullptr;
    }
    auto req = *it;
    running_reqs_.erase(it);
    return req;
}

int DecodeManager::inflight_tokens() const {
    int tokens_reserved = (page_size_ - 1) * static_cast<int>(running_reqs_.size());
    for (const auto& req : running_reqs_) {
        tokens_reserved += req->remain_len();
    }
    return tokens_reserved;
}

std::shared_ptr<Batch> DecodeManager::schedule_next_batch() const {
    if (running_reqs_.empty()) {
        return nullptr;
    }
    auto batch = std::make_shared<Batch>();
    batch->reqs = running_reqs_;
    batch->phase = BatchPhase::Decode;
    return batch;
}

bool DecodeManager::runnable() const {
    return !running_reqs_.empty();
}

}  // namespace sglang
