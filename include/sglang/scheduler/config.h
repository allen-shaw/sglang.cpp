#pragma once

#include <string>

#include "sglang/engine/config.h"

namespace sglang {

struct SchedulerConfig : public EngineConfig {
    int max_extend_tokens = 8192;
    std::string cache_type = "radix";
    bool enable_overlap_scheduling = false;
};

}  // namespace sglang
