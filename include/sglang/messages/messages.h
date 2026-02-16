#pragma once

#include "sglang.pb.h"

namespace sglang {
    // Re-export specific message types if needed, or use sglang::GenerateReq directly
    using GenerateReq = sglang::GenerateReq;
    using GenerateResp = sglang::GenerateResp;
}
