#include <gtest/gtest.h>
#include <torch/torch.h>
#include "sglang/moe/moe.h"

using namespace sglang;

TEST(MoETest, ExpertLayerForward) {
    MoELayer moe(8, 2, 128, 256, false);
    
    auto hidden = torch::randn({2, 128});
    auto router_logits = torch::randn({2, 8});

    auto out = moe.forward(hidden, router_logits);
    EXPECT_EQ(out.sizes().vec(), std::vector<int64_t>({2, 128}));
}
