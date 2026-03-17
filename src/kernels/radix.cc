#include "sglang/kernels/radix.h"

namespace sglang {

int64_t fast_compare_key(const torch::Tensor& lhs, const torch::Tensor& rhs) {
    TORCH_CHECK(lhs.dim() == 1 && rhs.dim() == 1,
                "fast_compare_key: both tensors must be 1-D");
    TORCH_CHECK(lhs.is_cpu() && rhs.is_cpu(),
                "fast_compare_key: both tensors must be on CPU");

    const int64_t len = std::min(lhs.size(0), rhs.size(0));
    if (len == 0) {
        return 0;
    }

    const auto lhs_acc = lhs.accessor<int32_t, 1>();
    const auto rhs_acc = rhs.accessor<int32_t, 1>();

    for (int64_t i = 0; i < len; ++i) {
        if (lhs_acc[i] != rhs_acc[i]) {
            return i;
        }
    }
    return len;
}

}  // namespace sglang
