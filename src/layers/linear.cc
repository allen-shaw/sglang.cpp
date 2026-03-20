#include "sglang/layers/linear.h"
#include <numeric>

namespace sglang {

LinearBase::LinearBase(int full_isize, int full_osize, int local_isize, int local_osize, bool has_bias)
    : full_input_size_(full_isize), full_output_size_(full_osize),
      local_input_size_(local_isize), local_output_size_(local_osize), has_bias_(has_bias) {
    weight = torch::empty({local_osize, local_isize}, torch::device(torch::kCUDA).dtype(torch::kFloat16));
    if (has_bias) {
        bias = torch::empty({local_osize}, torch::device(torch::kCUDA).dtype(torch::kFloat16));
    }
}

torch::Tensor LinearBase::forward(const torch::Tensor& x) {
    return torch::nn::functional::linear(x, weight, has_bias_ ? bias : torch::Tensor());
}

LinearReplicated::LinearReplicated(int input_size, int output_size, bool has_bias)
    : LinearBase(input_size, output_size, input_size, output_size, has_bias) {}

LinearColParallelMerged::LinearColParallelMerged(int input_size, const std::vector<int>& output_sizes, bool has_bias)
    : LinearBase(input_size,
                 std::accumulate(output_sizes.begin(), output_sizes.end(), 0),
                 input_size,
                 std::accumulate(output_sizes.begin(), output_sizes.end(), 0) / /*tp_size*/ 1, // FIXME: use actual tp size
                 has_bias) {}

LinearRowParallel::LinearRowParallel(int input_size, int output_size, bool has_bias)
    : LinearBase(input_size, output_size, input_size / /*tp_size*/ 1, output_size, has_bias) {}

torch::Tensor LinearRowParallel::forward(const torch::Tensor& x) {
    auto y = torch::nn::functional::linear(x, weight, has_bias_ ? bias : torch::Tensor());
    // FIXME: if tp_size > 1, all_reduce(y)
    return y;
}

LinearQKVMerged::LinearQKVMerged(int hidden_size, int head_dim, int num_qo_heads, int num_kv_heads, bool has_bias)
    : LinearBase(hidden_size,
                 (num_qo_heads + 2 * num_kv_heads) * head_dim,
                 hidden_size,
                 ((num_qo_heads / 1) + 2 * (num_kv_heads / 1)) * head_dim, // FIXME: adapt for TP size
                 has_bias) {}

LinearOProj::LinearOProj(int input_size, int output_size, bool has_bias)
    : LinearBase(input_size, output_size, input_size / /*tp_size*/ 1, output_size, has_bias) {}

torch::Tensor LinearOProj::forward(const torch::Tensor& x) {
    auto y = torch::nn::functional::linear(x, weight, has_bias_ ? bias : torch::Tensor());
    // FIXME: if tp_size > 1, all_reduce(y)
    return y;
}

}  // namespace sglang
