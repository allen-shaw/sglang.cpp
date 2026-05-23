#include "sglang/layers/linear.h"

#include "sglang/distributed/distributed.h"

#include <numeric>

namespace sglang {

LinearBase::LinearBase(int full_isize, int full_osize, int local_isize, int local_osize, bool has_bias)
    : full_input_size_(full_isize), full_output_size_(full_osize),
      local_input_size_(local_isize), local_output_size_(local_osize), has_bias_(has_bias) {
    weight = register_parameter(
        "weight",
        torch::empty({local_osize, local_isize}, torch::device(torch::kCUDA).dtype(torch::kFloat16))
    );
    if (has_bias) {
        bias = register_parameter(
            "bias",
            torch::empty({local_osize}, torch::device(torch::kCUDA).dtype(torch::kFloat16))
        );
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
                 std::accumulate(output_sizes.begin(), output_sizes.end(), 0) / tp_size(),
                 has_bias),
      output_sizes_(output_sizes) {
    local_output_sizes_.reserve(output_sizes.size());
    int local_total = 0;
    for (int size : output_sizes_) {
        int local_size = divide_even(size, tp_size(), "LinearColParallelMerged output size");
        local_output_sizes_.push_back(local_size);
        local_total += local_size;
    }
    TORCH_CHECK(local_total == local_output_size_,
                "LinearColParallelMerged local output size mismatch");
}

LinearRowParallel::LinearRowParallel(int input_size, int output_size, bool has_bias)
    : LinearBase(input_size, output_size, divide_even(input_size, tp_size(), "LinearRowParallel input size"), output_size, has_bias) {}

torch::Tensor LinearRowParallel::forward(const torch::Tensor& x) {
    auto y = torch::nn::functional::linear(x, weight, has_bias_ ? bias : torch::Tensor());
    return tensor_model_parallel_all_reduce(y);
}

LinearQKVMerged::LinearQKVMerged(int hidden_size, int head_dim, int num_qo_heads, int num_kv_heads, bool has_bias)
    : LinearBase(hidden_size,
                 (num_qo_heads + 2 * num_kv_heads) * head_dim,
                 hidden_size,
                 (divide_even(num_qo_heads, tp_size(), "num_qo_heads") +
                  2 * divide_even(num_kv_heads, tp_size(), "num_kv_heads")) * head_dim,
                 has_bias),
      head_dim_(head_dim),
      num_qo_heads_(num_qo_heads),
      num_kv_heads_(num_kv_heads),
      local_num_qo_heads_(divide_even(num_qo_heads, tp_size(), "num_qo_heads")),
      local_num_kv_heads_(divide_even(num_kv_heads, tp_size(), "num_kv_heads")) {}

LinearOProj::LinearOProj(int input_size, int output_size, bool has_bias)
    : LinearBase(input_size, output_size, divide_even(input_size, tp_size(), "LinearOProj input size"), output_size, has_bias) {}

torch::Tensor LinearOProj::forward(const torch::Tensor& x) {
    auto y = torch::nn::functional::linear(x, weight, has_bias_ ? bias : torch::Tensor());
    return tensor_model_parallel_all_reduce(y);
}

}  // namespace sglang
