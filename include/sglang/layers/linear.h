#pragma once

#include <torch/torch.h>

namespace sglang {

class LinearBase {
 public:
    LinearBase(int full_isize, int full_osize, int local_isize, int local_osize, bool has_bias);
    virtual ~LinearBase() = default;

    virtual torch::Tensor forward(const torch::Tensor& x);

    torch::Tensor weight;
    torch::Tensor bias;
    bool has_bias_;

 protected:
    int full_input_size_;
    int full_output_size_;
    int local_input_size_;
    int local_output_size_;
    // FIXME: Distributed communicator handle placeholder
};

class LinearReplicated : public LinearBase {
 public:
    LinearReplicated(int input_size, int output_size, bool has_bias);
};

class LinearColParallelMerged : public LinearBase {
 public:
    LinearColParallelMerged(int input_size, const std::vector<int>& output_sizes, bool has_bias);
};

class LinearRowParallel : public LinearBase {
 public:
    LinearRowParallel(int input_size, int output_size, bool has_bias);
    torch::Tensor forward(const torch::Tensor& x) override;
};

class LinearQKVMerged : public LinearBase {
 public:
    LinearQKVMerged(int hidden_size, int head_dim, int num_qo_heads, int num_kv_heads, bool has_bias);
};

class LinearOProj : public LinearBase {
 public:
    LinearOProj(int input_size, int output_size, bool has_bias);
    torch::Tensor forward(const torch::Tensor& x) override;
};

}  // namespace sglang
