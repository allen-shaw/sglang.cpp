#pragma once

#include <torch/torch.h>

namespace sglang {

class LinearBase : public torch::nn::Module {
 public:
    LinearBase(int full_isize, int full_osize, int local_isize, int local_osize, bool has_bias);
    virtual ~LinearBase() = default;

    virtual torch::Tensor forward(const torch::Tensor& x);

    torch::Tensor weight;
    torch::Tensor bias;
    bool has_bias_;

    int full_input_size() const { return full_input_size_; }
    int full_output_size() const { return full_output_size_; }
    int local_input_size() const { return local_input_size_; }
    int local_output_size() const { return local_output_size_; }

 protected:
    int full_input_size_;
    int full_output_size_;
    int local_input_size_;
    int local_output_size_;
};

class LinearReplicated : public LinearBase {
 public:
    LinearReplicated(int input_size, int output_size, bool has_bias);
};

class LinearColParallelMerged : public LinearBase {
 public:
    LinearColParallelMerged(int input_size, const std::vector<int>& output_sizes, bool has_bias);
    const std::vector<int>& output_sizes() const { return output_sizes_; }

 private:
    std::vector<int> output_sizes_;
    std::vector<int> local_output_sizes_;
};

class LinearRowParallel : public LinearBase {
 public:
    LinearRowParallel(int input_size, int output_size, bool has_bias);
    torch::Tensor forward(const torch::Tensor& x) override;
};

class LinearQKVMerged : public LinearBase {
 public:
    LinearQKVMerged(int hidden_size, int head_dim, int num_qo_heads, int num_kv_heads, bool has_bias);

    int head_dim() const { return head_dim_; }
    int num_qo_heads() const { return num_qo_heads_; }
    int num_kv_heads() const { return num_kv_heads_; }
    int local_num_qo_heads() const { return local_num_qo_heads_; }
    int local_num_kv_heads() const { return local_num_kv_heads_; }

 private:
    int head_dim_;
    int num_qo_heads_;
    int num_kv_heads_;
    int local_num_qo_heads_;
    int local_num_kv_heads_;
};

class LinearOProj : public LinearBase {
 public:
    LinearOProj(int input_size, int output_size, bool has_bias);
    torch::Tensor forward(const torch::Tensor& x) override;
};

}  // namespace sglang
