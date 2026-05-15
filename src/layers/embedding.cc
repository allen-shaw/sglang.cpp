#include "sglang/layers/embedding.h"

#include "sglang/distributed/distributed.h"

namespace sglang {

VocabParallelEmbedding::VocabParallelEmbedding(int num_embeddings, int embedding_dim)
    : num_embeddings_(num_embeddings) {
    auto tp = get_tp_info();
    tp_size_ = tp.size;
    num_embeddings_tp_ = (num_embeddings + tp_size_ - 1) / tp_size_;
    vocab_start_idx_ = num_embeddings_tp_ * tp.rank;
    vocab_end_idx_ = std::min(vocab_start_idx_ + num_embeddings_tp_, num_embeddings);

    weight = register_parameter(
        "weight",
        torch::empty({num_embeddings_tp_, embedding_dim}, torch::device(torch::kCUDA).dtype(torch::kFloat16))
    );
}

torch::Tensor VocabParallelEmbedding::forward(const torch::Tensor& x) {
    if (tp_size_ == 1) {
        return torch::nn::functional::embedding(x, weight);
    }
    auto mask = x.lt(vocab_start_idx_) | x.ge(vocab_end_idx_);
    auto local_ids = (x - vocab_start_idx_).masked_fill(mask, 0);
    auto out = torch::nn::functional::embedding(local_ids, weight);
    out.masked_fill_(mask.unsqueeze(-1), 0);
    return tensor_model_parallel_all_reduce(out);
}

ParallelLMHead::ParallelLMHead(int num_embeddings, int embedding_dim, bool has_bias, VocabParallelEmbedding* tied_embedding)
    : VocabParallelEmbedding(num_embeddings, embedding_dim), tied_embedding_(tied_embedding) {
    if (has_bias) {
        bias = register_parameter(
            "bias",
            torch::empty({num_embeddings_tp_}, torch::device(torch::kCUDA).dtype(torch::kFloat16))
        );
    }
}

torch::Tensor ParallelLMHead::forward(const torch::Tensor& x) {
    auto active_weight = tied_embedding_ ? tied_embedding_->weight : this->weight;
    return torch::nn::functional::linear(x, active_weight, this->bias.defined() ? this->bias : torch::Tensor());
}

}  // namespace sglang
