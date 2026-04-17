#include <gtest/gtest.h>
#include <torch/torch.h>
#include <iostream>
#include <filesystem>
#include <fstream>

#include "sglang/models/qwen3.h"
#include "sglang/models/config.h"
#include "sglang/models/weight_loader.h"
#include "sglang/attention/flashinfer_backend.h"
#include "sglang/kvcache/mha_kvcache.h"
#include "sglang/core/context.h"
#include "sglang/core/batch.h"
#include "sglang/core/req.h"
#include <tokenizers_cpp.h>
#include <fstream>
#include <sstream>

namespace {

// Find model path from env or default HF cache
std::string find_model_path() {
    const char* env_path = std::getenv("QWEN3_MODEL_PATH");
    if (env_path) {
        return std::string(env_path);
    }

    // Check default HuggingFace cache
    const char* home = std::getenv("HOME");
    if (!home) return "";

    std::string hf_dir = std::string(home) + "/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots";
    if (!std::filesystem::exists(hf_dir)) return "";

    // Return the first snapshot directory
    for (const auto& entry : std::filesystem::directory_iterator(hf_dir)) {
        if (entry.is_directory()) {
            return entry.path().string();
        }
    }
    return "";
}

bool contains_paris(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text.find("paris") != std::string::npos || text.find("巴黎") != std::string::npos;
}

}  // namespace

class Qwen3IntegrationTest : public ::testing::Test {
 protected:
    static std::string model_path_;

    static void SetUpTestSuite() {
        model_path_ = find_model_path();
        if (model_path_.empty()) {
            GTEST_SKIP() << "Model not found. Set QWEN3_MODEL_PATH or run: huggingface-cli download Qwen/Qwen3-0.6B";
        }
        std::cout << "Using model at: " << model_path_ << std::endl;
    }
};

std::string Qwen3IntegrationTest::model_path_;

// ============================================================
// Component Tests
// ============================================================

TEST_F(Qwen3IntegrationTest, TestLoadConfig) {
    if (model_path_.empty()) GTEST_SKIP();

    auto config_path = model_path_ + "/config.json";
    ASSERT_TRUE(std::filesystem::exists(config_path)) << "config.json not found at: " << config_path;

    auto config = sglang::ModelConfig::from_json_file(config_path);

    // Qwen3-0.6B specific values
    EXPECT_EQ(config.num_layers, 28);
    EXPECT_EQ(config.num_qo_heads, 16);
    EXPECT_EQ(config.num_kv_heads, 8);
    EXPECT_EQ(config.head_dim, 128);
    EXPECT_EQ(config.hidden_size, 1024);
    EXPECT_EQ(config.vocab_size, 151936);
    EXPECT_EQ(config.intermediate_size, 3072);
    EXPECT_FLOAT_EQ(config.rms_norm_eps, 1e-6f);
    EXPECT_FLOAT_EQ(config.rotary_config.base, 1000000.0f);
    EXPECT_EQ(config.rotary_config.max_position, 40960);
    EXPECT_TRUE(config.tie_word_embeddings);
    EXPECT_EQ(config.model_type, "qwen3");

    std::cout << "Config loaded successfully!" << std::endl;
    std::cout << "  num_layers=" << config.num_layers 
              << " num_qo=" << config.num_qo_heads
              << " num_kv=" << config.num_kv_heads
              << " head_dim=" << config.head_dim << std::endl;
}

TEST_F(Qwen3IntegrationTest, TestTokenizer) {
    if (model_path_.empty()) GTEST_SKIP();

    auto tokenizer_path = model_path_ + "/tokenizer.json";
    ASSERT_TRUE(std::filesystem::exists(tokenizer_path)) << "tokenizer.json not found";

    std::ifstream file(tokenizer_path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(buffer.str());

    // Test encode
    std::string text = "The capital of France is";
    auto ids = tokenizer->Encode(text);
    EXPECT_GT(ids.size(), 0) << "Encoding produced no tokens";
    std::cout << "Text: \"" << text << "\" -> " << ids.size() << " tokens: [";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << ids[i];
    }
    std::cout << "]" << std::endl;

    // Test decode roundtrip
    auto decoded = tokenizer->Decode(ids);
    std::cout << "Decoded: \"" << decoded << "\"" << std::endl;
    // Note: decode may not perfectly match due to special tokens, but should be close
    EXPECT_NE(decoded.find("capital"), std::string::npos);
}

TEST_F(Qwen3IntegrationTest, TestModelInit) {
    if (model_path_.empty()) GTEST_SKIP();

    auto config = sglang::ModelConfig::from_json_file(model_path_ + "/config.json");
    sglang::Qwen3ForCausalLM model(config);

    // Count parameters
    auto params = model.named_parameters();
    int count = 0;
    for (auto& p : params) {
        count++;
    }
    EXPECT_GT(count, 0) << "Model has no parameters";
    std::cout << "Model initialized with " << count << " parameter groups" << std::endl;
}

TEST_F(Qwen3IntegrationTest, TestLoadWeights) {
    if (model_path_.empty()) GTEST_SKIP();
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto config = sglang::ModelConfig::from_json_file(model_path_ + "/config.json");
    sglang::Qwen3ForCausalLM model(config);

    // Load weights
    ASSERT_NO_THROW(
        sglang::WeightLoader::load_weights(model, model_path_, torch::kBFloat16, torch::kCUDA)
    );

    // Verify some weights are non-zero
    auto params = model.named_parameters();
    for (auto& p : params) {
        if (p.key().find("embed_tokens") != std::string::npos) {
            EXPECT_GT(p.value().abs().sum().item<float>(), 0.0f)
                << "embed_tokens weights are all zeros";
            break;
        }
    }

    std::cout << "Weights loaded successfully!" << std::endl;
}

// ============================================================
// Phase A: Prefill-only Test
// ============================================================

TEST_F(Qwen3IntegrationTest, TestPrefillForward) {
    if (model_path_.empty()) GTEST_SKIP();
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    torch::NoGradGuard no_grad;

    // 1. Load config and model
    auto config = sglang::ModelConfig::from_json_file(model_path_ + "/config.json");
    sglang::Qwen3ForCausalLM model(config);
    sglang::WeightLoader::load_weights(model, model_path_, torch::kBFloat16, torch::kCUDA);

    // 2. Tokenize
    std::ifstream file(model_path_ + "/tokenizer.json");
    std::stringstream buffer;
    buffer << file.rdbuf();
    auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(buffer.str());
    auto token_ids = tokenizer->Encode("The capital of France is");
    int seq_len = static_cast<int>(token_ids.size());
    std::cout << "Input sequence length: " << seq_len << std::endl;

    // 3. Create input tensors
    auto input_ids = torch::tensor(token_ids, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    auto positions = torch::arange(seq_len, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto out_loc = torch::arange(seq_len, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    // 4. Set up infrastructure
    int num_pages = seq_len + 64;  // enough pages for this test
    auto kv_cache = std::make_shared<sglang::MHAKVCache>(
        config.num_kv_heads, config.num_layers, config.head_dim,
        num_pages, /*page_size=*/1,
        torch::kBFloat16, torch::kCUDA
    );

    auto attn_backend = std::make_shared<sglang::FlashInferBackend>(
        kv_cache, config.num_qo_heads, config.num_kv_heads, config.head_dim
    );

    // Reset global context if previously set (for test isolation)
    try { sglang::get_global_ctx(); } catch (...) {}

    auto ctx = std::make_shared<sglang::Context>(/*page_size=*/1, attn_backend);
    ctx->kv_cache = kv_cache;

    // 5. Create batch
    auto req = std::make_shared<sglang::Req>();
    req->input_ids = torch::tensor(token_ids, torch::kInt64);  // CPU
    req->cached_len = 0;
    req->output_len = 1;
    req->req_id = 0;
    req->table_idx = 0;

    auto batch = std::make_shared<sglang::Batch>();
    batch->reqs = {req};
    batch->phase = sglang::BatchPhase::Prefill;
    batch->input_ids = input_ids;
    batch->positions = positions;
    batch->out_loc = out_loc;
    batch->padded_reqs = {req};

    // Prepare attention metadata
    attn_backend->prepare_metadata(*batch);

    // Set global context and batch
    sglang::set_global_ctx(ctx);
    sglang::BatchGuard guard(ctx, batch);

    // 6. Forward pass
    auto logits = model.forward(input_ids, positions);

    // 7. Verify output
    ASSERT_EQ(logits.dim(), 2);
    EXPECT_EQ(logits.size(0), seq_len);
    EXPECT_EQ(logits.size(1), config.vocab_size);

    // Get the prediction for the last token
    auto last_logits = logits.select(0, seq_len - 1);  // [vocab_size]
    auto next_token = last_logits.argmax().item<int64_t>();
    
    // Decode the predicted token
    auto predicted_text = tokenizer->Decode({static_cast<int32_t>(next_token)});
    std::cout << "Next token prediction: " << next_token << " -> \"" << predicted_text << "\"" << std::endl;

    // The model should predict something reasonable after "The capital of France is"
    // We check that logits are not all the same (model is actually computing)
    EXPECT_GT(last_logits.max().item<float>() - last_logits.min().item<float>(), 0.1f)
        << "Logits have no variance - model may not be computing correctly";
    EXPECT_TRUE(contains_paris(predicted_text))
        << "Expected next token to mention Paris/巴黎, got: " << predicted_text;

    std::cout << "Prefill forward pass completed successfully!" << std::endl;
}

// ============================================================
// Phase B: Full Prefill + Decode Generation Test
// ============================================================

TEST_F(Qwen3IntegrationTest, TestGreedyGeneration) {
    if (model_path_.empty()) GTEST_SKIP();
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    torch::NoGradGuard no_grad;

    // 1. Load config and model
    auto config = sglang::ModelConfig::from_json_file(model_path_ + "/config.json");
    sglang::Qwen3ForCausalLM model(config);
    sglang::WeightLoader::load_weights(model, model_path_, torch::kBFloat16, torch::kCUDA);

    // 2. Tokenize
    std::ifstream file(model_path_ + "/tokenizer.json");
    std::stringstream buffer;
    buffer << file.rdbuf();
    auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(buffer.str());
    std::string prompt = "The capital of France is";
    auto token_ids = tokenizer->Encode(prompt);
    int prompt_len = static_cast<int>(token_ids.size());
    int max_new_tokens = 20;

    std::cout << "Prompt: \"" << prompt << "\" (" << prompt_len << " tokens)" << std::endl;

    // 3. Set up KV cache and attention backend
    int total_len = prompt_len + max_new_tokens;
    auto kv_cache = std::make_shared<sglang::MHAKVCache>(
        config.num_kv_heads, config.num_layers, config.head_dim,
        total_len + 16, /*page_size=*/1,
        torch::kBFloat16, torch::kCUDA
    );

    auto attn_backend = std::make_shared<sglang::FlashInferBackend>(
        kv_cache, config.num_qo_heads, config.num_kv_heads, config.head_dim
    );

    auto ctx = std::make_shared<sglang::Context>(/*page_size=*/1, attn_backend);
    ctx->kv_cache = kv_cache;
    sglang::set_global_ctx(ctx);

    // 4. Prefill phase
    auto input_ids = torch::tensor(token_ids, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    auto positions = torch::arange(prompt_len, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto out_loc = torch::arange(prompt_len, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    auto req = std::make_shared<sglang::Req>();
    req->input_ids = torch::tensor(token_ids, torch::kInt64);
    req->cached_len = 0;
    req->output_len = max_new_tokens;
    req->req_id = 0;
    req->table_idx = 0;

    auto batch = std::make_shared<sglang::Batch>();
    batch->reqs = {req};
    batch->phase = sglang::BatchPhase::Prefill;
    batch->input_ids = input_ids;
    batch->positions = positions;
    batch->out_loc = out_loc;
    batch->padded_reqs = {req};

    attn_backend->prepare_metadata(*batch);
    {
        sglang::BatchGuard guard(ctx, batch);
        auto logits = model.forward(input_ids, positions);
        auto next_token = logits.select(0, prompt_len - 1).argmax().item<int64_t>();
        token_ids.push_back(static_cast<int>(next_token));
        req->complete_one();
    }

    std::cout << "Prefill done. Generated tokens: [" << token_ids.back() << "]" << std::endl;

    // 5. Decode loop
    for (int step = 1; step < max_new_tokens; ++step) {
        int cur_pos = prompt_len + step - 1;

        // Single new token
        auto new_input = torch::tensor({token_ids.back()}, 
            torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
        auto new_pos = torch::tensor({cur_pos},
            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        auto new_out_loc = torch::tensor({static_cast<int64_t>(cur_pos)},
            torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

        auto decode_batch = std::make_shared<sglang::Batch>();
        decode_batch->reqs = {req};
        decode_batch->phase = sglang::BatchPhase::Decode;
        decode_batch->input_ids = new_input;
        decode_batch->positions = new_pos;
        decode_batch->out_loc = new_out_loc;
        decode_batch->padded_reqs = {req};

        attn_backend->prepare_metadata(*decode_batch);
        {
            sglang::BatchGuard guard(ctx, decode_batch);
            auto logits = model.forward(new_input, new_pos);
            auto next_token = logits.select(0, 0).argmax().item<int64_t>();
            token_ids.push_back(static_cast<int>(next_token));
            req->complete_one();
        }

        // Check for EOS
        if (token_ids.back() == 151645) {  // eos_token_id for Qwen3
            std::cout << "EOS reached at step " << step << std::endl;
            break;
        }
    }

    // 6. Decode full text
    auto generated_text = tokenizer->Decode(token_ids);
    std::cout << "Generated text: \"" << generated_text << "\"" << std::endl;

    // Verify output is reasonable
    EXPECT_GT(token_ids.size(), static_cast<size_t>(prompt_len))
        << "No new tokens were generated";
    EXPECT_TRUE(contains_paris(generated_text))
        << "Expected generated text to mention Paris/巴黎, got: " << generated_text;

    std::string lower_text = generated_text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    
    // Log all generated tokens for debugging
    std::cout << "Generated token IDs: [";
    for (size_t i = prompt_len; i < token_ids.size(); ++i) {
        if (i > static_cast<size_t>(prompt_len)) std::cout << ", ";
        std::cout << token_ids[i];
    }
    std::cout << "]" << std::endl;
}
