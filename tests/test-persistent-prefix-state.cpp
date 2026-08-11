#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static llama_context * make_context(const common_params & params, llama_model * model) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_ctx      = std::max<uint32_t>(cparams.n_ctx, 128);
    cparams.n_batch    = std::max<uint32_t>(cparams.n_batch, 32);
    cparams.n_ubatch   = std::max<uint32_t>(cparams.n_ubatch, 32);
    cparams.n_seq_max  = 4;
    cparams.n_rs_seq   = std::max<uint32_t>(cparams.n_rs_seq, 8);
    return llama_init_from_model(model, cparams);
}

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, llama_seq_id seq_id) {
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        common_batch_add(batch, tokens[i], (llama_pos) i, { seq_id }, i + 1 == tokens.size());
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_one(llama_context * ctx, llama_token token, llama_pos pos, llama_seq_id seq_id) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, token, pos, { seq_id }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool get_range_data(
        llama_context * ctx,
        llama_seq_id seq_id,
        uint32_t components,
        llama_state_seq_range range,
        llama_state_seq_flags flags,
        std::vector<uint8_t> & data) {
    const size_t size = llama_state_seq_get_size_range(ctx, seq_id, components, range, flags);
    if (size == 0) {
        return false;
    }
    data.resize(size);
    return llama_state_seq_get_data_range(ctx, data.data(), data.size(), seq_id, components, range,
                                          flags) == size;
}

static bool set_range_data(
        llama_context * ctx,
        const std::vector<uint8_t> & data,
        llama_seq_id dest_seq_id,
        uint32_t components,
        llama_state_seq_range range,
        llama_state_seq_flags flags) {
    return !data.empty() && llama_state_seq_set_data_range(ctx, data.data(), data.size(), dest_seq_id,
                                                            components, range, flags) == data.size();
}

static uint64_t range_checksum(const uint8_t * data, size_t size) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void put_u64_le(uint8_t * dst, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        dst[i] = uint8_t(value >> (i * 8));
    }
}

static bool patch_attention_position(
        std::vector<uint8_t> & data,
        size_t index,
        llama_pos position,
        bool mrope_text) {
    constexpr size_t header_size = 44;
    constexpr size_t cell_count_size = 4;
    const size_t metadata_cell_size = mrope_text ? 20 : 12;
    if (data.size() < header_size + cell_count_size + (index + 1) * metadata_cell_size) {
        return false;
    }
    const size_t position_offset = header_size + cell_count_size + index * metadata_cell_size;
    std::memcpy(data.data() + position_offset, &position, sizeof(position));
    put_u64_le(data.data() + 36, range_checksum(data.data() + header_size, data.size() - header_size));
    return true;
}

static bool patch_attention_ext(
        std::vector<uint8_t> & data,
        size_t index,
        llama_pos x,
        llama_pos y) {
    constexpr size_t header_size = 44;
    constexpr size_t cell_count_size = 4;
    constexpr size_t metadata_cell_size = 20;
    const size_t ext_offset = header_size + cell_count_size + index * metadata_cell_size + 8;
    if (data.size() < ext_offset + 2 * sizeof(llama_pos)) {
        return false;
    }
    std::memcpy(data.data() + ext_offset, &x, sizeof(x));
    std::memcpy(data.data() + ext_offset + sizeof(x), &y, sizeof(y));
    put_u64_le(data.data() + 36, range_checksum(data.data() + header_size, data.size() - header_size));
    return true;
}

static bool range_tensor_payload_equal(
        const std::vector<uint8_t> & lhs,
        const std::vector<uint8_t> & rhs,
        size_t data_offset) {
    return lhs.size() == rhs.size() &&
           lhs.size() >= data_offset &&
           std::memcmp(lhs.data() + data_offset, rhs.data() + data_offset, lhs.size() - data_offset) == 0;
}

static bool test_attention_ranges(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        llama_state_seq_flags flags) {
    constexpr uint32_t attention = LLAMA_STATE_SEQ_COMPONENT_ATTENTION;
    const bool mrope_text = flags == LLAMA_STATE_SEQ_FLAGS_MROPE_TEXT;
    const llama_state_seq_range first{ 0, 4 };
    const llama_state_seq_range second{ 4, 8 };

    llama_memory_clear(llama_get_memory(ctx), true);
    if (!decode_tokens(ctx, tokens, 0)) {
        return false;
    }

    std::vector<uint8_t> first_data;
    std::vector<uint8_t> second_data;
    if (!get_range_data(ctx, 0, attention, first, flags, first_data) ||
        !get_range_data(ctx, 0, attention, second, flags, second_data)) {
        std::fprintf(stderr, "attention: range capture failed\n");
        return false;
    }

    // A concrete destination sequence is required. The old whole-cache sentinel
    // must not be accepted by the range API.
    if (llama_state_seq_get_size_range(ctx, -1, attention, first, flags) != 0) {
        std::fprintf(stderr, "attention: concrete seq check failed\n");
        return false;
    }

    const size_t first_restore = llama_state_seq_set_data_range(ctx, first_data.data(), first_data.size(), 1,
                                                                 attention, first, flags);
    if (first_restore != first_data.size() ||
        llama_state_seq_get_size_range(ctx, 1, attention, first, flags) == 0) {
        std::fprintf(stderr, "attention: first restore failed (returned=%zu size=%zu)\n", first_restore,
                     first_data.size());
        return false;
    }

    // Range restore is additive and rejects an overlap.
    if (!set_range_data(ctx, second_data, 1, attention, second, flags) ||
        llama_state_seq_get_size_range(ctx, 1, attention, second, flags) == 0) {
        std::fprintf(stderr, "attention: second restore failed\n");
        return false;
    }
    if (llama_state_seq_set_data_range(ctx, second_data.data(), second_data.size(), 1, attention, second,
                                       flags) != 0) {
        std::fprintf(stderr, "attention: overlap accepted\n");
        return false;
    }

    // Version and checksum mismatches are cache misses, not best-effort loads.
    auto wrong_version = first_data;
    wrong_version[4] ^= 1;
    auto wrong_checksum = first_data;
    wrong_checksum[36] ^= 1;
    if (llama_state_seq_set_data_range(ctx, wrong_version.data(), wrong_version.size(), 2, attention, first,
                                       flags) != 0 ||
        llama_state_seq_set_data_range(ctx, wrong_checksum.data(), wrong_checksum.size(), 2, attention, first,
                                       flags) != 0) {
        std::fprintf(stderr, "attention: version/checksum accepted\n");
        return false;
    }

    // Valid envelopes still reject duplicate or missing logical positions in
    // the attention metadata before allocating destination cells.
    auto duplicate = first_data;
    auto missing = first_data;
    if (!patch_attention_position(duplicate, 1, 0, mrope_text) ||
        !patch_attention_position(missing, 0, 9, mrope_text) ||
        llama_state_seq_set_data_range(ctx, duplicate.data(), duplicate.size(), 2, attention, first,
                                       flags) != 0 ||
        llama_state_seq_set_data_range(ctx, missing.data(), missing.size(), 3, attention, first,
                                       flags) != 0) {
        std::fprintf(stderr, "attention: duplicate/missing accepted\n");
        return false;
    }

    if (mrope_text) {
        auto noncanonical_ext = first_data;
        if (!patch_attention_ext(noncanonical_ext, 0, 1, 0) ||
            llama_state_seq_set_data_range(ctx, noncanonical_ext.data(), noncanonical_ext.size(), 2,
                                           attention, first, flags) != 0) {
            std::fprintf(stderr, "attention: non-canonical M-RoPE ext metadata accepted\n");
            return false;
        }
    }

    // The envelope binds component and range. Supplying either a wrong component
    // or a false range must fail closed.
    if (llama_state_seq_set_data_range(ctx, first_data.data(), first_data.size(), 2,
                                       LLAMA_STATE_SEQ_COMPONENT_RECURRENT, first,
                                       flags) != 0 ||
        llama_state_seq_set_data_range(ctx, first_data.data(), first_data.size(), 2, attention, second,
                                       flags) != 0) {
        std::fprintf(stderr, "attention: wrong component/range accepted\n");
        return false;
    }

    // The source range must be complete; position 8 is not present.
    const llama_state_seq_range missing_range{ 0, (llama_pos) tokens.size() + 1 };
    if (llama_state_seq_get_size_range(ctx, 0, attention, missing_range, flags) != 0) {
        std::fprintf(stderr, "attention: missing source accepted\n");
        return false;
    }

    return true;
}

static bool test_recurrent_boundary_and_logits(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        uint32_t components,
        uint32_t capabilities,
        llama_state_seq_flags attention_flags) {
    if ((components & LLAMA_STATE_SEQ_COMPONENT_RECURRENT) == 0) {
        return true;
    }

    constexpr llama_seq_id dest_seq_id = 0;
    const llama_state_seq_range boundary{ 0, (llama_pos) tokens.size() };
    std::vector<uint8_t> recurrent_data;
    std::vector<std::pair<llama_state_seq_range, std::vector<uint8_t>>> attention_ranges;
    llama_memory_clear(llama_get_memory(ctx), true);
    if (!decode_tokens(ctx, tokens, 0) ||
        !get_range_data(ctx, 0, LLAMA_STATE_SEQ_COMPONENT_RECURRENT, boundary,
                        LLAMA_STATE_SEQ_FLAGS_NONE, recurrent_data)) {
        std::fprintf(stderr, "recurrent: capture failed\n");
        return false;
    }
    if ((capabilities & LLAMA_STATE_SEQ_CAPABILITY_ATTENTION_RANGE_SAVE) != 0) {
        constexpr llama_pos block_size = 256;
        for (llama_pos p0 = boundary.p0; p0 < boundary.p1; p0 += block_size) {
            const llama_state_seq_range range{ p0, std::min(p0 + block_size, boundary.p1) };
            attention_ranges.emplace_back(range, std::vector<uint8_t>{});
            if (!get_range_data(ctx, 0, LLAMA_STATE_SEQ_COMPONENT_ATTENTION, range,
                                attention_flags, attention_ranges.back().second)) {
                std::fprintf(stderr, "recurrent: attention range capture failed at [%d,%d)\n",
                             range.p0, range.p1);
                return false;
            }
        }
    }
    // Recurrent state is a boundary snapshot, not an arbitrary token slice.
    if (llama_state_seq_get_size_range(ctx, 0, LLAMA_STATE_SEQ_COMPONENT_RECURRENT,
                                       { 1, boundary.p1 }, LLAMA_STATE_SEQ_FLAGS_NONE) != 0) {
        std::fprintf(stderr, "recurrent: invalid boundary accepted\n");
        return false;
    }

    // Capture a complete boundary checkpoint only to obtain the source
    // continuation oracle without leaving a duplicate live prefix in unified
    // KV. The range restore below starts after the source sequence is erased,
    // matching the persistent-cache server transaction.
    const size_t full_state_size = llama_state_get_size(ctx);
    std::vector<uint8_t> full_state(full_state_size);
    if (full_state_size == 0 ||
        llama_state_get_data(ctx, full_state.data(), full_state.size()) != full_state_size ||
        !decode_one(ctx, tokens.back(), boundary.p1, 0)) {
        std::fprintf(stderr, "recurrent: source continuation oracle failed\n");
        return false;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const float * logits_src = llama_get_logits_ith(ctx, 0);
    if (logits_src == nullptr) {
        std::fprintf(stderr, "recurrent: source logits missing\n");
        return false;
    }
    std::vector<float> logits_expected(logits_src, logits_src + n_vocab);

    if (llama_state_set_data(ctx, full_state.data(), full_state.size()) != full_state_size ||
        !llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1)) {
        std::fprintf(stderr, "recurrent: failed to restore boundary checkpoint and erase source\n");
        return false;
    }

    // Restore into a different concrete sequence and verify that its metadata is
    // rebuilt well enough to continue. Attention is restored separately below.
    for (const auto & [range, data] : attention_ranges) {
        if (!set_range_data(ctx, data, dest_seq_id, LLAMA_STATE_SEQ_COMPONENT_ATTENTION, range,
                            attention_flags)) {
            std::fprintf(stderr, "recurrent: attention range restore failed at [%d,%d)\n",
                         range.p0, range.p1);
            return false;
        }
    }
    if (!set_range_data(ctx, recurrent_data, dest_seq_id, LLAMA_STATE_SEQ_COMPONENT_RECURRENT, boundary,
                        LLAMA_STATE_SEQ_FLAGS_NONE)) {
        std::fprintf(stderr, "recurrent: restore failed\n");
        return false;
    }

    std::vector<uint8_t> recurrent_restored;
    if (!get_range_data(ctx, dest_seq_id, LLAMA_STATE_SEQ_COMPONENT_RECURRENT, boundary,
                        LLAMA_STATE_SEQ_FLAGS_NONE, recurrent_restored) ||
        !range_tensor_payload_equal(recurrent_data, recurrent_restored, 44 + 4 + 8)) {
        std::fprintf(stderr, "recurrent: restored tensor payload differs from source\n");
        return false;
    }

    for (const auto & [range, data] : attention_ranges) {
        std::vector<uint8_t> restored;
        const size_t metadata_cell_size =
            attention_flags == LLAMA_STATE_SEQ_FLAGS_MROPE_TEXT ? 20 : 12;
        const size_t data_offset = 44 + 4 + (range.p1 - range.p0) * metadata_cell_size;
        if (!get_range_data(ctx, dest_seq_id, LLAMA_STATE_SEQ_COMPONENT_ATTENTION, range,
                            attention_flags, restored) ||
            !range_tensor_payload_equal(data, restored, data_offset)) {
            std::fprintf(stderr, "attention: restored tensor payload differs at [%d,%d)\n",
                         range.p0, range.p1);
            return false;
        }
    }

    if (!decode_one(ctx, tokens.back(), boundary.p1, dest_seq_id)) {
        std::fprintf(stderr, "recurrent: destination continuation failed\n");
        return false;
    }
    const float * logits_dst = llama_get_logits_ith(ctx, 0);
    if (logits_dst == nullptr) {
        std::fprintf(stderr, "recurrent: destination logits missing\n");
        return false;
    }
    if (!attention_ranges.empty()) {
        double abs_sum = 0.0;
        float max_abs = 0.0f;
        int max_abs_i = -1;
        int top_src = 0;
        int top_dst = 0;
        for (int i = 0; i < n_vocab; ++i) {
            const float abs_diff = std::fabs(logits_expected[i] - logits_dst[i]);
            abs_sum += abs_diff;
            if (abs_diff > max_abs) {
                max_abs = abs_diff;
                max_abs_i = i;
            }
            if (logits_expected[i] > logits_expected[top_src]) {
                top_src = i;
            }
            if (logits_dst[i] > logits_dst[top_dst]) {
                top_dst = i;
            }
        }
        std::fprintf(stderr,
                "recurrent: logits max=%g mean=%g top_src=%d top_dst=%d top_agree=%d\n",
                max_abs, abs_sum / n_vocab, top_src, top_dst, top_src == top_dst);
        if (max_abs > 1e-5f) {
            std::fprintf(stderr,
                    "recurrent: logits mismatch max=%g at %d mean=%g top_src=%d top_dst=%d top_delta=%g\n",
                    max_abs, max_abs_i, abs_sum / n_vocab, top_src, top_dst,
                    std::fabs(logits_expected[top_src] - logits_dst[top_src]));
            return false;
        }
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, boundary.p1 - 1, -1)) {
        // Some recurrent implementations do not expose rollback planes. The
        // boundary roundtrip above remains valid; rollback rejection is tested
        // on the Qwen hybrid fixture.
        return true;
    }
    if (llama_state_seq_get_size_range(ctx, 0, LLAMA_STATE_SEQ_COMPONENT_RECURRENT,
                                       { 0, boundary.p1 }, LLAMA_STATE_SEQ_FLAGS_NONE) != 0) {
        std::fprintf(stderr, "recurrent: rollback source accepted\n");
        return false;
    }

    // A pending recurrent rollback plane cannot be persisted.
    if (!llama_memory_seq_rm(llama_get_memory(ctx), dest_seq_id, boundary.p1 - 1, -1)) {
        return true;
    }
    if (llama_state_seq_get_size_range(ctx, dest_seq_id, LLAMA_STATE_SEQ_COMPONENT_RECURRENT,
                                       { 0, boundary.p1 }, LLAMA_STATE_SEQ_FLAGS_NONE) != 0) {
        std::fprintf(stderr, "recurrent: rollback destination accepted\n");
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    params.sampling.seed = 1234;
    params.swa_full = true;

    bool unified = false;
    size_t prefix_length = 9;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--test-unified") == 0) {
            unified = true;
            for (int j = i; j + 1 < argc; ++j) {
                argv[j] = argv[j + 1];
            }
            --argc;
            --i;
        } else if (std::strcmp(argv[i], "--test-prefix-length") == 0 && i + 1 < argc) {
            prefix_length = (size_t) std::strtoull(argv[i + 1], nullptr, 10);
            for (int j = i; j + 2 < argc; ++j) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            --i;
        }
    }
    params.kv_unified = unified;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    if (prefix_length < 9 || prefix_length > 2048) {
        std::fprintf(stderr, "%s: invalid --test-prefix-length\n", __func__);
        return 1;
    }
    params.n_ctx = std::max<uint32_t>(params.n_ctx, (uint32_t) prefix_length * 8);
    ggml_backend_load_all();

    common_init_result_ptr init = common_init_from_params(params);
    llama_model * model = init->model();
    if (model == nullptr) {
        std::fprintf(stderr, "%s: failed to load model\n", __func__);
        return 1;
    }
    if (!llama_model_is_hybrid(model) && !llama_model_is_recurrent(model)) {
        std::fprintf(stderr, "%s: skipping non-hybrid/non-recurrent model\n", __func__);
        return 0;
    }

    llama_context * ctx = make_context(params, model);
    if (ctx == nullptr) {
        std::fprintf(stderr, "%s: failed to create context\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    } else {
        tokens = common_tokenize(ctx, "The quick brown fox jumps over the lazy dog", true);
    }
    if (tokens.size() < 9) {
        tokens.resize(9, llama_vocab_bos(vocab));
    }
    tokens.resize(prefix_length, tokens.back());

    const uint32_t components = llama_state_seq_components(ctx);
    const uint32_t capabilities = llama_state_seq_capabilities(ctx);
    const bool mrope_text =
        (capabilities & LLAMA_STATE_SEQ_CAPABILITY_ATTENTION_MROPE_TEXT_RANGE) != 0;
    const llama_state_seq_flags attention_flags =
        mrope_text ? LLAMA_STATE_SEQ_FLAGS_MROPE_TEXT : LLAMA_STATE_SEQ_FLAGS_NONE;
    bool ok = true;
    if ((components & LLAMA_STATE_SEQ_COMPONENT_ATTENTION) != 0 &&
        (capabilities & LLAMA_STATE_SEQ_CAPABILITY_ATTENTION_RANGE_SAVE) != 0 &&
        (capabilities & LLAMA_STATE_SEQ_CAPABILITY_ATTENTION_RANGE_RESTORE) != 0) {
        if (mrope_text &&
            llama_state_seq_get_size_range(ctx, 0, LLAMA_STATE_SEQ_COMPONENT_ATTENTION, { 0, 4 },
                                           LLAMA_STATE_SEQ_FLAGS_NONE) != 0) {
            std::fprintf(stderr, "attention: generic flags accepted for specialized M-RoPE text cache\n");
            ok = false;
        }
        ok = test_attention_ranges(ctx, tokens, attention_flags) && ok;
    }
    if (components & LLAMA_STATE_SEQ_COMPONENT_RECURRENT) {
        ok = test_recurrent_boundary_and_logits(
                ctx, tokens, components, capabilities, attention_flags) && ok;
    }

    llama_free(ctx);
    return ok ? 0 : 1;
}
