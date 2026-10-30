#pragma once

#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

// The server batches only requests with the same edits. Restore the startup vector before another batch runs.
class server_activation_scope {
    llama_context * ctx;
    const common_control_vector_data & base;
    int32_t first;
    int32_t last;
    bool active = false;

public:
    server_activation_scope(llama_context * ctx, const common_control_vector_data & base,
                            int32_t first, int32_t last,
                            const std::map<std::pair<int32_t, int32_t>, float> & edits)
        : ctx(ctx), base(base), first(first), last(last) {
        if (edits.empty()) {
            return;
        }
        const auto * model = llama_get_model(ctx);
        const int32_t width = llama_model_n_embd(model);
        const int32_t layers = llama_model_n_layer(model);
        std::vector<float> data((size_t) width * layers, 0.0f);
        if (base.n_embd > 0) {
            for (int32_t layer = first; layer <= last && layer < layers; ++layer) {
                const size_t src = (size_t) (layer - 1) * width;
                if (src + width <= base.data.size()) {
                    std::copy_n(base.data.data() + src, width, data.data() + (size_t) layer * width);
                }
            }
        }
        for (const auto & edit : edits) {
            auto & value = data[(size_t) edit.first.first * width + edit.first.second];
            value += edit.second;
            if (!std::isfinite(value)) {
                throw std::invalid_argument("activation edit overflows startup control vector");
            }
        }
        llama_synchronize(ctx);
        if (llama_set_adapter_cvec(ctx, data.data(), data.size(), width, 0, layers - 1) != 0) {
            throw std::runtime_error("failed to apply activation edits");
        }
        active = true;
    }

    ~server_activation_scope() {
        if (active) {
            llama_synchronize(ctx);
            llama_set_adapter_cvec(ctx, base.n_embd > 0 ? base.data.data() : nullptr,
                                   base.data.size(), base.n_embd, first, last);
        }
    }

    server_activation_scope(const server_activation_scope &) = delete;
    server_activation_scope & operator=(const server_activation_scope &) = delete;
};
