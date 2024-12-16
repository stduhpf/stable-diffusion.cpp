#ifndef __LINDIT_HPP__
#define __LINDIT_HPP__

#include "ggml_extend.hpp"
#include "model.h"

// TODO: estimate actual size
#define LINDIT_GRAPH_SIZE 10240

struct LinearAttentionHead : public GGMLBlock {
public:
    LinearAttentionHead(int64 dim, bool qkv_bias = false) {
        blocks["to_q"]     = std::shared_ptr<Linear>(new Linear(dim, qkv_bias));
        blocks["to_k"]     = std::shared_ptr<Linear>(new Linear(dim, qkv_bias));
        blocks["to_v"]     = std::shared_ptr<Linear>(new Linear(dim, qkv_bias));
        blocks["to_out.0"] = std::shared_ptr<Linear>(new Linear(dim, true));
    }
};

struct MixFFN : public GGMLBlock {
public:
    MixFFN(int64 hidden_size = 1152, int64 ffn_dim = 5760, int64_t what = 2880) {
        // inverted residual block
        blocks["conv_inverted"] = std::shared_ptr<Conv2d>(new Conv2d(hidden_size, ffn_dim, 1));
        // Convolution
        blocks["conv_depth"] = std::shared_ptr<Conv2d>(new Conv2d(1, ffn_dim, 3, {1, 1}, {0, 0}, {1, 1}, true));
        // residual block
        blocks["conv_point"] = std::shared_ptr<Conv2d>(new Conv2d(what, hidden_size, 1));
    }
};

struct LinearTransformerBlock : public GGMLBlock {
protected:
    int64_t hidden_size;

    void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "") {
        ggml_type wtype             = (tensor_types.find(prefix + "scale") != tensor_types.end()) ? tensor_types[prefix + "scale"] : GGML_TYPE_F32;
        params["scale_shift_table"] = ggml_new_tensor_2d(ctx, wtype, hidden_size, 6);
    }

public:
    LinearTransformerBlock(int64 dim) {
        // TODO: init
        blocks["attn1"] = std::shared_ptr<LinearAttentionHead>(new LinearAttentionHead(dim));
        blocks["attn2"] = std::shared_ptr<LinearAttentionHead>(new LinearAttentionHead(dim, true));
        blocks["ff"]    = std::shared_ptr<MixFFN>(new MixFFN(dim));
    }
};

struct LinDiT : public GGMLBlock {
public:
    LinDiT(std::map<std::string, enum ggml_type>& tensor_types) {
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x,
                                struct ggml_tensor* t,
                                struct ggml_tensor* y        = NULL,
                                struct ggml_tensor* context  = NULL,
                                std::vector<int> skip_layers = std::vector<int>()) {
    }
};

struct LinDiTRunner : public GGMLRunner {
    LinDiT model;
    static std::map<std::string, enum ggml_type> empty_tensor_types;

    MMDiTRunner(ggml_backend_t backend,
                std::map<std::string, enum ggml_type>& tensor_types = empty_tensor_types,
                const std::string prefix                            = "")
        : GGMLRunner(backend), model(tensor_types) {
        model.init(params_ctx, tensor_types, prefix);
    }

    std::string get_desc() {
        return "lindit";
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        model.get_param_tensors(tensors, prefix);
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* x,
                                    struct ggml_tensor* timesteps,
                                    struct ggml_tensor* context,
                                    struct ggml_tensor* y,
                                    std::vector<int> skip_layers = std::vector<int>()) {
        struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, LINDIT_GRAPH_SIZE, false);

        x         = to_backend(x);
        context   = to_backend(context);
        y         = to_backend(y);
        timesteps = to_backend(timesteps);

        struct ggml_tensor* out = model.forward(compute_ctx,
                                                x,
                                                timesteps,
                                                y,
                                                context,
                                                skip_layers);

        ggml_build_forward_expand(gf, out);

        return gf;
    }

    void compute(int n_threads,
                 struct ggml_tensor* x,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* context,
                 struct ggml_tensor* y,
                 struct ggml_tensor** output     = NULL,
                 struct ggml_context* output_ctx = NULL,
                 std::vector<int> skip_layers    = std::vector<int>()) {
        // x: [N, in_channels, h, w]
        // timesteps: [N, ]
        // context: [N, max_position, hidden_size]([N, 154, 4096]) or [1, max_position, hidden_size]
        // y: [N, adm_in_channels] or [1, adm_in_channels]
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(x, timesteps, context, y, skip_layers);
        };

        GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
    }
};

#endif