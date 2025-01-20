#ifndef __LINDIT_HPP__
#define __LINDIT_HPP__

#include "ggml_extend.hpp"
#include "model.h"

// TODO: estimate actual size
#define LINDIT_GRAPH_SIZE 10240

struct PatchEmbed : public GGMLBlock {
    // 2D Image to Patch Embedding
protected:
    bool flatten;
    bool dynamic_img_pad;
    int patch_size;

public:
    PatchEmbed(int64_t img_size     = 224,
               int patch_size       = 16,
               int64_t in_chans     = 3,
               int64_t embed_dim    = 1536,
               bool bias            = true,
               bool flatten         = true,
               bool dynamic_img_pad = true)
        : patch_size(patch_size),
          flatten(flatten),
          dynamic_img_pad(dynamic_img_pad) {
        // img_size is always None
        // patch_size is always 2
        // in_chans is always 16
        // norm_layer is always False
        // strict_img_size is always true, but not used

        blocks["proj"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_chans,
                                                               embed_dim,
                                                               {patch_size, patch_size},
                                                               {patch_size, patch_size},
                                                               {0, 0},
                                                               {1, 1},
                                                               bias));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, C, H, W]
        // return: [N, H*W, embed_dim]
        auto proj = std::dynamic_pointer_cast<Conv2d>(blocks["proj"]);

        if (dynamic_img_pad) {
            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int pad_h = (patch_size - H % patch_size) % patch_size;
            int pad_w = (patch_size - W % patch_size) % patch_size;
            x         = ggml_pad(ctx, x, pad_w, pad_h, 0, 0);  // TODO: reflect pad mode
        }
        x = proj->forward(ctx, x);

        if (flatten) {
            x = ggml_reshape_3d(ctx, x, x->ne[0] * x->ne[1], x->ne[2], x->ne[3]);
            x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
        }
        return x;
    }
};

struct LinearAttentionHead : public GGMLBlock {
public:
    LinearAttentionHead(int64 dim, bool qkv_bias = false, bool output_bias = true) {
        blocks["to_q"]     = std::shared_ptr<Linear>(new Linear(dim, qkv_bias));
        blocks["to_k"]     = std::shared_ptr<Linear>(new Linear(dim, qkv_bias));
        blocks["to_v"]     = std::shared_ptr<Linear>(new Linear(dim, qkv_bias));
        blocks["to_out.0"] = std::shared_ptr<Linear>(new Linear(dim, output_bias));
    }
};

struct MixFFN : public GGMLBlock {
public:
    MixFFN(int64 hidden_size = 1152, int64 ffn_ratio = 5, int64_t point_downscale = 2) {
        int64_t mid_dim   = hidden_size * ffn_ratio;
        int64_t point_dim = mid_dim / point_downscale;
        // inverted residual block
        blocks["conv_inverted"] = std::shared_ptr<Conv2d>(new Conv2d(hidden_size, mid_dim, 1));
        // Convolution
        blocks["conv_depth"] = std::shared_ptr<Conv2d>(new Conv2d(1, mid_dim, 3, {1, 1}, {0, 0}, {1, 1}, true));
        // residual block
        blocks["conv_point"] = std::shared_ptr<Conv2d>(new Conv2d(point_dim, hidden_size, 1));
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
protected:
    int64_t input_size  = -1;
    int64_t patch_size  = 1;
    int64_t in_channels = 32;
    int64_t hidden_size = 1152;
    int64_t depth       = 0;

    void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, std::string prefix = "") {
        {
            enum ggml_type wtype   = GGML_TYPE_F32;  //(tensor_types.find(prefix + "caption_norm") != tensor_types.end()) ? tensor_types[prefix + "caption_norm"] : GGML_TYPE_F32;
            params["caption_norm"] = ggml_new_tensor_1d(ctx, wtype, hidden_size);
        }
        {
            enum ggml_type wtype        = GGML_TYPE_F32;  //(tensor_types.find(prefix + "scale_shift_table") != tensor_types.end()) ? tensor_types[prefix + "scale_shift_table"] : GGML_TYPE_F32;
            params["scale_shift_table"] = ggml_new_tensor_2d(ctx, wtype, hidden_size, 2);
        }
    }

public:
    LinDiT(std::map<std::string, enum ggml_type>& tensor_types) {
        for (auto pair : tensor_types) {
            std::string tensor_name = pair.first;
            if (tensor_name.find("model.diffusion_model.") == std::string::npos)
                continue;
            size_t tb = tensor_name.find("transformer_blocks.");
            if (tb != std::string::npos) {
                tensor_name     = tensor_name.substr(tb);  // remove prefix
                int block_depth = atoi(tensor_name.substr(13, tensor_name.find(".", 13)).c_str());
                if (block_depth + 1 > depth) {
                    depth = block_depth + 1;
                }
            }
        }
        LOG_INFO("SANA layers: %d", depth);

        // Same as SD3.x, maybe it's worng
        blocks["patch_embed"] = std::shared_ptr<GGMLBlock>(new PatchEmbed(input_size, patch_size, in_channels, hidden_size, true));
        // time_embed
        // time_embed.emb
        // time_embed.emb.timestep_embedder.linear_1.bias	[1 152]
        // time_embed.emb.timestep_embedder.linear_1.weight	[1 152, 256]
        // time_embed.emb.timestep_embedder.linear_2.bias	[1 152]
        // time_embed.emb.timestep_embedder.linear_2.weight	[1 152, 1 152]
        // time_embed.linear
        // time_embed.linear.bias	[6 912]
        // time_embed.linear.weight	[6 912, 1 152]
        blocks["time_embed"] = std::shared_ptr<GGMLBlock>(/*TODO*/);

        // caption_projection
        // caption_projection.linear_1
        // caption_projection.linear_1.bias	[1 152]
        // caption_projection.linear_1.weight	[1 152, 2 304]
        // caption_projection.linear_2
        // caption_projection.linear_2.bias	[1 152]
        // caption_projection.linear_2.weight	[1 152, 1 152]
        blocks["caption_projection"] = std::shared_ptr<GGMLBlock>(/*TODO*/);

        blocks["proj_out"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, in_channels));
        for (int i = 0; i < depth; i++) {
            blocks["transformer_blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new LinearTransformerBlock(hidden_size));
        }
    }

    struct ggml_tensor*
    forward(struct ggml_context* ctx,
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