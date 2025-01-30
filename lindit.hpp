#ifndef __LINDIT_HPP__
#define __LINDIT_HPP__

#include "ggml_extend.hpp"
#include "model.h"

// TODO: estimate actual size
#define LINDIT_GRAPH_SIZE 10240

namespace Sana {

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

    struct TimeEmbed : GGMLBlock {
        struct Emb : GGMLBlock {
        public:
            Emb(int64_t frequency_embedding_size = 256, int64_t hidden_size = 1152) {
                blocks["linear_1"] = std::shared_ptr<GGMLBlock>(new Linear(frequency_embedding_size, hidden_size, true));
                blocks["linear_2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size, true));
            }
            // TODO : frequency embeddings
            //  https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py#L87

            // TODO: forward: frequency embeddings (cos/sin) -> l1 -> SiLU -> l2
        };

    public:
        TimeEmbed(int64_t frequency_embedding_size = 256, int64_t hidden_size = 1152, int ratio = 6) {
            blocks["emb.timestep_embedder"] = std::shared_ptr<GGMLBlock>(new Emb(frequency_embedding_size, hidden_size));
            blocks["linear"]                = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size * ratio, true));
        }

        // TODO: forward
        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* t) {
            // emb
            // silu
            // linear

            return t;
        }
    };

    struct CaptionProj : GGMLBlock {
    public:
        CaptionProj(int64_t caption_dim = 2304, int64_t hidden_size = 1152) {
            blocks["linear_1"] = std::shared_ptr<GGMLBlock>(new Linear(caption_dim, hidden_size, true));
            blocks["linear_2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size, true));
        }

        // TODO: forward (GELU(approximate="tanh"))
    };

    struct LinearAttentionHead : public GGMLBlock {
    public:
        LinearAttentionHead(int64_t dim, bool qkv_bias = false, bool output_bias = true) {
            blocks["to_q"]     = std::shared_ptr<GGMLBlock>(new Linear(dim, qkv_bias));
            blocks["to_k"]     = std::shared_ptr<GGMLBlock>(new Linear(dim, qkv_bias));
            blocks["to_v"]     = std::shared_ptr<GGMLBlock>(new Linear(dim, qkv_bias));
            blocks["to_out.0"] = std::shared_ptr<GGMLBlock>(new Linear(dim, output_bias));
        }
        // TODO: forward
    };

    struct MixFFN : public GGMLBlock {
    public:
        MixFFN(int64_t hidden_size = 1152, int64_t ffn_ratio = 5, int64_t point_downscale = 2) {
            int64_t mid_dim   = hidden_size * ffn_ratio;
            int64_t point_dim = mid_dim / point_downscale;
            // inverted residual block
            blocks["conv_inverted"] = std::shared_ptr<GGMLBlock>(new Conv2d(hidden_size, mid_dim, {1, 1}));
            // Convolution
            blocks["conv_depth"] = std::shared_ptr<GGMLBlock>(new Conv2d(1, mid_dim, {3, 3}, {1, 1}, {0, 0}, {1, 1}, true));
            // residual block
            blocks["conv_point"] = std::shared_ptr<GGMLBlock>(new Conv2d(point_dim, hidden_size, {1, 1}));
        }
        // TODO: forwards
    };

    struct LinearTransformerBlock : public GGMLBlock {
    protected:
        int64_t hidden_size;

        void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "") {
            ggml_type wtype             = (tensor_types.find(prefix + "scale_shift_table") != tensor_types.end()) ? tensor_types[prefix + "scale_shift_table"] : GGML_TYPE_F32;
            params["scale_shift_table"] = ggml_new_tensor_2d(ctx, wtype, hidden_size, 6);
        }

    public:
        LinearTransformerBlock(int64_t dim)
            : hidden_size(dim) {
            // TODO: init
            blocks["attn1"] = std::shared_ptr<GGMLBlock>(new LinearAttentionHead(dim));
            blocks["attn2"] = std::shared_ptr<GGMLBlock>(new LinearAttentionHead(dim, true));
            blocks["ff"]    = std::shared_ptr<GGMLBlock>(new MixFFN(dim));
        }
        // TODO: forward
        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* t, struct ggml_tensor* y) {
            return x;
        }
    };

    struct LinDiT : public GGMLBlock {
    protected:
        int64_t input_size               = -1;
        int64_t patch_size               = 1;
        int64_t in_channels              = 32;
        int64_t hidden_size              = 1152;
        int64_t depth                    = 0;
        int64_t frequency_embedding_size = 256;
        int64_t caption_size             = 2304;

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

            blocks["time_embed"] = std::shared_ptr<GGMLBlock>(new TimeEmbed(frequency_embedding_size, hidden_size));

            blocks["caption_projection"] = std::shared_ptr<GGMLBlock>(new CaptionProj(caption_size, hidden_size));

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
                std::vector<int> skip_layers = std::vector<int>()) {
            // TODO
            auto x_embedder = std::dynamic_pointer_cast<PatchEmbed>(blocks["patch_embed"]);
            auto t_embedder = std::dynamic_pointer_cast<TimeEmbed>(blocks["time_embed"]);
            auto y_embedder = std::dynamic_pointer_cast<CaptionProj>(blocks["caption_projection"]);

            // NoPE: No Positionnal Encoder
            x = x_embedder->forward(ctx, x);

            auto c = t_embedder->forward(ctx, t);
            y      = t_embedder->forward(ctx, y);

            for (int i = 0; i < depth; i++) {
                if (skip_layers.size() > 0 && std::find(skip_layers.begin(), skip_layers.end(), i) != skip_layers.end()) {
                    continue;
                }
                auto block = std::dynamic_pointer_cast<LinearTransformerBlock>(blocks["transformer_blocks." + std::to_string(i)]);
                x          = block->forward(ctx, x, t, y);
            }

            return x;
        }
    };

    struct LinDiTRunner : public GGMLRunner {
        LinDiT model;
        static std::map<std::string, enum ggml_type> empty_tensor_types;

        LinDiTRunner(ggml_backend_t backend,
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
                                        struct ggml_tensor* y,
                                        std::vector<int> skip_layers = std::vector<int>()) {
            struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, LINDIT_GRAPH_SIZE, false);

            x         = to_backend(x);
            timesteps = to_backend(timesteps);
            y         = to_backend(y);

            struct ggml_tensor* out = model.forward(compute_ctx,
                                                    x,
                                                    timesteps,
                                                    y,
                                                    skip_layers);

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        void compute(int n_threads,
                     struct ggml_tensor* x,
                     struct ggml_tensor* timesteps,
                     struct ggml_tensor* y,
                     struct ggml_tensor** output     = NULL,
                     struct ggml_context* output_ctx = NULL,
                     std::vector<int> skip_layers    = std::vector<int>()) {
            // x: [N, in_channels, h, w]
            // timesteps: [N, ]
            // context: [N, max_position, hidden_size]([N, 154, 4096]) or [1, max_position, hidden_size]
            // y: [N, adm_in_channels] or [1, adm_in_channels]
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_graph(x, timesteps, y, skip_layers);
            };

            GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
        }
    };
}  // namespace Sana

#endif