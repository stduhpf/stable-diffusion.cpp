#ifndef __HY_HPP__
#define __HY_HPP__

#include "ggml_extend.hpp"
#include "model.h"

#include "mmdit.hpp"

#define HY_GRAPH_SIZE 10240

namespace Hunyuan {

    // From PixArtAlpha (todo: move it to a PixArt header file?)
    struct TextProjection : public GGMLBlock {
        enum act_fns { act_gelu_tanh,
                       act_silu,
                       act_silu_fp32 };
        act_fns act;
        TextProjection(int64_t in_features, int64_t hidden_size, int64_t out_features = -1, std::sting act_fn = "gelu_tanh") {
            if (out_features < 0)
                out_features = hidden_size;
            blocks["linear_1"] = std::shared_ptr<GGMLBlock>(new Linear(in_features, hidden_size, true));
            if (act_fn == "gelu_tanh") {
                act = act_gelu_tanh;
            } else if (act_fn == "silu") {
                act = act_silu;
            } else if (act_fn == "silu_fp32") {
                act = act_silu_fp32;
            } else {
                LOG_ERROR("Unexpected activation function: %s", act_fn.c_str());
            }
            blocks["linear_2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, out_features, true));
        }

        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* caption) {
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
            auto linear_2 = std::dynamic_pointer_cast<Linear>(blocks["linear_2"]);

            struct ggml_tensor* hidden_states = linear_1->forward(ctx, caption);

            // TODO: activation functions
            if (act == act_gelu_tanh) {
                // Default gelu is tanh approx in ggml
                hidden_states = ggml_gelu_inplace(ctx, hidden_states);
            } else if (act == act_silu) {
                hidden_states = ggml_silu_inplace(ctx, hidden_states);
            } else if (act == act_silu_fp32) {
                // TODO : force cast to fp32?
                hidden_states = ggml_silu_inplace(ctx, hidden_states);
            }

            hidden_states = linear_2->forward(ctx, hidden_states);
            return hidden_states;
        }
    };

    struct TimestepEmbedding : GGMLBlock {
        bool cond_proj = false;
        TimestepEmbedding(int64_t in_channels,
                          int64_t time_embed_dim,
                          std::string act_fn      = "silu",
                          int64_t out_dim         = -1,
                          std::string post_act_fn = "",
                          int64_t cond_proj_dim   = -1,
                          bool sample_proj_bias   = true) : cond_proj(cond_proj_dim > 0) {
            if (act_fn != "silu") {
                LOG_ERROR("Unexpected activation function: %s", act_fn.c_str());
            }
            blocks["linear_1"] = std::shared_ptr<GGMLBlock>(new Linear(in_channels, time_embed_dim, sample_proj_bias));
            if (cond_proj) {
                blocks["cond_proj"] = std::shared_ptr<GGMLBlock>(new Linear(cond_proj_dim, in_channels, false));
            }
            blocks["linear_2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, out_dim > 0 ? out_dim : time_embed_dim, sample_proj_bias));
            if (post_act_fn != "" && post_act_fn != "none") {
                LOG_ERROR("Unexpected post activation function: %s", act_fn.c_str());
            }
        }

        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* sample, struct ggml_tensor* condition = NULL) {
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
            auto linear_2 = std::dynamic_pointer_cast<Linear>(blocks["linear_2"]);
            if (condition && cond_proj) {
                auto cond_proj = std::dynamic_pointer_cast<Linear>(blocks["cond_proj"]);
                sample         = ggml_add_inplace(ctx, sample, cond_proj->forward(condition));
            }
            sample = linear_1->forward(sample);

            // assuming act is always "silu"
            sample = ggml_silu_inplace(ctx, sample);

            sample = linear_2->forward(ctx, sample);
            // assuming post_act is never set
            return sample;
        }
    };

    struct AttentionPool : GGMLBlock {
        int64_t embed_dim;
        int64_t special_dim;
        int num_heads;
        void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, std::string prefix = "") {
            enum ggml_type wtype            = (tensor_types.find(prefix + "positional_embedding") != tensor_types.end()) ? tensor_types[prefix + "positional_embedding"] : GGML_TYPE_F32;
            params["positional_embedding "] = ggml_new_tensor_2d(ctx, wtype, embed_dim, special_dim + 1);
        }
        AttentionPool(int64_t spacial_dim, int64_t embed_dim, int num_heads, int64_t output_dim = -1) : special_dim(special_dim), embed_dim(embed_dim), num_heads(num_heads) {
            // TODO
            blocks["k_proj"] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim));
            blocks["q_proj"] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim));
            blocks["v_proj"] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim));
            blocks["c_proj"] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, output_dim > 0 ? output_dim : embed_dim));
        }

        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
            auto k_proj                 = std::dynamic_pointer_cast<Linear>(blocks["k_proj"]);
            auto q_proj                 = std::dynamic_pointer_cast<Linear>(blocks["q_proj"]);
            auto v_proj                 = std::dynamic_pointer_cast<Linear>(blocks["v_proj"]);
            auto c_proj                 = std::dynamic_pointer_cast<Linear>(blocks["c_proj"]);
            struct ggml_tensor* pos_emb = params["positional_embedding"];

            const int64_t C = x->ne[0];                       // embed_dim
            const int64_t L = x->ne[1];                       // spacial_dim
            const int64_t N = (x->ne[2] > 0) ? x->ne[2] : 1;  // batch_size

            struct ggml_tensor* x_p             = ggml_permute(ctx, x, 0, 2, 1, 3);
            struct ggml_tensor* x_p_reshaped    = ggml_reshape_2d(ctx, x_p, L, C * N);
            struct ggml_tensor* x_mean_reshaped = ggml_mean(ctx, x_p_reshaped);
            struct ggml_tensor* x_mean          = ggml_reshape_3d(ctx, x_mean_reshaped, C, N, 1);

            x = ggml_concat(ctx, x_mean, x_p, 0);
            x = ggml_add_inplace(ctx, x, ggml_repeat(ctx, pos_emb, x));

            struct ggml_tensor* q = q_proj->forward(ctx, x);
            struct ggml_tensor* k = k_proj->forward(ctx, x);
            struct ggml_tensor* v = v_proj->forward(ctx, x);

            x = ggml_nn_attention_ext(ctx, q, k, v, num_heads);

            x = c_proj->forward(ctx, x);
            return x;
        }
    };

    struct CombinedTimestepTextSizeStyleEmbedding : GGMLBlock {
        int64_t embedding_dim;
        bool style_embedder;
        void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, std::string prefix = "") {
            if (style_embedder) {
                enum ggml_type wtype     = (tensor_types.find(prefix + "style_embedder") != tensor_types.end()) ? tensor_types[prefix + "style_embedder"] : GGML_TYPE_F32;
                params["style_embedder"] = ggml_new_tensor_2d(ctx, wtype, embedding_dim, 1);
            }
        }
        CombinedTimestepTextSizeStyleEmbedding(int64_t embedding_dim,
                                               int64_t pooled_projection_dim           = 1024,
                                               int64_t seq_len                         = 256,
                                               int64_t cross_attention_dim             = 2048,
                                               bool use_style_cond_and_image_meta_size = true) : embedding_dim(embedding_dim), style_embedder(use_style_cond_and_image_meta_size) {
            blocks["timestep_embedder"] = std::shared_ptr<GGMLBlock>(new TimestepEmbedding(256, embedding_dim));
            blocks["pooler"]            = std::shared_ptr<GGMLBlock>(new AttentionPool(seq_len, cross_attention_dim, 8, pooled_projection_dim));

            int64_t extra_in_dim = pooled_projection_dim;
            if (use_style_cond_and_image_meta_size) {
                extra_in_dim = 256 * 6 + embedding_dim + pooled_projection_dim;
            }
            blocks["extra_embedder"] = std::shared_ptr<GGMLBlock>(new TextProjection(extra_in_dim, embedding_dim * 4, embedding_dim, "silu_fp32"));
        }

        // Generate timesteps_proj and image_meta_size before
        struct ggml_tensor* forward(struct ggml_context* ctx,
                                    struct ggml_tensor* timesteps_proj,
                                    struct ggml_tensor* encoder_hidden_states,
                                    struct ggml_tensor* image_meta_size = NULL,
                                    struct ggml_tensor* style           = NULL) {
            auto timestep_embedder = std::dynamic_pointer_cast<TimestepEmbedding>(blocks["timestep_embedder"]);
            auto pooler            = std::dynamic_pointer_cast<AttentionPool>(blocks["pooler"]);
            auto extra_embedder    = std::dynamic_pointer_cast<TextProjection>(blocks["extra_embedder"]);

            struct ggml_tensor* timestep_emb = timestep_embedder->forward(ctx, timesteps_proj);

            struct ggml_tensor* pooled_projections = pooler->forward(ctx, encoder_hidden_states);

            struct ggml_tensor* extra_cond = pooled_projections;
            if (style_embedder) {
                if (image_meta_size) {
                    extra_cond = ggml_concat(ctx, extra_cond, image_meta_size, 0);
                }

                if (style) {
                    struct ggml_tensor* style_embedding = ggml_get_rows(ctx, params["style_embedder"], style);
                    extra_cond                          = ggml_concat(ctx, extra_cond, style_embedding, 0);
                }
            }

            struct ggml_tensor* conditioning = ggml_add_inplace(ctx, timestep_emb, extra_cond);
            return conditioning;
        }
    };

    struct AdaLayerNormContinuous : GGMLBlock {
        AdaLayerNormContinuous(int64_t embedding_dim, int64_t conditioning_embedding_dim, bool elementwise_affine = true, float eps = 1e-5, bool bias = true, std::string norm_type = "layer_norm") {
            if (norm_type == "layer_norm") {
                blocks["norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(embedding_dim, eps, elementwise_affine, bias));
            } else if (norm_type == "rms_norm") {
                blocks["norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(embedding_dim, eps));
            } else {
                LOG_ERROR("Unexpected norm type: %s", norm_type.c_str());
            }
            blocks["linear"] = std::shared_ptr<GGMLBlock>(new Linear(conditioning_embedding_dim, embedding_dim * 2, bias));
        }
        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* conditioning) {
            auto norm   = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm"]);
            auto linear = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            // emb = self.linear(self.silu(conditioning_embedding).to(x.dtype)).
            auto emb = ggml_silu_inplace(ctx, conditioning);
            emb      = linear->forward(ctx, emb);
            // scale, shift = torch.chunk(emb, 2, dim=1)
            auto scale = ggml_cont(ctx, ggml_view_2d(ctx, emb, emb->ne[0] / 2, emb->ne[1], emb->nb[1], 0));
            auto shift = ggml_cont(ctx, ggml_view_2d(ctx, emb, emb->ne[0] / 2, emb->ne[1], emb->nb[1], emb->ne[0] / 2 * ggml_type_size(emb->type)));
            // x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
            x = norm->forward(ctx, x);
            x = ggml_add_inplace(ctx, ggml_mul(ctx, x, ggml_add(ctx, ggml_new_f32(ctx, 1.0f), scale)), shift);
            return x;
        }
    };

    struct AdaLayerNormShift : GGMLBlock {
        AdaLayerNormShift(int64_t embedding_dim, ool elementwise_affine = true, float eps = 1e-6) {
            blocks["norm"]   = std::shared_ptr<GGMLBlock>(new LayerNorm(embedding_dim, eps, elementwise_affine));
            blocks["linear"] = std::shared_ptr<GGMLBlock>(new Linear(embedding_dim, embedding_dim));
        }

        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* emb) {
            auto norm   = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm"]);
            auto linear = std::dynamic_pointer_cast<Linear>(blocks["linear"]);

            // shift = self.linear(self.silu(emb).to(torch.float32)).to(emb.dtype)
            auto shift = linear->forward(ctx, ggml_silu_inplace(ctx, emb));

            // x = self.norm(x) + shift.unsqueeze(dim=1)
            x = norm->forward(ctx, x);
            x = ggml_add_inplace(ctx, x, ggml_view_2d(ctx, shift, 1, shift->ne[1], shift->nb[1], 0));

            return x;
        }
    };

    struct Attention : public MultiheadAttention {
        bool use_qk_norm = true;
        Attention(
            int query_dim,
            int cross_attention_dim             = -1,
            int heads                           = 8,
            int kv_heads                        = -1,
            int dim_head                        = 64,
            float dropout                       = 0.0,
            bool bias                           = false,
            bool upcast_attention               = false,
            bool upcast_softmax                 = false,
            std::string cross_attention_norm    = "",
            int cross_attention_norm_num_groups = 32,
            std::string qk_norm                 = "",
            // int added_kv_proj_dim               = -1,
            bool added_proj_bias             = true,
            int norm_num_groups              = -1,
            int spatial_norm_dim             = -1,
            bool out_bias                    = true,
            bool scale_qk                    = true,
            bool only_cross_attention        = false,
            float eps                        = 1e-5,
            float rescale_output_factor      = 1.0,
            bool residual_connection         = false,
            bool _from_deprecated_attn_block = false,
            // AttnProcessor* processor            = nullptr,
            int out_dim         = -1,
            int out_context_dim = -1,
            // bool context_pre_only   = false,
            bool pre_only           = false,
            bool elementwise_affine = true,
            bool is_causal          = false) {
            // TODO: support processor inside this?
            float inner_dim     = out_dim > 0 ? out_dim : dim_head * heads;
            float inner_kv_dim  = kv_heads > 0 ? dim_head * kv_heads : inner_dim;
            cross_attention_dim = cross_attention_dim > 0 ? cross_attention_dim : query_dim;
            out_dim             = out_dim > 0 ? out_dim : query_dim;

            if (qk_norm == "layer_norm") {
                blocks["norm_q"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim_head, eps, elementwise_affine));
                blocks["norm_k"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim_head, eps, elementwise_affine));
            } else {
                // TODO support other norms?
                use_qk_norm = false;
            }
            // TODO: norm_cross?

            blocks["to_q"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, bias));
            if (!only_cross_attention) {
                blocks["to_k"] = std::shared_ptr<GGMLBlock>(new Linear(cross_attention_dim, inner_kv_dim, bias));
                blocks["to_v"] = std::shared_ptr<GGMLBlock>(new Linear(cross_attention_dim, inner_kv_dim, bias));
            }

            // if (added_kv_proj_dim > 0) {
            //     // TODO add_k_proj, add_v_proj
            //     if (!context_pre_only) {
            //         // TODO add_q_proj
            //     }
            // }

            if (!pre_only) {
                blocks["to_out.0"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, out_dim, bias));
                // REMINDER: dropout
            }

            // if (!context_pre_only) {
            //     // TODO to_add_out
            // }

            // TODO: norm_added
        }
    };

    struct DiTBlock : public GGMLBlock {
        bool skip_connexion = false;
        DiTBlock(int64_t dim,
                 int64_t num_attention_heads,
                 int64_t cross_attention_dim  = 1024,
                 dropout                      = 0.0,
                 str activation_fn            = "geglu",
                 bool norm_elementwise_affine = true,
                 float norm_eps               = 1e-6,
                 bool final_dropout           = false,
                 int64_t ff_inner_dim         = -1,
                 bool ff_bias                 = true,
                 bool skip                    = false,
                 bool qk_norm                 = true) : skip_connexion(skip) {
            // Self-attn
            // TODO
            blocks["norm1"] = std::shared_ptr<GGMLBlock>(new AdaLayerNormShift(dim, norm_elementwise_affine, norm_eps));
            blocks["attn1"] = std::shared_ptr<GGMLBlock>(new Attention());

            // Cross-attn
            blocks["norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, norm_eps, norm_elementwise_affine));
            blocks["attn2"] = std::shared_ptr<GGMLBlock>(new Attention());

            // Feed-forward
            blocks["norm3"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, norm_eps, norm_elementwise_affine));

            if (ff_inner_dim < 0) {
                ff_inner_dim = dim * 4;
            }
            blocks["ff.net.0.proj"] = std::shared_ptr<GGMLBlock>(new Linear(dim, ff_inner_dim));
            blocks["ff.net.2"]      = std::shared_ptr<GGMLBlock>(new Linear(ff_inner_dim, dim));

            if (skip) {
                blocks["skip_norm"]   = std::shared_ptr<GGMLBlock>(new LayerNorm(2 * dim, norm_eps, true));
                blocks["skip_linear"] = std::shared_ptr<GGMLBlock>(new Linear(2 * dim, dim));
            }
        }
    };

    struct DiT : public GGMLBlock {
        // "activation_fn": "gelu-approximate",
        // "attention_head_dim": 88,
        // "cross_attention_dim": 1024,
        // "cross_attention_dim_t5": 2048,
        // "hidden_size": 1408,
        // "in_channels": 4,
        // "learn_sigma": true,
        // "mlp_ratio": 4.3637,
        // "norm_type": "layer_norm",
        // "num_attention_heads": 16,
        // "num_layers": 40,
        // "patch_size": 2,
        // "pooled_projection_dim": 1024,
        // "sample_size": 128,
        // "text_len": 77,
        // "text_len_t5": 256,
        // "use_style_cond_and_image_meta_size": false
        int num_layers          = 40;
        int64_t hidden_size     = 1408;
        int in_channels         = 4;
        bool learn_sigma        = true;
        float mlp_ratio         = 4.3637;
        int num_attention_heads = 16;
        int patch_size          = 2;
        int64_t sample_size     = 128;

        int64_t attention_head_dim     = 88;
        int64_t cross_attention_dim    = 1024;
        int64_t cross_attention_dim_t5 = 2048;
        int64_t pooled_projection_dim  = 1024;

        int64_t text_len    = 77;
        int64_t text_len_t5 = 256;

        // "activation_fn": "gelu-approximate",
        // "norm_type": "layer_norm",
        bool use_style_cond_and_image_meta_size = false;

        void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "") {
            ggml_type wtype                  = (tensor_types.find(prefix + "text_embedding_padding") != tensor_types.end()) ? tensor_types[prefix + "text_embedding_padding"] : GGML_TYPE_F32;
            params["text_embedding_padding"] = ggml_new_tensor_2d(ctx, wtype, cross_attention_dim, text_len + text_len_t5);
        }

        DiT() {
            // TODO: IP-Adapter support?

            int64_t out_channels = learn_sigma ? in_channels * 2 : in_channels;
            int64_t inner_dim    = num_attention_heads * attention_head_dim;

            // TODO: Sincos pos embeds
            blocks["pos_embed"] = std::shared_ptr<GGMLBlock>(new PatchEmbed(sample_size, patch_size, in_channels, hidden_size));

            blocks["text_embedder"] = std::shared_ptr<GGMLBlock>(new TextProjection(cross_attention_dim_t5, cross_attention_dim_t5 * 4, cross_attention_dim, "silu_fp32"));

            for (int i = 0; i < num_layers; i++) {
                // TODO : blocks
                blocks["blocks." + std::to_string(i)] = td::shared_ptr<GGMLBlock>(new DiTBlock());
            }
            blocks["norm_out"] = std::shared_ptr<GGMLBlock>(new AdaLayerNormContinuous(inner_dim, inner_dim, false, 1e-6));
            blocks["proj_out"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, patch_size * patch_size * out_channels, true));

            blocks["time_extra_emb"] = std::shared_ptr<GGMLBlock>(new CombinedTimestepTextSizeStyleEmbedding(hidden_size, pooled_projection_dim, text_len_t5, cross_attention_dim_t5, use_style_cond_and_image_meta_size));
        }

        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* timesteps, struct ggml_tensor* encoder_hidden_states) {
            // TODO
            return NULL;
        }
    };

    struct DiTRunner : public GGMLRunner {
        DiT model;
        DiTRunner(ggml_backend_t backend,
                  std::map<std::string, enum ggml_type>& tensor_types,
                  const std::string prefix = "",
                  bool flash_attn          = false) : GGMLRunner(backend) {
        }
    };
}

#endif  // __HY_HPP__