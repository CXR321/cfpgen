
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


from typing import Optional
from typing import List
import torch
import torch.nn as nn
from byprot.models import register_model
from byprot.datamodules.dataset.tokenized_protein import DPLM2Tokenizer
from torch.nn import functional as F
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from transformers.models.esm.modeling_esm import *
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from copy import deepcopy
import random

class ModifiedRotaryEmbedding(RotaryEmbedding):
    """Rotary position embeddings based on those in.

    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    def __init__(self, dim: int):
        super().__init__(dim)
        self.aa_type = 1
        self.struct_type = 0

    def _update_cos_sin_tables(self, x, type_ids, seq_dimension=2):
        seq_len = x.shape[seq_dimension]
        if self.aa_type in type_ids and self.struct_type in type_ids:
            seq_len /= 2

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (
            seq_len != self._seq_len_cached
            or self._cos_cached.device != x.device
        ):
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(
                self.inv_freq
            )
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, type_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(
            k, type_ids=type_ids, seq_dimension=-2
        )

        if self.aa_type in type_ids and self.struct_type in type_ids:
            q_1, q_2 = q.chunk(2, dim=-2)
            k_1, k_2 = k.chunk(2, dim=-2)
            q_1 = apply_rotary_pos_emb(q_1, self._cos_cached, self._sin_cached)
            q_2 = apply_rotary_pos_emb(q_2, self._cos_cached, self._sin_cached)
            k_1 = apply_rotary_pos_emb(k_1, self._cos_cached, self._sin_cached)
            k_2 = apply_rotary_pos_emb(k_2, self._cos_cached, self._sin_cached)
            q = torch.cat((q_1, q_2), dim=-2)
            k = torch.cat((k_1, k_2), dim=-2)
            return (q, k)
        else:
            return (
                apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
                apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
            )

class ModifiedEsmSelfAttention(EsmSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)
        self.rotary_embeddings = ModifiedRotaryEmbedding(
            dim=self.attention_head_size
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(
                self.key(encoder_hidden_states)
            )
            value_layer = self.transpose_for_scores(
                self.value(encoder_hidden_states)
            )
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Matt: Our BERT model (which this code was derived from) scales attention logits down by sqrt(head_dim).
        # ESM scales the query down by the same factor instead. Modulo numerical stability these are equivalent,
        # but not when rotary embeddings get involved. Therefore, we scale the query here to match the original
        # ESM code and fix rotary embeddings.
        query_layer = query_layer * self.attention_head_size**-0.5

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = self.rotary_embeddings(
                query_layer, key_layer, type_ids
            )

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            raise NotImplementedError

        # if attention_mask is not None:
        #     # Apply the attention mask is (precomputed for all layers in EsmModel forward() function)
        #     attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            raise NotImplementedError

        query_layer = query_layer.contiguous()
        key_layer = key_layer.contiguous()
        value_layer = value_layer.contiguous()
        # start_time = time.time()
        context_layer = F.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            scale=1.0,
        )
        # end_time = time.time()
        # print('FlashAttn: ', start_time - end_time)

        # context_layer = torch.matmul(attention_probs, value_layer)
        # start_time = time.time()
        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # attention_scores = attention_scores + attention_mask
        # attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # attention_probs = self.dropout(attention_probs)
        # context_layer = torch.matmul(attention_probs, value_layer)
        # end_time = time.time()
        # print('Naive impl.: ', start_time - end_time)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size,
        )
        context_layer = context_layer.view(new_context_layer_shape)

        # outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        outputs = (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class RCFEBlock(nn.Module):
    def __init__(self, base_block=None, block_index=None, hidden_size=None):
        super().__init__()

        self.copied_block = deepcopy(base_block)
        self.block_index = block_index
        self.hidden_size = hidden_size

        if self.block_index == 0:
            self.before_proj = nn.Linear(hidden_size, hidden_size)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)

        self.after_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)


    def forward(self, x, mask=None, c=None, y=None):

        if self.block_index == 0:
            c = self.before_proj(c)
            c = self.copied_block(x + c, mask, cond_input=y)[0]
            c_skip = self.after_proj(c)
        else:
            c = self.copied_block(c, mask, cond_input=y)[0]
            c_skip = self.after_proj(c)

        return c, c_skip


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def modulate_diff(x, shift, scale):
    b, n, hid = x.shape

    # print(f"x.shape: {x.shape}")
    # print(f"shift.shape: {shift[0].shape}")
    # print(f"scale.shape: {scale[0].shape}")
    n_half = n // 2  # 假设 n 是偶数

    # 生成一个 (b, n, hid) 的掩码，标记前半部分为1，后半部分为0
    mask = torch.zeros_like(x)
    mask[:, :n_half, :] = 1  # 前半部分为1，后半部分为0

    # 分别计算前后两部分的调制结果
    modulated = (
        (x * (1 + scale[0].unsqueeze(1)) + shift[0].unsqueeze(1))  # 前半部分调制
    ) * mask + (
        (x * (1 + scale[1].unsqueeze(1)) + shift[1].unsqueeze(1))  # 后半部分调制
    ) * (1 - mask)

    return modulated


class AGFMSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, gate=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if gate is not None:
            if isinstance(gate, list):  # 如果 gate 是 list，则分段处理
                gate1, gate2 = gate  # 拆分成两部分
                n = hidden_states.shape[1]  # 序列长度
                n_half = n // 2  # 前半部分长度

                # print(f"gate1.shape: {gate1.shape}")
                # print(f"gate2.shape: {gate2.shape}")
                # print(f"hidden_states.shape: {hidden_states.shape}")
                # print(f"input_tensor.shape: {input_tensor.shape}")
                    
                # 生成前后半部分的 Mask (形状: (1, n, 1))
                mask = torch.zeros_like(hidden_states)
                mask[:, :n_half, :] = 1  # 前半部分为 1，后半部分为 0

                # 计算前后半部分的调制结果
                modulated_part1 = gate1.unsqueeze(1) * hidden_states + input_tensor
                modulated_part2 = gate2.unsqueeze(1) * hidden_states + input_tensor

                # 合并结果（Mask 加权）
                hidden_states = mask * modulated_part1 + (1 - mask) * modulated_part2
            else:  # 如果 gate 不是 list，则全局应用
                hidden_states = gate.unsqueeze(1) * hidden_states + input_tensor
        else:
            hidden_states = hidden_states + input_tensor
        return hidden_states


class AGFMOutput(EsmOutput):
    def forward(self, hidden_states, input_tensor, gate=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if gate is not None:
            if isinstance(gate, list):  # 如果 gate 是 list，则分段处理
                gate1, gate2 = gate  # 拆分成两部分
                n = hidden_states.shape[1]  # 序列长度
                n_half = n // 2  # 前半部分长度

                # print(f"gate1.shape: {gate1.shape}")
                # print(f"gate2.shape: {gate2.shape}")
                # print(f"hidden_states.shape: {hidden_states.shape}")
                # print(f"input_tensor.shape: {input_tensor.shape}")

                # 生成前后半部分的 Mask (形状: (1, n, 1))
                mask = torch.zeros_like(hidden_states)
                mask[:, :n_half, :] = 1  # 前半部分为 1，后半部分为 0

                # 计算前后半部分的调制结果
                modulated_part1 = gate1.unsqueeze(1) * hidden_states + input_tensor
                modulated_part2 = gate2.unsqueeze(1) * hidden_states + input_tensor

                # 合并结果（Mask 加权）
                hidden_states = mask * modulated_part1 + (1 - mask) * modulated_part2
            else:  # 如果 gate 不是 list，则全局应用
                hidden_states = gate.unsqueeze(1) * hidden_states + input_tensor
        else:
            hidden_states = hidden_states + input_tensor
        return hidden_states


class AGFMAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = EsmSelfAttention(config)
        self.output = AGFMSelfOutput(config)
        self.pruned_heads = set()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            shift_msa=None,
            scale_msa=None,
            gate_msa=None,
    ):
        hidden_states_ln = self.LayerNorm(hidden_states)

        if shift_msa is not None:
            hidden_states_ln = modulate(hidden_states_ln, shift_msa, scale_msa)

        self_outputs = self.self(
            hidden_states_ln,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )

        if gate_msa is not None:
            attention_output = self.output(self_outputs[0], hidden_states, gate_msa)
        else:
            attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class AGFMAttentionDPLM2(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = ModifiedEsmSelfAttention(config)
        self.output = AGFMSelfOutput(config)
        self.pruned_heads = set()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            shift_msa=None,
            scale_msa=None,
            gate_msa=None,
            type_ids=None,
    ):
        hidden_states_ln = self.LayerNorm(hidden_states)

        # TODO: 为seq struct 设计各自的 msa (self.self / self.output)
        if shift_msa is not None:
            if isinstance(shift_msa, list):
                hidden_states_ln = modulate_diff(hidden_states_ln, shift_msa, scale_msa)
            else:
                hidden_states_ln = modulate(hidden_states_ln, shift_msa, scale_msa)

        self_outputs = self.self(
            hidden_states_ln,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            type_ids=type_ids,
        )

        if gate_msa is not None:
            attention_output = self.output(self_outputs[0], hidden_states, gate_msa)
        else:
            attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class AGFMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len_dim = 1
        self.attention = AGFMAttention(config)
        self.intermediate = EsmIntermediate(config)
        self.output = AGFMOutput(config)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True),  # use gate_msa
        )


    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            cond_input=None,
    ):

        if cond_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond_input).chunk(6, dim=1)

        else:
            shift_msa = scale_msa = shift_mlp = scale_mlp = gate_msa = gate_mlp = None

        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            shift_msa=shift_msa,
            scale_msa=scale_msa,
            gate_msa=gate_msa,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        # feed_forward_chunk: layer_norm->linear(5120)->linear(1280)
        attention_output_ln = self.LayerNorm(attention_output)
        if cond_input is not None:
            attention_output_ln = modulate(attention_output_ln, shift_mlp, scale_mlp)
            intermediate_output = self.intermediate(attention_output_ln)
            layer_output = self.output(intermediate_output, attention_output, gate_mlp)
        else:
            intermediate_output = self.intermediate(attention_output_ln)
            layer_output = self.output(intermediate_output, attention_output)

        outputs = (layer_output,) + outputs
        return outputs

class AGFMLayerDPLM2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len_dim = 1
        self.attention = AGFMAttentionDPLM2(config)
        self.intermediate = EsmIntermediate(config)
        self.output = AGFMOutput(config)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if not config.use_diff_modulation:
            self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True),  # use gate_msa
        )
        else:
            self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 12 * config.hidden_size, bias=True),  # use gate_msa
        )

        self.use_diff_modulation = getattr(config, "use_diff_modulation", False)
        self.use_func_cross_attn = getattr(config, "use_func_cross_attn", False)
        self.use_motif_struct_emb = getattr(config, "use_motif_struct_emb", False)

        # print(f"use_func_cross_attn: {self.use_func_cross_attn}")

        if self.use_func_cross_attn:

            # === 新增：功能 cross-attn 模块（F -> tokens）===
            self.func_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            self.cross_attn_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                batch_first=True,
            )
            # 残差缩放（可学，初始1.0；你也可以设为0.0实现“渐进启用”）
            self.cross_res_scale = nn.Parameter(torch.tensor(1.0))

        if self.use_motif_struct_emb:

            # === 新增：功能 motif-cross-attn 模块（F -> tokens）===
            self.motif_proj = nn.Linear(1280, config.hidden_size, bias=True)
            self.motif_cross_attn_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.motif_cross_attn = nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                batch_first=True,
            )
            # 残差缩放（可学，初始1.0；你也可以设为0.0实现“渐进启用”）
            self.motif_cross_res_scale = nn.Parameter(torch.tensor(1.0))


    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            cond_input=None,
            type_ids=None,
            motif_struct_emb=None,
    ):

        if cond_input is not None:
            if self.use_diff_modulation:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa_seq, scale_msa_seq, gate_msa_seq, shift_mlp_seq, scale_mlp_seq, gate_mlp_seq = self.adaLN_modulation(cond_input).chunk(12, dim=1)

                shift_msa = [shift_msa, shift_msa_seq]
                scale_msa = [scale_msa, scale_msa_seq]
                gate_msa = [gate_msa, gate_msa_seq]
                shift_mlp = [shift_mlp, shift_mlp_seq]
                scale_mlp = [scale_mlp, scale_mlp_seq]
                gate_mlp = [gate_mlp, gate_mlp_seq]
            else:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond_input).chunk(6, dim=1)

        else:
            shift_msa = scale_msa = shift_mlp = scale_mlp = gate_msa = gate_mlp = None

        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            shift_msa=shift_msa,
            scale_msa=scale_msa,
            gate_msa=gate_msa,
            type_ids=type_ids,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        cross_out = torch.zeros_like(attention_output)
        motif_cross_out = torch.zeros_like(attention_output)

        if self.use_func_cross_attn:
            # ====== （新增）功能 cross-attention：tokens(Q) ← F(K,V) ======
            # 作用于 AA 与 Struct 全体 token；若 use_diff_modulation=True，用 type_ids 做门控区分
            if cond_input is not None:
                # cond_input: [B, D] or [B, 1, D]
                if cond_input.dim() == 2:
                    func_tok = cond_input.unsqueeze(1)     # [B, 1, D]
                else:
                    func_tok = cond_input                  # 允许 [B, 1, D]

                func_tok = self.func_proj(func_tok)        # 线性投影到 d_model

                # 预归一化 + CrossAttn
                q = self.cross_attn_ln(attention_output)           # [B, L, D]
                cross_out, cross_w = self.cross_attn(
                    query=q, key=func_tok, value=func_tok,
                    need_weights=output_attentions,
                    attn_mask=None, key_padding_mask=None
                )  # cross_out: [B, L, D]

                # 残差 + 门控（AA/Struct 差异化）
                # attention_output = attention_output + self.cross_res_scale * cross_out

                if output_attentions:
                    outputs = (cross_w,) + outputs  # 可选：把功能 cross-attn 的权重也返回，便于可视化
            # else: 不传 cond_input 就不做功能 cross-attn

        # feed_forward_chunk: layer_norm->linear(5120)->linear(1280)
        # attention_output_ln = self.LayerNorm(attention_output)

        if self.use_motif_struct_emb:
            # print(motif_struct_emb)
            # exit()
            # ====== （新增）功能 cross-attention：tokens(Q) ← F(K,V) ======
            # 作用于 AA 与 Struct 全体 token；若 use_diff_modulation=True，用 type_ids 做门控区分
            if motif_struct_emb is not None:
                # cond_input: [B, D] or [B, 1, D]
                if motif_struct_emb.dim() == 2:
                    motif_tok = motif_struct_emb.unsqueeze(1)     # [B, 1, D]
                else:
                    motif_tok = motif_struct_emb                  # 允许 [B, 1, D]

                motif_tok = self.motif_proj(motif_tok)        # 线性投影到 d_model

                # 预归一化 + CrossAttn
                q = self.motif_cross_attn_ln(attention_output)           # [B, L, D]
                motif_cross_out, motif_cross_w = self.motif_cross_attn(
                    query=q, key=motif_tok, value=motif_tok,
                    need_weights=output_attentions,
                    attn_mask=None, key_padding_mask=None
                )  # cross_out: [B, L, D]

                # 残差 + 门控（AA/Struct 差异化）
                # attention_output = attention_output + self.motif_cross_res_scale * motif_cross_out

                if output_attentions:
                    outputs = (motif_cross_w,) + outputs  # 可选：把功能 cross-attn 的权重也返回，便于可视化
            # else: 不传 cond_input 就不做功能 cross-attn
        
        n_half = attention_output.size(1) // 2
        struct_mask = torch.zeros_like(attention_output)
        struct_mask[:, :n_half, :] = 1  # 前半部分为1，后半部分为0
        attention_output = attention_output + self.cross_res_scale * cross_out + self.motif_cross_res_scale * motif_cross_out * struct_mask

        # feed_forward_chunk: layer_norm->linear(5120)->linear(1280)
        attention_output_ln = self.LayerNorm(attention_output)

        if cond_input is not None:
            # TODO: use cond different from seq struct
            if self.use_diff_modulation:
                attention_output_ln = modulate_diff(attention_output_ln, shift_mlp, scale_mlp)
            else:
                attention_output_ln = modulate(attention_output_ln, shift_mlp, scale_mlp)
            intermediate_output = self.intermediate(attention_output_ln)
            layer_output = self.output(intermediate_output, attention_output, gate_mlp)
        else:
            intermediate_output = self.intermediate(attention_output_ln)
            layer_output = self.output(intermediate_output, attention_output)

        outputs = (layer_output,) + outputs
        return outputs

class FuncTagEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        use_cfg_embedding = True
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


class CFPGenEncoder(EsmEncoder):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

        self.use_go, self.use_ipr, self.use_ec = config.use_go, config.use_ipr, config.use_ec

        if self.use_go:
            self.go_class_num = config.go_num
            self.go_cls_dropout_all = config.go_drop
            self.go_cls_dropout_each = 0.1
            self.go_embedder = FuncTagEmbedder(config.go_num, config.hidden_size)

        if self.use_ipr:
            self.ipr_class_num = config.ipr_num
            self.ipr_cls_dropout_all = config.ipr_drop
            self.ipr_cls_dropout_each = 0.1
            self.ipr_embedder = FuncTagEmbedder(config.ipr_num, config.hidden_size)

        if self.use_ec:
            self.ec_class_num = config.ec_num
            self.ec_cls_dropout_all = config.ec_drop
            self.ec_cls_dropout_each = 0
            self.ec_embedder = FuncTagEmbedder(config.ec_num, config.hidden_size)


        self.layer = nn.ModuleList([AGFMLayer(deepcopy(config)) for _ in range(config.num_hidden_layers)])

        if config.use_seq_motif:
            self.copy_blocks_num = config.num_hidden_layers//2
            self.anno_dropout = 0.5
            self.seq_controlnet = nn.ModuleList(
                [RCFEBlock(AGFMLayer(deepcopy(config)), i, config.hidden_size) for i in range(self.copy_blocks_num)]
            )
        else:
            self.seq_controlnet = None


    def drop_anno_ids(self, class_tensor, embedder, class_num, training, drop_all_prob, drop_each_prob):
        """
        Drop annotation class IDs either at sample level or element level, then compute embeddings.
        """
        if training:
            # Drop all class IDs in a row with drop_all_prob
            drop_all = torch.rand(class_tensor.size(0), device=class_tensor.device) < drop_all_prob
            full_replacement = torch.full_like(class_tensor, class_num)
            class_tensor = torch.where(drop_all.unsqueeze(1), full_replacement, class_tensor)

            # Drop individual elements in class_tensor with drop_each_prob
            drop_each = torch.rand_like(class_tensor, dtype=torch.float) < drop_each_prob
            class_tensor = torch.where(drop_each, full_replacement, class_tensor)

        class_embeds = []
        for i, class_split in enumerate(class_tensor.split(1, dim=-1)):
            class_ids = class_split.squeeze(-1)
            class_embed = embedder(class_ids)
            # Zero-out embeddings where class_id == class_num (i.e., dropped)
            mask = (class_ids == class_num).unsqueeze(-1)
            class_embed = torch.where(mask, torch.zeros_like(class_embed), class_embed)
            class_embeds.append(class_embed)

        # Combine class embeddings by summation
        return torch.sum(torch.stack(class_embeds, dim=0), dim=0)


    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            **kwargs
    ):


        '''
        Annotation-Guided Feature Modulation (AGFM)
        '''

        anno_tag = kwargs.get('anno_tag')
        anno_embed = None

        if anno_tag is not None:

            go_class = anno_tag.get('go')
            ipr_class = anno_tag.get('ipr')
            ec_class = anno_tag.get('ec')

            seq_num = hidden_states.size(0)

            def prepare_class(cls, class_num):
                """Replace -1 with class_num and broadcast if needed."""
                if not self.training and cls.dim() == 1:
                    cls = cls.unsqueeze(0).repeat(seq_num, 1)
                return torch.where(cls == -1, torch.full_like(cls, class_num), cls)

            if self.use_go and go_class is not None:
                go_class = prepare_class(go_class, self.go_embedder.num_classes)
                anno_embed = self.drop_anno_ids(go_class, self.go_embedder, self.go_class_num,
                                                self.training, self.go_cls_dropout_all, self.go_cls_dropout_each)

            if self.use_ipr and ipr_class is not None:
                ipr_class = prepare_class(ipr_class, self.ipr_embedder.num_classes)
                ipr_embed = self.drop_anno_ids(ipr_class, self.ipr_embedder, self.ipr_class_num,
                                               self.training, self.ipr_cls_dropout_all, self.ipr_cls_dropout_each)
                anno_embed = anno_embed + ipr_embed if anno_embed is not None else ipr_embed

            if self.use_ec and ec_class is not None:
                ec_class = prepare_class(ec_class, self.ec_embedder.num_classes)
                ec_embed = self.drop_anno_ids(ec_class, self.ec_embedder, self.ec_class_num,
                                              self.training, self.ec_cls_dropout_all, self.ec_cls_dropout_each)
                anno_embed = anno_embed + ec_embed if anno_embed is not None else ec_embed


        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                    "`use_cache=False`..."
                )
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        '''
        Residue-Controlled Functional Encoding (RCFE)
        '''
        if self.seq_controlnet and anno_tag['seq_cond'] is not None and anno_tag['seq_cond'].numel() > 0:

            motif = anno_tag['seq_cond']
          
            random_go_embed = anno_embed if (not self.training or random.random() > self.anno_dropout) else None  # motif embedding 多大程度参考 global condition

            for index in range(1, self.copy_blocks_num + 1):
                motif, motif_skip = self.seq_controlnet[index - 1](hidden_states, attention_mask, motif, random_go_embed)
                hidden_states = self.layer[index](hidden_states+motif_skip, attention_mask, cond_input=random_go_embed)[0]

            for index in range(self.copy_blocks_num + 1, len(self.layer)):
                hidden_states = self.layer[index](hidden_states, attention_mask, cond_input=random_go_embed)[0]

        else:
            for i, layer_module in enumerate(self.layer):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i] if head_mask is not None else None
                past_key_value = past_key_values[i] if past_key_values is not None else None

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer_module.__call__,
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                        anno_embed,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = next_decoder_cache + (layer_outputs[-1],)
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
                    if self.config.add_cross_attention:
                        all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class CFPGenEncoderDPLM2(EsmEncoder):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

        self.use_go, self.use_ipr, self.use_ec = config.use_go, config.use_ipr, config.use_ec

        if self.use_go:
            self.go_class_num = config.go_num
            self.go_cls_dropout_all = config.go_drop
            self.go_cls_dropout_each = 0.1
            self.go_embedder = FuncTagEmbedder(config.go_num, config.hidden_size)

        if self.use_ipr:
            self.ipr_class_num = config.ipr_num
            self.ipr_cls_dropout_all = config.ipr_drop
            self.ipr_cls_dropout_each = 0.1
            self.ipr_embedder = FuncTagEmbedder(config.ipr_num, config.hidden_size)

        if self.use_ec:
            self.ec_class_num = config.ec_num
            self.ec_cls_dropout_all = config.ec_drop
            self.ec_cls_dropout_each = 0
            self.ec_embedder = FuncTagEmbedder(config.ec_num, config.hidden_size)


        self.layer = nn.ModuleList([AGFMLayerDPLM2(deepcopy(config)) for _ in range(config.num_hidden_layers)])

        # TODO: 初始化RCFE 
        if config.use_seq_motif and False:
            self.copy_blocks_num = config.num_hidden_layers//2
            self.anno_dropout = 0.5
            self.seq_controlnet = nn.ModuleList(
                [RCFEBlock(AGFMLayerDPLM2(deepcopy(config)), i, config.hidden_size) for i in range(self.copy_blocks_num)]
            )
        else:
            self.seq_controlnet = None


    def drop_anno_ids(self, class_tensor, embedder, class_num, training, drop_all_prob, drop_each_prob):
        """
        Drop annotation class IDs either at sample level or element level, then compute embeddings.
        """
        if training:
            # Drop all class IDs in a row with drop_all_prob
            drop_all = torch.rand(class_tensor.size(0), device=class_tensor.device) < drop_all_prob
            full_replacement = torch.full_like(class_tensor, class_num)
            class_tensor = torch.where(drop_all.unsqueeze(1), full_replacement, class_tensor)

            # Drop individual elements in class_tensor with drop_each_prob
            drop_each = torch.rand_like(class_tensor, dtype=torch.float) < drop_each_prob
            class_tensor = torch.where(drop_each, full_replacement, class_tensor)

        class_embeds = []
        for i, class_split in enumerate(class_tensor.split(1, dim=-1)):
            class_ids = class_split.squeeze(-1)
            class_embed = embedder(class_ids)
            # Zero-out embeddings where class_id == class_num (i.e., dropped)
            mask = (class_ids == class_num).unsqueeze(-1)
            class_embed = torch.where(mask, torch.zeros_like(class_embed), class_embed)
            class_embeds.append(class_embed)

        # Combine class embeddings by summation
        return torch.sum(torch.stack(class_embeds, dim=0), dim=0)


    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            type_ids=None,
            **kwargs
    ):


        '''
        Annotation-Guided Feature Modulation (AGFM)
        '''

        anno_tag = kwargs.get('anno_tag')
        anno_embed = None
        motif_struct_emb = None

        if anno_tag is not None:

            go_class = anno_tag.get('go')
            ipr_class = anno_tag.get('ipr')
            ec_class = anno_tag.get('ec')

            motif_struct_emb = anno_tag.get('motif_struct_emb')
            # print(motif_struct_emb)
            # exit()

            seq_num = hidden_states.size(0)

            def prepare_class(cls, class_num):
                """Replace -1 with class_num and broadcast if needed."""
                if not self.training and cls.dim() == 1:
                    cls = cls.unsqueeze(0).repeat(seq_num, 1)
                return torch.where(cls == -1, torch.full_like(cls, class_num), cls)

            if self.use_go and go_class is not None:
                if hasattr(self.go_embedder, 'original_module'):
                    num_classes = self.go_embedder.original_module.num_classes
                else:
                    num_classes = self.go_embedder.num_classes
                go_class = prepare_class(go_class, num_classes)
                anno_embed = self.drop_anno_ids(go_class, self.go_embedder, self.go_class_num,
                                                self.training, self.go_cls_dropout_all, self.go_cls_dropout_each)

            if self.use_ipr and ipr_class is not None:
                if hasattr(self.ipr_embedder, 'original_module'):
                    num_classes = self.ipr_embedder.original_module.num_classes
                else:
                    num_classes = self.ipr_embedder.num_classes
                ipr_class = prepare_class(ipr_class, num_classes)
                ipr_embed = self.drop_anno_ids(ipr_class, self.ipr_embedder, self.ipr_class_num,
                                               self.training, self.ipr_cls_dropout_all, self.ipr_cls_dropout_each)
                anno_embed = anno_embed + ipr_embed if anno_embed is not None else ipr_embed

            if self.use_ec and ec_class is not None:
                if hasattr(self.ec_embedder, 'original_module'):
                    num_classes = self.ec_embedder.original_module.num_classes
                else:
                    num_classes = self.ec_embedder.num_classes
                ec_class = prepare_class(ec_class, num_classes)
                ec_embed = self.drop_anno_ids(ec_class, self.ec_embedder, self.ec_class_num,
                                              self.training, self.ec_cls_dropout_all, self.ec_cls_dropout_each)
                anno_embed = anno_embed + ec_embed if anno_embed is not None else ec_embed


        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                    "`use_cache=False`..."
                )
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        '''
        Residue-Controlled Functional Encoding (RCFE)
        '''
        # TODO: 主要是处理seq struct双通道的，暂时不做
        if self.seq_controlnet and anno_tag['seq_cond'] is not None and anno_tag['seq_cond'].numel() > 0 and False:

            motif = anno_tag['seq_cond']
          
            random_go_embed = anno_embed if (not self.training or random.random() > self.anno_dropout) else None  # motif embedding 多大程度参考 global condition

            for index in range(1, self.copy_blocks_num + 1):
                motif, motif_skip = self.seq_controlnet[index - 1](hidden_states, attention_mask, motif, random_go_embed)
                hidden_states = self.layer[index](hidden_states+motif_skip, attention_mask, cond_input=random_go_embed)[0]

            for index in range(self.copy_blocks_num + 1, len(self.layer)):
                hidden_states = self.layer[index](hidden_states, attention_mask, cond_input=random_go_embed)[0]

        else:
            for i, layer_module in enumerate(self.layer):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i] if head_mask is not None else None
                past_key_value = past_key_values[i] if past_key_values is not None else None

                if self.gradient_checkpointing and self.training:
                    print(f"Bad gradient_checkpointing: {self.gradient_checkpointing}")
                    layer_outputs = self._gradient_checkpointing_func(
                        layer_module.__call__,
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                        type_ids,
                        motif_struct_emb,
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                        anno_embed,
                        type_ids,
                        motif_struct_emb,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = next_decoder_cache + (layer_outputs[-1],)
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
                    if self.config.add_cross_attention:
                        all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class ModifiedEsmModel(EsmModel):
    def __init__(self, config, add_pooling_layer=True):
        EsmPreTrainedModel.__init__(self, config)
        self.config = config

        self.embeddings = EsmEmbeddings(config)
        self.encoder = CFPGenEncoder(config)

        self.pooler = EsmPooler(config) if add_pooling_layer else None

        self.contact_head = EsmContactPredictionHead(
            in_features=config.num_hidden_layers * config.num_attention_heads, bias=True
        )

        self.hidden_size = config.hidden_size

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, AGFMLayer):
            # set shift and scale to 0
            nn.init.constant_(module.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(module.adaLN_modulation[-1].bias[:2 * self.hidden_size], 0)
            nn.init.constant_(module.adaLN_modulation[-1].bias[3 * self.hidden_size:5 * self.hidden_size], 0)
            # set gate to 1
            nn.init.constant_(module.adaLN_modulation[-1].bias[2 * self.hidden_size:3 * self.hidden_size], 1)
            nn.init.constant_(module.adaLN_modulation[-1].bias[5 * self.hidden_size:6 * self.hidden_size], 1)


    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            seq_cond_attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            try:
                input_shape = input_ids['x_t'].size()
            except (KeyError, TypeError, AttributeError, IndexError):
                input_shape = input_ids.size() if torch.is_tensor(input_ids) else None
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        if not torch.is_tensor(input_ids):
            device = input_ids['x_t'].device if input_ids is not None else inputs_embeds.device
        else:
            device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            # encoder_extended_attention_mask = None
            encoder_extended_attention_mask = encoder_attention_mask

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids if torch.is_tensor(input_ids) else input_ids['x_t'],
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        if input_ids['seq_cond'] is not None:
            seq_cond_embedding = self.embeddings(
                input_ids=input_ids['seq_cond'],
                position_ids=position_ids,
                attention_mask=seq_cond_attention_mask,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )
            input_ids['seq_cond'] = seq_cond_embedding
            input_ids['seq_cond_att_mask'] = seq_cond_attention_mask

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            anno_tag=input_ids,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class ModifiedEsmModelDPLM2(EsmModel):
    def __init__(self, config, add_pooling_layer=True):
        EsmPreTrainedModel.__init__(self, config)
        self.config = config

        self.embeddings = EsmEmbeddings(config)
        self.encoder = CFPGenEncoderDPLM2(config)

        self.pooler = EsmPooler(config) if add_pooling_layer else None

        self.contact_head = EsmContactPredictionHead(
            in_features=config.num_hidden_layers * config.num_attention_heads, bias=True
        )

        self.hidden_size = config.hidden_size

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, AGFMLayer):
            # set shift and scale to 0
            nn.init.constant_(module.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(module.adaLN_modulation[-1].bias[:2 * self.hidden_size], 0)
            nn.init.constant_(module.adaLN_modulation[-1].bias[3 * self.hidden_size:5 * self.hidden_size], 0)
            # set gate to 1
            nn.init.constant_(module.adaLN_modulation[-1].bias[2 * self.hidden_size:3 * self.hidden_size], 1)
            nn.init.constant_(module.adaLN_modulation[-1].bias[5 * self.hidden_size:6 * self.hidden_size], 1)


    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            seq_cond_attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            type_ids: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:

    

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            try:
                input_shape = input_ids['x_t'].size()
                # input_shape = input_ids.size()
            except (KeyError, TypeError, AttributeError, IndexError):
                input_shape = input_ids.size() if torch.is_tensor(input_ids) else None
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Note 这里为什么可以用这种方式获取shape 理论上datamodule返回的是个字典
        batch_size, seq_length = input_shape
        if not torch.is_tensor(input_ids):
            device = input_ids['x_t'].device if input_ids is not None else inputs_embeds.device
        else:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
        # device = (
        #     input_ids.device if input_ids is not None else inputs_embeds.device
        # )

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        
        # TODO: maybe bug
        if attention_mask.dim() == 4:
            extended_attention_mask = attention_mask
        elif attention_mask.dim() == 2:
            extended_attention_mask: torch.Tensor = (
                self.get_extended_attention_mask(attention_mask, input_shape)
            )
        else:
            raise ValueError(
                f"Wrong shape for attention_mask (shape {attention_mask.shape})! "
                "Should be [batch_size, seq_length] or [batch_size, seq_length, seq_length]."
            )
        # extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            # encoder_extended_attention_mask = None
            encoder_extended_attention_mask = encoder_attention_mask

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # embedding_output = self.embeddings(
        #     input_ids=input_ids if torch.is_tensor(input_ids) else input_ids['x_t'],
        #     position_ids=position_ids,
        #     attention_mask=attention_mask,
        #     inputs_embeds=inputs_embeds,
        #     past_key_values_length=past_key_values_length,
        # )
        embedding_output = self.embeddings(
            input_ids=input_ids if torch.is_tensor(input_ids) else input_ids['x_t'],
            position_ids=position_ids,
            attention_mask=input_ids['x_t'].ne(
                self.config.pad_token_id
            ),  # attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        # TODO: seq cond input and attention mask
        # if input_ids['seq_cond'] is not None:
        #     seq_cond_embedding = self.embeddings(
        #         input_ids=input_ids['seq_cond'],
        #         position_ids=position_ids,
        #         attention_mask=seq_cond_attention_mask,
        #         inputs_embeds=inputs_embeds,
        #         past_key_values_length=past_key_values_length,
        #     )
        #     input_ids['seq_cond'] = seq_cond_embedding
        #     input_ids['seq_cond_att_mask'] = seq_cond_attention_mask

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            anno_tag=input_ids,
            type_ids=type_ids,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

@register_model('func_mlm_esm')
class EsmForCFPGEN(EsmForMaskedLM):
    def __init__(self, config, dropout=0.1):
        tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        config.hidden_dropout_prob = dropout
        
        EsmPreTrainedModel.__init__(self, config)
        self.esm = ModifiedEsmModel(config, add_pooling_layer=False)
        self.lm_head = EsmLMHead(config)

        self.init_weights()
        
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.eos_token_id
        self.x_id = tokenizer._token_to_id['X']
        
        self.contact_head = None
        self.tokenizer = tokenizer
    
    def forward(self,
                input_ids,
                attention_mask=None,
                inputs_embeds=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                decoder_inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
            ):


        assert isinstance(input_ids, dict)
        attention_mask = input_ids['x_t'].ne(self.pad_id)

        seq_cond_attention_mask = input_ids['seq_cond'].ne(self.pad_id) if input_ids['seq_cond'] is not None else None

        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            seq_cond_attention_mask=seq_cond_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        sequence_output = outputs[0]
        logits = self.lm_head(sequence_output)
        
        result = {
            "logits": logits,
            "last_hidden_state": sequence_output,
        }
        return result


def sample_from_categorical(logits=None, temperature=1.0):
    if temperature:
        dist = torch.distributions.Categorical(logits=logits.div(temperature))
        tokens = dist.sample()
        scores = dist.log_prob(tokens)
    else:
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores




@register_model('func_mlm_esm_dplm2')
class EsmForCFPGEN_DPLM2(EsmForMaskedLM):
    def __init__(self, config, dropout=0.1):
        print(f"Loading model from {config._name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        config.hidden_dropout_prob = dropout
        
        EsmPreTrainedModel.__init__(self, config)
        self.esm = ModifiedEsmModelDPLM2(config, add_pooling_layer=False)
        self.lm_head = EsmLMHead(config)

        self.init_weights()
        
        # self.mask_id = tokenizer.mask_token_id
        # self.pad_id = tokenizer.pad_token_id
        self.pad_id = tokenizer.pad_token_id
        self.config.pad_token_id = self.pad_id
        # self.bos_id = tokenizer.cls_token_id
        # self.eos_id = tokenizer.eos_token_id
        # self.x_id = tokenizer._token_to_id['X']
        
        self.contact_head = None
        self.tokenizer = tokenizer
    
    def forward(self,
                input_ids,
                attention_mask=None,
                type_ids=None,
                inputs_embeds=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                decoder_inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
            ):


        assert isinstance(input_ids, dict)
        # attention_mask = input_ids['x_t'].ne(self.pad_id)

        # TODO special for seq_conda attention mask
        # seq_cond_attention_mask = input_ids['seq_cond'].ne(self.pad_id) if input_ids['seq_cond'] is not None else None
        seq_cond_attention_mask = None

        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            seq_cond_attention_mask=seq_cond_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            type_ids=type_ids,
        )
        sequence_output = outputs[0]
        logits = self.lm_head(sequence_output)
        
        result = {
            "logits": logits,
            "last_hidden_state": sequence_output,
        }
        return result
