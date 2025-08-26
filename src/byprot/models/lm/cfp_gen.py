
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import math
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from byprot.models import register_model
from omegaconf import OmegaConf
from byprot.models.lm.model_utils import LoRAConfig, NetConfig, CondConfig, get_net, get_net_class, \
    sample_from_categorical, stochastic_sample_from_categorical, top_k_top_p_filtering, topk_masking, topk_masking_prior
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm
import os
import random

from byprot.datamodules.dataset.tokenized_protein import DPLM2Tokenizer

from dataclasses import dataclass, field


@dataclass
class SelfMixupConfig:
    enable: bool = field(default=False)
    with_original_loss: bool = field(default=False)


@dataclass
class TokenizerConfig:
    vocab_file: str = field(default="airkingbd/dplm2_650m")
    # amino acid tokens (33) + struct tokens (8192) + 4 special struct tokens
    vocab_size: int = field(default=33 + 8192 + 4)


@dataclass
class StructTokenizerConfig:
    enable: bool = field(default=True)
    exp_path: str = field(default="airkingbd/struct_tokenizer")

@dataclass
class CFPGENConfig:
    num_diffusion_timesteps: int = field(
        default=500
    )
    lora: LoRAConfig = field(default=LoRAConfig())
    net: NetConfig = field(default=NetConfig())
    gradient_ckpt: bool = field(
        default=False
    )
    rdm_couple: bool = field(
        default=False
    )
    cond: CondConfig = field(default_factory=CondConfig)

@dataclass
class CFPGENConfig_DPLM2:
    ## DPLM model
    num_diffusion_timesteps: int = field(default=500)
    tokenizer: TokenizerConfig = field(default=TokenizerConfig())
    lora: LoRAConfig = field(default=LoRAConfig())
    net: NetConfig = field(default=NetConfig())
    gradient_ckpt: bool = field(default=False)

    ## multi-modal training
    training_stage: str = field(default="train_from_dplm")
    self_mixup: SelfMixupConfig = field(
        default=SelfMixupConfig()
    )  # training strategy
    single_modality_ratio: float = field(default=0.25)
    folding_loss_ratio: float = field(default=0.25)
    inverse_folding_loss_ratio: float = field(default=0.25)
    joint_loss_ratio: float = field(default=0.25)
    independent_loss_ratio: float = field(default=0.0)

    ## struct tokenizer
    struct_tokenizer: StructTokenizerConfig = field(
        default=StructTokenizerConfig()
    )

    rdm_couple: bool = field(
        default=False
    )
    cond: CondConfig = field(default_factory=CondConfig)

    use_diff_modulation: bool = field(default=False)
    use_func_cross_attn: bool = field(default=False)
    use_diff_ce: bool = field(default=False)
    use_motif_struct_emb: bool = field(default=False)


@register_model('cfp_gen')
class CondDiffusionProteinLanguageModel(nn.Module):
    _default_cfg = CFPGENConfig()
    
    def __init__(self, cfg, net=None):
        super().__init__()
        self._update_cfg(cfg)
        
        # Note：嵌入很深，触及到esm.encode层的注意力机制，小心改
        self.net = get_net(cfg) if net is None else net
        self.tokenizer = self.net.tokenizer

        self.mask_id = self.net.mask_id
        self.pad_id = self.net.pad_id
        self.bos_id = self.net.bos_id
        self.eos_id = self.net.eos_id
        self.x_id = self.net.x_id
        
        if self.cfg.gradient_ckpt:
            self.net.supports_gradient_checkpointing = True
            self.net.gradient_checkpointing_enable()
    
    @classmethod
    def from_pretrained(cls, net_name, cfg_override={}, net_override={}, from_huggingface=False):
        if not from_huggingface:
            # Load model checkpoint from local if you pretrain a DPLM with this repo
            # The net_name should be like:
            # ${name}/checkpoints/last.ckpt
            # and there should be .hydra/config.yaml in the ${name} directory that is automatically generated during training.
            from byprot.utils.config import load_yaml_config
            from pathlib import Path
            from collections import OrderedDict
            
            cfg_path = Path(net_name).parents[1]
            cfg_path = Path(cfg_path, '.hydra', 'config.yaml')
            cfg = load_yaml_config(str(cfg_path)).model
            cfg.net.pretrain = False
            cfg.pop('_target_')
            model = cls(cfg)
            
            pretrained_state_dict = torch.load(net_name, map_location=torch.device("cpu"))['state_dict']
            new_pretrained_state_dict = OrderedDict()
            
            # remove the module prefix "model."
            for k, v in pretrained_state_dict.items():
                new_pretrained_state_dict[k[6:]] = v
            model.load_state_dict(new_pretrained_state_dict, strict=True)
            return model
        else:
            # Load DPLM model checkpoint from huggingface
            net_type = AutoConfig.from_pretrained(net_name).model_type
            net_class = get_net_class(net_type)
            net = net_class.from_pretrained(net_name, **net_override)
            return cls(cfg=cfg_override, net=net)
    
    def _update_cfg(self, cfg):
        # if '_target_' in cfg.net:
        #     cfg.net.pop('_target_')
        self.cfg = OmegaConf.merge(self._default_cfg, cfg)
        
    def q_sample_coupled(self, x_0, t1, t2, maskable_mask):
        # partial mask: True for the part should not be mask
        t1_eq_t2_mask = (t1 == t2)
        t1, t2 = torch.maximum(t1, t2).float(), torch.minimum(t1, t2).float()

        # sample t1
        u = torch.rand_like(x_0, dtype=torch.float)
        t1_mask = (u < (t1 / self.cfg.num_diffusion_timesteps)[:, None]) & maskable_mask
        x_t1 = x_0.masked_fill(t1_mask, self.mask_id) 

        # sample t2
        u = torch.rand_like(x_0, dtype=torch.float)
        t2_mask = t1_mask & (u > ((t1 - t2) / t1)[:, None])
        u = torch.rand_like(x_0[t1_eq_t2_mask], dtype=torch.float)
        t2_mask[t1_eq_t2_mask] = (u < (t1[t1_eq_t2_mask] / self.cfg.num_diffusion_timesteps)[:, None]) & (maskable_mask[t1_eq_t2_mask])
        x_t2 = x_0.masked_fill(t2_mask, self.mask_id) 

        return {
            "x_t": torch.cat([x_t1, x_t2], dim=0),
            "t": torch.cat([t1, t2]),
            "mask_mask": torch.cat([t1_mask, t2_mask], dim=0)
        }

    def q_sample(self, x_0, t1, maskable_mask):
        # sample t1
        u = torch.rand_like(x_0, dtype=torch.float)
        t1_mask = (u < (t1 / self.cfg.num_diffusion_timesteps)[:, None]) & maskable_mask
        x_t1 = x_0.masked_fill(t1_mask, self.mask_id)
        x_t1 = x_t1.masked_fill(t1_mask, self.mask_id)

        return {
            "x_t": x_t1,
            "t": t1,
            "mask_mask": t1_mask,
        }
        
    def forward(self, input_ids, return_last_hidden_state=False, **kwargs):
        outputs = self.net(
            input_ids=input_ids,
        )
        logits = outputs['logits']
        if return_last_hidden_state:
            last_hidden_state = outputs['last_hidden_state']
            return logits, last_hidden_state
        else:
            return logits


    def get_motif_original(self, target, motif_start_end, motif_len_min, motif_len_max, min_mask_ratio=0.05, max_mask_ratio=0.1):
        batch_size, sequence_length = target.shape
        masked_targets = []

        for i in range(batch_size):
            current_target = target[i].clone()

            non_special_sym_mask = (
                    (current_target != self.pad_id) &
                    (current_target != self.bos_id) &
                    (current_target != self.eos_id)
            )
            effective_indices = torch.where(non_special_sym_mask)[0]

            if len(effective_indices) == 0:
                masked_targets.append(torch.full_like(current_target, fill_value=self.mask_id))
                continue

            total_length = len(effective_indices)
            retain_min_len = max(motif_len_min, int(min_mask_ratio * total_length))
            retain_max_len = max(motif_len_max, int(max_mask_ratio * total_length))

            start, end = motif_start_end[i]

            if start == 0 and end == 0:
                retain_length = torch.randint(retain_min_len, retain_max_len + 1, (1,)).item()
                retain_start_idx = torch.randint(0, total_length - retain_length + 1, (1,)).item()
                retain_start = effective_indices[retain_start_idx].item()
                retain_end = effective_indices[retain_start_idx + retain_length - 1].item()
            else:
                motif_length = end - start
                if motif_length < retain_min_len:
                    retain_length = retain_min_len
                elif motif_length > retain_max_len:
                    retain_length = retain_max_len
                else:
                    retain_length = motif_length

                if end - start - retain_length > 0:
                    retain_start = torch.randint(start, end - retain_length + 1, (1,)).item()
                else:
                    retain_start = start

                retain_end = retain_start + retain_length - 1

            sequence_indices = torch.arange(sequence_length, device=target.device)
            mask = non_special_sym_mask & ((sequence_indices < retain_start) | (sequence_indices > retain_end))
            masked_target = current_target.clone()
            masked_target[mask] = self.mask_id

            masked_targets.append(masked_target)

        return torch.stack(masked_targets)

    def get_motif_middle(self, target, motif_start_end, motif_len_min=10, motif_len_max=30):

        batch_size, sequence_length = target.shape
        masked_targets = []

        for i in range(batch_size):
            current_target = target[i].clone()
            if sum(motif_start_end[i]) == 0:
                non_special_sym_mask = (
                        (current_target != self.pad_id) &
                        (current_target != self.bos_id) &
                        (current_target != self.eos_id)
                )
                effective_indices = torch.where(non_special_sym_mask)[0]
                if len(effective_indices) == 0:
                    masked_targets.append(torch.full_like(current_target, fill_value=self.mask_id))
                    continue

                start = effective_indices[0].item()
                end = effective_indices[-1].item()
            else:
                start, end = motif_start_end[i]

            motif_length = end - start

            if motif_length < motif_len_min:
                crop_len = motif_length
            else:
                crop_len = min(torch.randint(motif_len_min, min(motif_len_max, motif_length) + 1, (1,)).item(), motif_length)

            non_special_sym_mask = (
                    (current_target != self.pad_id) &
                    (current_target != self.bos_id) &
                    (current_target != self.eos_id)
            )

            effective_indices = torch.where(non_special_sym_mask)[0]
            if len(effective_indices) == 0:
                masked_targets.append(torch.full_like(current_target, fill_value=self.mask_id))
                continue

            middle_position = (effective_indices[0] + effective_indices[-1]) // 2
            crop_start = max(middle_position - crop_len // 2, effective_indices[0])
            crop_end = min(crop_start + crop_len, effective_indices[-1] + 1)
            crop_start = crop_end - crop_len

            masked_target = current_target.clone()
            masked_target[non_special_sym_mask] = self.mask_id
            masked_target[crop_start:crop_end] = current_target[crop_start:crop_end]

            masked_targets.append(masked_target)

        masked_target = torch.stack(masked_targets)

        return masked_target


    def compute_loss(self, batch, weighting='constant'):
        target = batch['targets']

        # couple
        t1, t2 = torch.randint(
            1, self.cfg.num_diffusion_timesteps + 1,
            (2 * target.size(0), ),
            device=target.device
        ).chunk(2)

        if self.cfg.rdm_couple:
            x_t, t, loss_mask = list(
                self.q_sample_coupled(
                    target, t1, t2,
                    maskable_mask=self.get_non_special_sym_mask(target)
                ).values()
            )
            target = target.repeat(2, 1)
        else:
            x_t, t, loss_mask = list(
                self.q_sample(
                    target, t1,
                    maskable_mask=self.get_non_special_sym_mask(target)
                ).values()
            )

        masked_target = None
        if self.cfg.cond.use_seq_motif:
            if random.random() < 0.5:
                masked_target = self.get_motif_original(target, batch['motif_start_end'], motif_len_min=self.cfg.cond.motif_min_len, motif_len_max=self.cfg.cond.motif_max_len)
            else:
                masked_target = self.get_motif_middle(target, batch['motif_start_end'], motif_len_min=self.cfg.cond.motif_min_len, motif_len_max=self.cfg.cond.motif_max_len)

        inputs = dict(x_t=x_t, seq_cond=masked_target, go=batch['go_type'], ipr=batch['ipr_type'], ec=batch['ec_type'])

        logits = self.forward(inputs)

        num_timesteps = self.cfg.num_diffusion_timesteps
        weight = {
                     "linear": (num_timesteps - (t - 1)),    # num_timesteps * (1 - (t-1)/num_timesteps)
                     "constant": num_timesteps * torch.ones_like(t)
                 }[weighting][:, None].float() / num_timesteps

        return logits, target, loss_mask, weight

    def forward_encoder(self, batch, **kwargs):
        return {}

    def initialize_output_tokens(self, batch, partial_masks=None, **kwargs):
        tokens = batch['input_ids']
        if tokens is None:
            raise NotImplementedError
        else:
            output_mask = self.get_non_special_sym_mask(tokens, partial_masks=partial_masks)

            output_tokens = tokens.masked_fill(output_mask, self.mask_id)
            output_scores = torch.zeros_like(output_tokens, dtype=torch.float)

            return output_tokens, output_scores

    def resample_conditional(self, _tokens, _scores, ratio, scale, go=None, ipr=None, seq_cond=None, ec=None):
        to_be_resample_idx = []
        resample_input = []
        resample_input_mask = []
        resample_input_scores = []
        resample_input_seq_cond = []
        for i, seq in enumerate(_tokens):
            most_token_dict = {}
            most_token = None
            most_token_num = -1
            for j, token in enumerate(seq):
                token = int(token)
                if token not in most_token_dict:
                    most_token_dict[token] = [j]
                else:
                    most_token_dict[token].append(j)
                if len(most_token_dict[token]) > most_token_num:
                    most_token = token
                    most_token_num = len(most_token_dict[token])
            if most_token_num > len(seq) * ratio:#max(0.3/(step+1) ** 0.2, 0.1):
                to_be_resample_idx.append(i)
                resample_input_scores.append(_scores[i])
                mask = torch.zeros_like(seq).bool()
                for k, v in most_token_dict.items():
                    if len(v) > len(seq) * ratio:#max(0.3/(step+1) ** 0.2, 0.1):
                        mask |= seq.eq(k)
                resample_input_mask.append(mask)
                resample_input.append(seq.masked_fill(mask, self.mask_id))
                if seq_cond is not None:
                    resample_input_seq_cond.append(seq_cond[i].masked_fill(mask, self.mask_id))
                #resample_input.append(seq.masked_scatter(mask, xt[i][mask]))
            
        if len(to_be_resample_idx) > 0:
            resample_input = torch.stack(resample_input, dim=0).type_as(_tokens)
            resample_input_scores = torch.stack(resample_input_scores, dim=0).type_as(_scores)
            resample_input_mask = torch.stack(resample_input_mask, dim=0).type_as(_tokens).bool()
            if seq_cond is not None:
                resample_input_seq_cond = torch.stack(resample_input_seq_cond, dim=0).type_as(_tokens)

            inputs = dict(x_t=resample_input, go=go, ipr=ipr, seq_cond=resample_input_seq_cond if seq_cond is not None else None, ec=ec)
            resample_logits = self.net(
                input_ids=inputs,
            )['logits']
            if resample_logits.dtype != _scores.dtype:
                resample_logits = resample_logits.type_as(_scores)
            resample_logits[..., self.mask_id] = -math.inf
            resample_logits[..., self.x_id] = -math.inf
            resample_logits[..., self.pad_id] = -math.inf
            resample_logits[..., self.bos_id] = -math.inf
            resample_logits[..., self.eos_id] = -math.inf
            
            resample_logits = top_k_top_p_filtering(resample_logits, top_p=0.95)
            #noise_scale = 1.5 - 0.2 * ((step + 1) / max_step)
            noise_scale = scale
            assert resample_logits.size(0) == len(to_be_resample_idx)
            resample_tokens, resample_scores = stochastic_sample_from_categorical(resample_logits, temperature=0.0, noise_scale=noise_scale)
            resample_input.masked_scatter_(resample_input_mask, resample_tokens[resample_input_mask])
            resample_input_scores.masked_scatter_(resample_input_mask, resample_scores[resample_input_mask])
            _tokens[to_be_resample_idx], _scores[to_be_resample_idx] = resample_input, resample_input_scores
            
    def forward_decoder(self, prev_decoder_out, encoder_out=None, need_attn_weights=False, partial_masks=None,
                        sampling_strategy='gumbel_argmax', go_label=None, ipr_label=None, seq_cond=None, ec_label=None,):
        output_tokens = prev_decoder_out['output_tokens'].clone()
        output_scores = prev_decoder_out['output_scores'].clone()
        step, max_step = prev_decoder_out['step'], prev_decoder_out['max_step']
        temperature = prev_decoder_out['temperature']
        history = prev_decoder_out['history']

        output_masks = self.get_non_special_sym_mask(output_tokens, partial_masks=partial_masks)

        inputs = dict(x_t=output_tokens, go=go_label, ipr=ipr_label, seq_cond=seq_cond, ec=ec_label)
        net_out = self.net(inputs)
        
        logits = net_out['logits']
        attentions = net_out['attentions'] if need_attn_weights else None
        
        if logits.dtype != output_scores.dtype:
            logits = logits.type_as(output_scores)

        logits[..., self.mask_id] = -math.inf
        logits[..., self.x_id] = -math.inf
        logits[..., self.pad_id] = -math.inf
        logits[..., self.bos_id] = -math.inf
        logits[..., self.eos_id] = -math.inf
        
        #logits = top_k_top_p_filtering(logits, top_p=0.95)

        if sampling_strategy == 'vanilla':
            _tokens, _scores = sample_from_categorical(logits, temperature=temperature)
        elif sampling_strategy == 'argmax':
            _scores, _tokens = logits.max(-1)
        elif sampling_strategy == 'gumbel_argmax':
            noise_scale = 1.0
            _tokens, _scores = stochastic_sample_from_categorical(logits, temperature=0.0, noise_scale=noise_scale)

            self.resample_conditional(_tokens, _scores, ratio=0.25, scale=1.0, go=go_label, ipr=ipr_label, seq_cond=seq_cond, ec=ec_label)
        else:
            raise NotImplementedError
        
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        history.append(output_tokens.clone())

        return dict(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attentions=attentions, # [B, L, H, T, T]
            step=step + 1,
            max_step=max_step,
            history=history,
            hidden_states=net_out['last_hidden_state']
        )

    def get_non_special_sym_mask(self, output_tokens, partial_masks=None):
        non_special_sym_mask = (
            output_tokens.ne(self.pad_id) &
            output_tokens.ne(self.bos_id) &
            output_tokens.ne(self.eos_id)
        )
        if partial_masks is not None:
            non_special_sym_mask &= (~partial_masks)
        return non_special_sym_mask

    def _reparam_decoding(
        self,
        output_tokens,
        output_scores,
        cur_tokens,
        cur_scores,
        decoding_strategy,
        xt_neq_x0,
        non_special_sym_mask,
        t,
        max_step,
        noise,
    ):
        """
            This function is used to perform reparameterized decoding.
        """
        # output_tokens: [B, N]
        # output_scores: [B, N]
        # cur_tokens: [B, N]
        # cur_scores: [B, N]
        # xt_neq_x0: equivalent to not_b_t [B, N]
        # non_special_sym_mask: [B, N]
        # noise: either [B, N] or scalar (if using the mask noise)

        # decoding_strategy needs to take the form of "reparam-<conditioning>-<topk_mode>-<schedule>"
        _, condition, topk_mode, schedule = decoding_strategy.split("-")

        # first set the denoising rate according to the schedule
        if schedule == "linear":
            rate = 1 - t / max_step
        elif schedule == "cosine":
            rate = np.cos(t / max_step * np.pi * 0.5)
        else:
            raise NotImplementedError

        # compute the cutoff length for denoising top-k positions
        cutoff_len = (
            non_special_sym_mask.sum(1, keepdim=True).type_as(output_scores) * rate
        ).long()
        # set the scores of special symbols to a large value so that they will never be selected
        _scores_for_topk = cur_scores.masked_fill(~non_special_sym_mask, 1000.0)
        
        to_be_resample = []
        for i, seq in enumerate(cur_tokens):
            most_token_dict = {}
            most_token = None
            most_token_num = -1
            for j, token in enumerate(seq):
                token = int(token)
                if token == self.pad_id:
                    continue
                if token not in most_token_dict:
                    most_token_dict[token] = [j]
                else:
                    most_token_dict[token].append(j)
                if len(most_token_dict[token]) > most_token_num:
                    most_token = token
                    most_token_num = len(most_token_dict[token])
            if most_token_num > len(seq) * 0.25:
                to_be_resample.append(i)
                
        # the top-k selection can be done in two ways: stochastic by injecting Gumbel noise or deterministic
        if topk_mode.startswith("stochastic"):
            noise_scale = float(topk_mode.replace("stochastic", ""))
            lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=True, temp=noise_scale * rate)
        elif topk_mode == "deterministic":
            lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=False)
            if len(to_be_resample) > 0:
                noise_scale = 1.5
                #print(lowest_k_mask[to_be_resample[0]])
                lowest_k_mask[to_be_resample] = topk_masking(_scores_for_topk[to_be_resample], cutoff_len[to_be_resample], 
                                                             stochastic=True, temp=noise_scale * rate)
        else:
            raise NotImplementedError

        # Various choices to generate v_t := [v1_t, v2_t].
        # Note that
        #   v1_t governs the outcomes of tokens where b_t = 1,
        #   v2_t governs the outcomes of tokens where b_t = 0.

        # #### the `uncond` mode ####
        # In our reparameterized decoding,
        # both v1_t and v2_t can be fully determined by the current token scores .

        # #### the `cond` mode ####
        # However, we can also impose some conditional constraints on v1_t so that
        # the decoding can be performed in a more conservative manner.
        # For example, we can set v1_t = 0 only when
        # (the newly output tokens are the same as previous denoised results, AND
        # the current token score becomes lower, AND
        # the current token score is not in the top-k share among all tokens).
        if condition == "cond":
            not_v1_t = (cur_tokens == output_tokens) & (cur_scores < output_scores) & lowest_k_mask
        elif condition == "uncond":
            not_v1_t = lowest_k_mask
        else:
            raise NotImplementedError

        # for b_t = 0, the token is set to noise if it is in the lowest k scores.
        not_v2_t = lowest_k_mask

        last_mask_position = xt_neq_x0
        masked_to_noise = (~xt_neq_x0 & not_v1_t) | (xt_neq_x0 & not_v2_t)
        if isinstance(noise, torch.Tensor):
            output_tokens.masked_scatter_(masked_to_noise, noise[masked_to_noise])
        elif isinstance(noise, (int, float)):
            output_tokens.masked_fill_(masked_to_noise, noise)
        else:
            raise NotImplementedError("noise should be either a tensor or a scalar")
        output_scores.masked_fill_(masked_to_noise, -math.inf)

        masked_to_x0 = xt_neq_x0 & ~not_v2_t
        output_tokens.masked_scatter_(masked_to_x0, cur_tokens[masked_to_x0])
        output_scores.masked_scatter_(masked_to_x0, cur_scores[masked_to_x0])
        assert ((masked_to_x0 & last_mask_position) == masked_to_x0).all()
        # b_{t} = (b_{t+1} & u_t) | v_t
        # For convenience, save the NOT of b_t for the next iteration
        # NOT_b_{t} = (NOT_b_{t+1} | not_v1_t) & not_v2_t
        #
        # # When condition is 'uncond', the not_v1_t is equal to not_v2_t, the new_xt_neq_x0 is always equal to not_v1/v2_t
        new_xt_neq_x0 = (xt_neq_x0 | not_v1_t) & not_v2_t
        assert (new_xt_neq_x0 == not_v2_t).all()
        return new_xt_neq_x0, output_tokens, output_scores

    def generate(self, batch, tokenizer=None, 
                 max_iter=None, temperature=None, 
                 partial_masks=None,
                 sampling_strategy='gumbel_argmax'):
        # tokenizer = tokenizer
        # max_iter = max_iter
        # temperature = temperature

        # 0) encoding
        encoder_out = self.forward_encoder(batch)
        # 1) initialized from all mask tokens, where partial_masks will fix motif
        initial_output_tokens, initial_output_scores = self.initialize_output_tokens(
            batch, encoder_out=encoder_out, partial_masks=partial_masks)  #
        prev_decoder_out = dict(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            output_masks=None,
            attentions=None,
            step=0,
            max_step=max_iter,
            history=[initial_output_tokens.clone()],
            temperature=temperature,
        )

        prev_decoder_out['output_masks'] = self.get_non_special_sym_mask(
                prev_decoder_out['output_tokens'], partial_masks=partial_masks
            )

        for step in tqdm(range(max_iter), desc='Decoding'):
            # 2.1: predict
            with torch.no_grad():
                decoder_out = self.forward_decoder(
                    prev_decoder_out=prev_decoder_out,
                    encoder_out=encoder_out,
                    need_attn_weights=False,
                    partial_masks=partial_masks,
                    sampling_strategy=sampling_strategy,
                    go_label=batch.get('go_label', None),
                    ipr_label=batch.get('ipr_label', None),
                    seq_cond=batch.get('seq_cond', None),
                    ec_label=batch.get('ec_label', None),

                )

            output_tokens = decoder_out['output_tokens']
            output_scores = decoder_out['output_scores']

            # 2.2: re-mask skeptical parts of low confidence
            non_special_sym_mask = self.get_non_special_sym_mask(
                prev_decoder_out['output_tokens'], partial_masks=partial_masks
            )
            
            output_masks, result_tokens, result_scores = self._reparam_decoding(
                output_tokens=prev_decoder_out['output_tokens'].clone(),
                output_scores=prev_decoder_out['output_scores'].clone(),
                cur_tokens=output_tokens.clone(),
                cur_scores=output_scores.clone(),
                decoding_strategy='reparam-uncond-deterministic-linear',#'reparam-uncond-stochastic1.0-linear'
                xt_neq_x0=prev_decoder_out['output_masks'],
                non_special_sym_mask=non_special_sym_mask,
                t=step + 1,
                max_step=max_iter,
                noise=self.mask_id,
            )
            # output_masks, result_tokens, result_scores = self._reparam_decoding(
            #     output_tokens=output_tokens.clone(),#output_tokens,#
            #     output_scores=output_scores.clone(),#output_scores,##
            #     cur_tokens=prev_decoder_out['output_tokens'].clone(),#prev_decoder_out['output_tokens'],##
            #     cur_scores=prev_decoder_out['output_scores'].clone(),#prev_decoder_out['output_scores'],##
            #     decoding_strategy='reparam-uncond-deterministic-linear',#'reparam-uncond-stochastic1.0-linear',#,##
            #     # decoding_strategy='reparam-uncond-deterministic-cosine',
            #     xt_neq_x0=prev_decoder_out['output_masks'],
            #     non_special_sym_mask=non_special_sym_mask,
            #     t=step + 1,
            #     max_step=max_iter,
            #     noise=self.mask_id, # if 'init_pred' not in encoder_out else encoder_out['init_pred'],
            #     mask_811=False
            # )
            prev_decoder_out.update(output_masks=output_masks)
            output_tokens = result_tokens
            output_scores = result_scores

            prev_decoder_out.update(
                output_tokens=output_tokens,
                output_scores=output_scores,
                step=step + 1,
                history=decoder_out['history']
            )

        decoder_out = prev_decoder_out
        return decoder_out['output_tokens'], decoder_out['output_scores']


@register_model('cfp_gen_dplm2')
class CondDiffusionProteinLanguageModel2(nn.Module):
    _default_cfg = CFPGENConfig_DPLM2()
    
    def __init__(self, cfg, net=None):
        # print("init cdplm2")
        super().__init__()
        # print("init cdplm2")
        self._update_cfg(cfg)
        # print("init cdplm2")
        self.tokenizer = DPLM2Tokenizer.from_pretrained(
            self.cfg.tokenizer.vocab_file
        )
        self._prepare_special_token()
        self.cfg.tokenizer.vocab_size = len(self.tokenizer)
        # print("init cdplm2")

        # Note：嵌入很深，触及到esm.encode层的注意力机制，小心改
        self.net = get_net(cfg) if net is None else net
        # self.tokenizer = self.net.tokenizer
        # print("init cdplm2")

        self.use_diff_ce = getattr(self.cfg, 'use_diff_ce', False)
        self.use_motif_struct_emb = getattr(self.cfg, 'use_motif_struct_emb', False)

        
        if self.cfg.gradient_ckpt:
            self.net.supports_gradient_checkpointing = True
            self.net.gradient_checkpointing_enable()
    
        self._struct_tokenizer = None
        # print("init cdplm2 done")

    def _prepare_special_token(self):
        self.aa_bos_id = self.tokenizer._token_to_id["<cls_aa>"]
        self.aa_eos_id = self.tokenizer._token_to_id["<eos_aa>"]
        self.aa_mask_id = self.tokenizer._token_to_id["<mask_aa>"]
        self.struct_bos_id = self.tokenizer._token_to_id["<cls_struct>"]
        self.struct_eos_id = self.tokenizer._token_to_id["<eos_struct>"]
        self.struct_mask_id = self.tokenizer._token_to_id["<mask_struct>"]
        self.pad_id = self.tokenizer._token_to_id["<pad>"]
        self.aa_unk_id = self.tokenizer._token_to_id["<unk_aa>"]
        self.struct_unk_id = self.tokenizer._token_to_id["<unk_struct>"]

        self.aa_X_id = self.tokenizer._token_to_id["X"]
        self.aa_B_id = self.tokenizer._token_to_id["B"]
        self.aa_U_id = self.tokenizer._token_to_id["U"]
        self.aa_Z_id = self.tokenizer._token_to_id["Z"]
        self.aa_O_id = self.tokenizer._token_to_id["O"]

        self.aa_type = 1
        self.struct_type = 0
        self.pad_type = 2

    @property
    def special_token_list(self):
        return [
            self.aa_bos_id,
            self.aa_eos_id,
            self.aa_mask_id,
            self.struct_bos_id,
            self.struct_eos_id,
            self.struct_mask_id,
            self.pad_id,
            self.aa_unk_id,
            self.struct_unk_id,
            self.aa_X_id,
            self.aa_B_id,
            self.aa_U_id,
            self.aa_Z_id,
            self.aa_O_id,
        ]

    @classmethod
    def from_pretrained(cls, net_name, cfg_override={}, net_override={}, from_huggingface=False):
        if not from_huggingface:
            # Load model checkpoint from local if you pretrain a DPLM with this repo
            # The net_name should be like:
            # ${name}/checkpoints/last.ckpt
            # and there should be .hydra/config.yaml in the ${name} directory that is automatically generated during training.
            from byprot.utils.config import load_yaml_config
            from pathlib import Path
            from collections import OrderedDict
            
            cfg_path = Path(net_name).parents[1]
            cfg_path = Path(cfg_path, '.hydra', 'config.yaml')
            cfg = load_yaml_config(str(cfg_path)).model
            cfg.net.pretrain = False
            cfg.pop('_target_')
            model = cls(cfg)
            
            pretrained_state_dict = torch.load(net_name, map_location=torch.device("cpu"))['state_dict']
            new_pretrained_state_dict = OrderedDict()
            
            # remove the module prefix "model."
            for k, v in pretrained_state_dict.items():
                new_pretrained_state_dict[k[6:]] = v
            model.load_state_dict(new_pretrained_state_dict, strict=True)
            return model
        else:
            # Load DPLM model checkpoint from huggingface
            net_type = AutoConfig.from_pretrained(net_name).model_type
            net_class = get_net_class(net_type)
            net = net_class.from_pretrained(net_name, **net_override)
            return cls(cfg=cfg_override, net=net)

    @property
    def device(self):
        try:
            device = next(self.parameters()).device
        except:
            device = torch.device("cpu")
        return device

    # @property
    # def struct_tokenizer(self):
    #     if not exists(self._struct_tokenizer):
    #         print(f"Loading struct_tokenizer...")
    #         self._struct_tokenizer = get_struct_tokenizer(
    #             self.cfg.struct_tokenizer.exp_path
    #         ).to(self.device)
    #     return self._struct_tokenizer


    def _update_cfg(self, cfg):
        # if '_target_' in cfg.net:
        #     cfg.net.pop('_target_')
        self.cfg = OmegaConf.merge(self._default_cfg, cfg)
        
    def q_sample_coupled(self, x_0, t1, t2, maskable_mask):
        raise NotImplementedError
        pass
        # partial mask: True for the part should not be mask
        t1_eq_t2_mask = (t1 == t2)
        t1, t2 = torch.maximum(t1, t2).float(), torch.minimum(t1, t2).float()

        # sample t1
        u = torch.rand_like(x_0, dtype=torch.float)
        t1_mask = (u < (t1 / self.cfg.num_diffusion_timesteps)[:, None]) & maskable_mask
        x_t1 = x_0.masked_fill(t1_mask, self.mask_id) 

        # sample t2
        u = torch.rand_like(x_0, dtype=torch.float)
        t2_mask = t1_mask & (u > ((t1 - t2) / t1)[:, None])
        u = torch.rand_like(x_0[t1_eq_t2_mask], dtype=torch.float)
        t2_mask[t1_eq_t2_mask] = (u < (t1[t1_eq_t2_mask] / self.cfg.num_diffusion_timesteps)[:, None]) & (maskable_mask[t1_eq_t2_mask])
        x_t2 = x_0.masked_fill(t2_mask, self.mask_id) 

        return {
            "x_t": torch.cat([x_t1, x_t2], dim=0),
            "t": torch.cat([t1, t2]),
            "mask_mask": torch.cat([t1_mask, t2_mask], dim=0)
        }


    def q_sample(self, x_0, t, type_ids, maskable_mask):
        aa_position = type_ids == self.aa_type
        struct_position = type_ids == self.struct_type

        # sample x_t
        u = torch.rand_like(x_0, dtype=torch.float)
        t_mask = (
            u < (t / self.cfg.num_diffusion_timesteps)[:, None]
        ) & maskable_mask
        x_t = x_0.masked_fill(t_mask & aa_position, self.aa_mask_id)
        x_t = x_t.masked_fill(t_mask & struct_position, self.struct_mask_id)

        return x_t, t_mask
        
    def get_modality_type(self, input_ids):
        input_mask = input_ids.ne(self.pad_id)
        # HACK: all amino acid token id < 33, while all struct token id >= 33
        # 0 stands for struct, 1 stands for aa
        modality_type = ((input_ids < 33) & input_mask).int()
        # 2 stands for padding
        modality_type[~input_mask] = self.pad_type
        return modality_type
    
    def forward(self, input_ids, return_last_hidden_state=False, **kwargs):
        # outputs = self.net(
        #     input_ids=input_ids,
        # )
        # logits = outputs['logits']
        # if return_last_hidden_state:
        #     last_hidden_state = outputs['last_hidden_state']
        #     return logits, last_hidden_state
        # else:
        #     return logits
        # print(f"input_ids: {input_ids}")

        input_mask = input_ids['x_t'].ne(self.pad_id)

        type_ids = self.get_modality_type(input_ids['x_t'])

        L = input_ids['x_t'].shape[1]
        num_heads = self.net.config.num_attention_heads
        # [B, num_heads, L+2, L+2]
        attention_bias: torch.FloatType = (
            self.net.esm.get_extended_attention_mask(
                input_mask, input_ids['x_t'].shape
            ).repeat(1, num_heads, L, 1)
        )  # -inf for padding positions, 0 otherwise

        if "single_modality" in kwargs:
            single_modality_index = kwargs["single_modality"]
            struct_attention_bias, aa_attention_bias = attention_bias.chunk(
                2, dim=-2
            )
            struct_attention_bias[
                single_modality_index, :, :, L // 2 :
            ] = -math.inf
            aa_attention_bias[
                single_modality_index, :, :, : L // 2
            ] = -math.inf
            attention_bias = torch.concat(
                [struct_attention_bias, aa_attention_bias], dim=-2
            )

        # [B, L, d_model]
        # input_embeds = self.net.esm.embeddings(
        #     input_ids, attention_mask=input_mask
        # )
        input_embeds = None

        outputs = self.net(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_bias,
            type_ids=type_ids,
        )

        return outputs       


    def get_motif_original(self, target, motif_start_end, motif_len_min, motif_len_max, min_mask_ratio=0.05, max_mask_ratio=0.1):
        batch_size, sequence_length = target.shape
        masked_targets = []

        for i in range(batch_size):
            current_target = target[i].clone()

            non_special_sym_mask = (
                    (current_target != self.pad_id) &
                    (current_target != self.bos_id) &
                    (current_target != self.eos_id)
            )
            effective_indices = torch.where(non_special_sym_mask)[0]

            if len(effective_indices) == 0:
                masked_targets.append(torch.full_like(current_target, fill_value=self.mask_id))
                continue

            total_length = len(effective_indices)
            retain_min_len = max(motif_len_min, int(min_mask_ratio * total_length))
            retain_max_len = max(motif_len_max, int(max_mask_ratio * total_length))

            start, end = motif_start_end[i]

            if start == 0 and end == 0:
                retain_length = torch.randint(retain_min_len, retain_max_len + 1, (1,)).item()
                retain_start_idx = torch.randint(0, total_length - retain_length + 1, (1,)).item()
                retain_start = effective_indices[retain_start_idx].item()
                retain_end = effective_indices[retain_start_idx + retain_length - 1].item()
            else:
                motif_length = end - start
                if motif_length < retain_min_len:
                    retain_length = retain_min_len
                elif motif_length > retain_max_len:
                    retain_length = retain_max_len
                else:
                    retain_length = motif_length

                if end - start - retain_length > 0:
                    retain_start = torch.randint(start, end - retain_length + 1, (1,)).item()
                else:
                    retain_start = start

                retain_end = retain_start + retain_length - 1

            sequence_indices = torch.arange(sequence_length, device=target.device)
            mask = non_special_sym_mask & ((sequence_indices < retain_start) | (sequence_indices > retain_end))
            masked_target = current_target.clone()
            masked_target[mask] = self.mask_id

            masked_targets.append(masked_target)

        return torch.stack(masked_targets)

    def get_motif_middle(self, target, motif_start_end, motif_len_min=10, motif_len_max=30):

        batch_size, sequence_length = target.shape
        masked_targets = []

        for i in range(batch_size):
            current_target = target[i].clone()
            if sum(motif_start_end[i]) == 0:
                non_special_sym_mask = (
                        (current_target != self.pad_id) &
                        (current_target != self.bos_id) &
                        (current_target != self.eos_id)
                )
                effective_indices = torch.where(non_special_sym_mask)[0]
                if len(effective_indices) == 0:
                    masked_targets.append(torch.full_like(current_target, fill_value=self.mask_id))
                    continue

                start = effective_indices[0].item()
                end = effective_indices[-1].item()
            else:
                start, end = motif_start_end[i]

            motif_length = end - start

            if motif_length < motif_len_min:
                crop_len = motif_length
            else:
                crop_len = min(torch.randint(motif_len_min, min(motif_len_max, motif_length) + 1, (1,)).item(), motif_length)

            non_special_sym_mask = (
                    (current_target != self.pad_id) &
                    (current_target != self.bos_id) &
                    (current_target != self.eos_id)
            )

            effective_indices = torch.where(non_special_sym_mask)[0]
            if len(effective_indices) == 0:
                masked_targets.append(torch.full_like(current_target, fill_value=self.mask_id))
                continue

            middle_position = (effective_indices[0] + effective_indices[-1]) // 2
            crop_start = max(middle_position - crop_len // 2, effective_indices[0])
            crop_end = min(crop_start + crop_len, effective_indices[-1] + 1)
            crop_start = crop_end - crop_len

            masked_target = current_target.clone()
            masked_target[non_special_sym_mask] = self.mask_id
            masked_target[crop_start:crop_end] = current_target[crop_start:crop_end]

            masked_targets.append(masked_target)

        masked_target = torch.stack(masked_targets)

        return masked_target

    def construct_x_t(self, struct_target, aatype_target):
        bsz = struct_target.size(0)
        # seperately add noise to struct and aa
        struct_t = torch.randint(
            1,
            self.cfg.num_diffusion_timesteps + 1,
            (bsz,),
            device=struct_target.device,
        )
        aatype_t = torch.randint(
            1,
            self.cfg.num_diffusion_timesteps + 1,
            (bsz,),
            device=aatype_target.device,
        )

        assert (
            self.cfg.single_modality_ratio
            + self.cfg.folding_loss_ratio
            + self.cfg.inverse_folding_loss_ratio
            + self.cfg.joint_loss_ratio
            + self.cfg.independent_loss_ratio
            == 1.0
        )

        split_sizes = [
            int(bsz * self.cfg.single_modality_ratio),
            int(bsz * self.cfg.folding_loss_ratio),
            int(bsz * self.cfg.inverse_folding_loss_ratio),
            int(bsz * self.cfg.independent_loss_ratio),
            int(bsz * self.cfg.joint_loss_ratio),
        ]
        split_sizes[-1] = bsz - sum(split_sizes[:-1])

        rand_index = torch.randperm(bsz).type_as(struct_target)
        int_index_list = torch.split(rand_index, split_sizes)

        bool_index_list = []
        for int_index in int_index_list:
            bool_index = torch.zeros(bsz, dtype=torch.bool).to(
                struct_target.device
            )
            bool_index[int_index] = True
            bool_index_list.append(bool_index)

        (
            single_modality_index,
            folding_index,
            inverse_folding_index,
            independent_index,
            joint_index,
        ) = bool_index_list

        struct_t = struct_t.masked_fill(inverse_folding_index, 0)
        struct_type_id = self.get_modality_type(struct_target)
        struct_x_t, struct_loss_mask = self.q_sample(
            struct_target,
            struct_t,
            struct_type_id,
            maskable_mask=self.get_non_special_symbol_mask(struct_target),
        )
        aatype_t = aatype_t.masked_fill(folding_index, 0)
        aatype_t = aatype_t.masked_scatter(joint_index, struct_t[joint_index])
        aa_type_id = self.get_modality_type(aatype_target)
        aatype_x_t, aa_loss_mask = self.q_sample(
            aatype_target,
            aatype_t,
            aa_type_id,
            maskable_mask=self.get_non_special_symbol_mask(aatype_target),
        )

        return (
            {"t": struct_t, "x_t": struct_x_t, "mask": struct_loss_mask},
            {"t": aatype_t, "x_t": aatype_x_t, "mask": aa_loss_mask},
            single_modality_index,
        )

    
    def compute_loss(self, batch, weighting='constant'):
        target = batch['targets']

        struct_target = batch["struct_tokens"]["targets"]
        aatype_target = batch["aatype_tokens"]["targets"]

        (
            struct_noised,
            aatype_noised,
            single_modality_index,
        ) = self.construct_x_t(struct_target, aatype_target)
        x_t = torch.concat([struct_noised["x_t"], aatype_noised["x_t"]], dim=1)

        masked_target = None
        if self.cfg.cond.use_seq_motif:
            if random.random() < 0.5:
                masked_target = self.get_motif_original(target, batch['motif_start_end'], motif_len_min=self.cfg.cond.motif_min_len, motif_len_max=self.cfg.cond.motif_max_len)
            else:
                masked_target = self.get_motif_middle(target, batch['motif_start_end'], motif_len_min=self.cfg.cond.motif_min_len, motif_len_max=self.cfg.cond.motif_max_len)

        motif_struct_emb = None
        if self.use_motif_struct_emb:
            # print(f"use motif_struct_emb")
            motif_struct_emb = batch['motif_struct_emb']
            # print(motif_struct_emb)

        inputs = dict(x_t=x_t, seq_cond=masked_target, go=batch['go_type'], ipr=batch['ipr_type'], ec=batch['ec_type'], motif_struct_emb=motif_struct_emb)

        model_outputs = self.forward(
                input_ids=inputs,
                single_modality=single_modality_index,
            )

        struct_logits, aatype_logits = model_outputs["logits"].chunk(2, dim=1)
        num_timesteps = self.cfg.num_diffusion_timesteps
        struct_weight = {
            "linear": (
                num_timesteps - (struct_noised["t"] - 1)
            ),  # num_timesteps * (1 - (t-1)/num_timesteps)
            "constant": num_timesteps * torch.ones_like(struct_noised["t"]),
        }[weighting][:, None].float() / num_timesteps
        struct_weight = struct_weight.expand(struct_target.size())

        aatype_weight = {
            "linear": (
                num_timesteps - (aatype_noised["t"] - 1)
            ),  # num_timesteps * (1 - (t-1)/num_timesteps)
            "constant": num_timesteps * torch.ones_like(aatype_noised["t"]),
        }[weighting][:, None].float() / num_timesteps
        aatype_weight = aatype_weight.expand(aatype_target.size())

        if self.use_diff_ce:
            # print("in diff ce ")
            struct_t = struct_noised["t"].float()
            aatype_t = aatype_noised["t"].float()
            
            motif_mask = batch['motif_mask']

            # 结构权重计算（基于struct_t）
            struct_time_factor = (struct_t - 1) / (num_timesteps)  # 归一化到[0,1]
            struct_motif_weight_coeff = 1.0 + 0.15 * torch.exp(-2.5 * (1 - struct_time_factor))
            struct_motif_weight_coeff = struct_motif_weight_coeff[:, None].expand(-1, motif_mask.shape[1])

            # print(f"motif_mask: {motif_mask.shape}, struct_motif_weight_coeff: {struct_motif_weight_coeff.shape}, struct_weight: {struct_weight.shape}")
            # assert motif_mask.shape == struct_motif_weight_coeff.shape == struct_weight.shape

            # print(f"sum motif_mask: {motif_mask.sum()}, sum struct_motif_weight_coeff: {struct_motif_weight_coeff.sum()}, sum struct_weight: {struct_weight.sum()}")
            # 应用结构motif权重
            struct_weight_coeff = torch.where(~motif_mask, struct_motif_weight_coeff, 1.0)
            struct_weight = struct_weight * struct_weight_coeff

            # 氨基酸类型权重计算（基于aatype_t）
            aatype_time_factor = (aatype_t - 1) / (num_timesteps)  # 归一化到[0,1]
            aatype_motif_weight_coeff = 1.0 + 0.15 * torch.exp(-2.5 * (1 - aatype_time_factor))
            aatype_motif_weight_coeff = aatype_motif_weight_coeff[:, None].expand(-1, motif_mask.shape[1])
            # 应用氨基酸类型motif权重
            aatype_weight_coeff = torch.where(~motif_mask, aatype_motif_weight_coeff, 1.0)
            aatype_weight = aatype_weight * aatype_weight_coeff

        return (
            {
                "aatype": aatype_logits,
                "struct": struct_logits,
            },  # model pred logits
            {
                "aatype": aatype_target,
                "struct": struct_target,
            },  # training targets
            {  # training loss mask
                "aatype": aatype_noised["mask"],
                "struct": struct_noised["mask"],
            },
            {
                "aatype": aatype_weight,
                "struct": struct_weight,
            },  # training loss weight
        )

    def forward_encoder(self, batch, **kwargs):
        return {}

    def get_non_special_symbol_mask(self, output_tokens, partial_masks=None):
        non_special_symbol_mask = (
            output_tokens.ne(self.pad_id)
            & output_tokens.ne(self.aa_bos_id)
            & output_tokens.ne(self.aa_eos_id)
            & output_tokens.ne(self.struct_bos_id)
            & output_tokens.ne(self.struct_eos_id)
        )
        if partial_masks is not None:
            non_special_symbol_mask &= ~partial_masks
        return non_special_symbol_mask

    def initialize_output_tokens(
        self, input_tokens, partial_masks=None, **kwargs
    ):
        type_ids = self.get_modality_type(input_tokens)
        output_mask = self.get_non_special_symbol_mask(
            input_tokens, partial_masks=partial_masks
        )
        # fill the aatype part and struct part with specialized mask token
        aa_position = type_ids.eq(self.aa_type) & output_mask
        struct_position = type_ids.eq(self.struct_type) & output_mask
        output_tokens = input_tokens.masked_fill(aa_position, self.aa_mask_id)
        output_tokens = output_tokens.masked_fill(
            struct_position, self.struct_mask_id
        )
        output_scores = torch.zeros_like(output_tokens, dtype=torch.float)

        return output_tokens, output_scores

    def resample_conditional(self, _tokens, _scores, ratio, scale, go=None, ipr=None, seq_cond=None, ec=None):
        to_be_resample_idx = []
        resample_input = []
        resample_input_mask = []
        resample_input_scores = []
        resample_input_seq_cond = []
        for i, seq in enumerate(_tokens):
            most_token_dict = {}
            most_token = None
            most_token_num = -1
            for j, token in enumerate(seq):
                
                token = int(token)
                if token == self.pad_id or token >= 33:
                    # just check aa
                    continue
                
                if token not in most_token_dict:
                    most_token_dict[token] = [j]
                else:
                    most_token_dict[token].append(j)
                if len(most_token_dict[token]) > most_token_num:
                    most_token = token
                    most_token_num = len(most_token_dict[token])
            if most_token_num > len(seq) * ratio * 0.5:#max(0.3/(step+1) ** 0.2, 0.1):
                to_be_resample_idx.append(i)
                resample_input_scores.append(_scores[i])
                mask = torch.zeros_like(seq).bool()
                for k, v in most_token_dict.items():
                    if len(v) > len(seq) * ratio * 0.5:#max(0.3/(step+1) ** 0.2, 0.1):
                        mask |= seq.eq(k)
                # resample_input_mask.append(mask)
                # resample_input.append(seq.masked_fill(mask, self.aa_mask_id))

                seq = seq.masked_fill(mask, self.aa_mask_id)
                
                struct_mask = torch.zeros_like(seq).bool()
                for id, value in enumerate(mask):
                    if value:
                        struct_mask[id - (len(seq) // 2)] = value

                seq = seq.masked_fill(struct_mask, self.struct_mask_id)

                all_mask = struct_mask | mask
                resample_input_mask.append(all_mask)
                resample_input.append(seq)
                
                if seq_cond is not None:
                    raise NotImplementedError
                    # resample_input_seq_cond.append(seq_cond[i].masked_fill(mask, self.mask_id))
                #resample_input.append(seq.masked_scatter(mask, xt[i][mask]))
            
        if len(to_be_resample_idx) > 0:
            resample_input = torch.stack(resample_input, dim=0).type_as(_tokens)
            resample_input_scores = torch.stack(resample_input_scores, dim=0).type_as(_scores)
            resample_input_mask = torch.stack(resample_input_mask, dim=0).type_as(_tokens).bool()
            if seq_cond is not None:
                raise NotImplementedError
                resample_input_seq_cond = torch.stack(resample_input_seq_cond, dim=0).type_as(_tokens)

            inputs = dict(x_t=resample_input, go=go, ipr=ipr, seq_cond=resample_input_seq_cond if seq_cond is not None else None, ec=ec)

            type_ids = self.get_modality_type(_tokens)
            
            resample_logits = self.net(
                input_ids=inputs, type_ids=type_ids
            )['logits']
            if resample_logits.dtype != _scores.dtype:
                resample_logits = resample_logits.type_as(_scores)

            
            output_masks = self.get_non_special_symbol_mask(_tokens, partial_masks=None)

            aa_position = type_ids.eq(self.aa_type) & output_masks
            struct_position = type_ids.eq(self.struct_type) & output_masks
            indices_aa = torch.where(aa_position)
            indices_struct = torch.where(struct_position)

            # HACK: all amino acid token id < 33, while all struct token id >= 33
            resample_logits[indices_aa[0], indices_aa[1], 33:] = -math.inf
            resample_logits[indices_struct[0], indices_struct[1], :33] = -math.inf

            resample_logits[..., self.special_token_list] = -math.inf

            resample_logits = top_k_top_p_filtering(resample_logits, top_p=0.95)
            #noise_scale = 1.5 - 0.2 * ((step + 1) / max_step)
            noise_scale = scale
            assert resample_logits.size(0) == len(to_be_resample_idx)
            resample_tokens, resample_scores = stochastic_sample_from_categorical(resample_logits, temperature=0.0, noise_scale=noise_scale)
            resample_input.masked_scatter_(resample_input_mask, resample_tokens[resample_input_mask])
            resample_input_scores.masked_scatter_(resample_input_mask, resample_scores[resample_input_mask])
            _tokens[to_be_resample_idx], _scores[to_be_resample_idx] = resample_input, resample_input_scores
            
    def forward_decoder(self, prev_decoder_out, need_attn_weights=False, partial_masks=None,
                        sampling_strategy='gumbel_argmax', go_label=None, ipr_label=None, seq_cond=None, ec_label=None,):
        output_tokens = prev_decoder_out['output_tokens'].clone()
        output_scores = prev_decoder_out['output_scores'].clone()
        step, max_step = prev_decoder_out['step'], prev_decoder_out['max_step']
        temperature = prev_decoder_out['temperature']
        history = prev_decoder_out['history']

        output_masks = self.get_non_special_symbol_mask(output_tokens, partial_masks=partial_masks)

        inputs = dict(x_t=output_tokens, go=go_label, ipr=ipr_label, seq_cond=seq_cond, ec=ec_label)

        input_mask = output_tokens.ne(self.pad_id)
        L = output_tokens.shape[1]
        num_heads = self.net.config.num_attention_heads
        attention_bias: torch.FloatType = (
            self.net.esm.get_extended_attention_mask(
                input_mask, output_tokens.shape
            ).repeat(1, num_heads, L, 1)
        )
        # print(attention_bias.shape)

        net_out = self.net(input_ids=inputs,attention_mask=attention_bias,type_ids=self.get_modality_type(output_tokens))

        # TODO: BUG?? 没取logsoftmax，检查要不要logsoftmax，后续处理也有存在logsoftmax的地方
        # 但是不影响
        logits = net_out['logits']
        # logits = net_out["logits"].log_softmax(dim=-1)
        attentions = net_out['attentions'] if need_attn_weights else None
        
        if logits.dtype != output_scores.dtype:
            logits = logits.type_as(output_scores)

        type_ids = self.get_modality_type(output_tokens)
        aa_position = type_ids.eq(self.aa_type) & output_masks
        struct_position = type_ids.eq(self.struct_type) & output_masks
        indices_aa = torch.where(aa_position)
        indices_struct = torch.where(struct_position)

        # HACK: all amino acid token id < 33, while all struct token id >= 33
        logits[indices_aa[0], indices_aa[1], 33:] = -math.inf
        logits[indices_struct[0], indices_struct[1], :33] = -math.inf

        logits[..., self.special_token_list] = -math.inf
        
        # # logits = top_k_top_p_filtering(logits, top_p=0.95)

        # if sampling_strategy == 'vanilla':
        #     _tokens, _scores = sample_from_categorical(logits, temperature=temperature)
        # elif sampling_strategy == 'argmax':
        #     _scores, _tokens = logits.max(-1)
        # elif sampling_strategy == 'gumbel_argmax':
        #     noise_scale = 1.0
        #     _tokens, _scores = stochastic_sample_from_categorical(logits, temperature=0.0, noise_scale=noise_scale)

        #     # 针对batch中的seq，如果一条seq的某aa类型频率过高就mask这条seq的对应aa类型的位置，重新加上功能条件sample一次
        #     self.resample_conditional(_tokens, _scores, ratio=0.25, scale=1.0, go=go_label, ipr=ipr_label, seq_cond=seq_cond, ec=ec_label)
        # else:
        #     raise NotImplementedError
        
        # output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        # output_scores.masked_scatter_(output_masks, _scores[output_masks])

        # history.append(output_tokens.clone())

        # return dict(
        #     output_tokens=output_tokens,
        #     output_scores=output_scores,
        #     attentions=attentions, # [B, L, H, T, T]
        #     step=step + 1,
        #     max_step=max_step,
        #     history=history,
        #     hidden_states=net_out['last_hidden_state']
        # )

        logits = top_k_top_p_filtering(logits, top_p=0.95)

        if sampling_strategy == "argmax":
            _scores, _tokens = logits.max(-1)
        elif sampling_strategy == "gumbel_argmax":
            noise_scale = temperature
            # TODO：结尾有logits的logmax
            _tokens, _scores = stochastic_sample_from_categorical(
                logits, temperature=0.0, noise_scale=noise_scale
            )

            self.resample_conditional(_tokens, _scores, ratio=0.25, scale=1.0, go=go_label, ipr=ipr_label, seq_cond=seq_cond, ec=ec_label)

            _tokens.masked_scatter_(
                ~output_masks, output_tokens[~output_masks]
            )

            
        elif sampling_strategy.startswith("annealing"):
            max_temp, min_temp = map(
                float, sampling_strategy.split("@")[1].split(":")
            )
            rate = 1 - step / max_step
            temperature = min_temp + (max_temp - min_temp) * rate
            _tokens, _scores = sample_from_categorical(
                logits, temperature=temperature
            )

            self.resample_conditional(_tokens, _scores, ratio=0.25, scale=1.0, go=go_label, ipr=ipr_label, seq_cond=seq_cond, ec=ec_label)
        else:
            _tokens, _scores = sample_from_categorical(
                logits, temperature=temperature
            )

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        history.append(output_tokens.clone())

        return dict(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attentions=attentions,
            step=step + 1,
            max_step=max_step,
            history=history,
            hidden_states=net_out["last_hidden_state"],
        )

    # def _reparam_decoding(
    #     self,
    #     output_tokens,
    #     output_scores,
    #     cur_tokens,
    #     cur_scores,
    #     decoding_strategy,
    #     xt_neq_x0,
    #     non_special_sym_mask,
    #     t,
    #     max_step,
    #     noise,
    # ):
    #     """
    #         This function is used to perform reparameterized decoding.
    #     """
    #     # output_tokens: [B, N]
    #     # output_scores: [B, N]
    #     # cur_tokens: [B, N]
    #     # cur_scores: [B, N]
    #     # xt_neq_x0: equivalent to not_b_t [B, N]
    #     # non_special_sym_mask: [B, N]
    #     # noise: either [B, N] or scalar (if using the mask noise)

    #     # decoding_strategy needs to take the form of "reparam-<conditioning>-<topk_mode>-<schedule>"
    #     _, condition, topk_mode, schedule = decoding_strategy.split("-")

    #     # first set the denoising rate according to the schedule
    #     if schedule == "linear":
    #         rate = 1 - t / max_step
    #     elif schedule == "cosine":
    #         rate = np.cos(t / max_step * np.pi * 0.5)
    #     else:
    #         raise NotImplementedError

    #     # compute the cutoff length for denoising top-k positions
    #     cutoff_len = (
    #         non_special_sym_mask.sum(1, keepdim=True).type_as(output_scores) * rate
    #     ).long()
    #     # set the scores of special symbols to a large value so that they will never be selected
    #     _scores_for_topk = cur_scores.masked_fill(~non_special_sym_mask, 1000.0)
        
    #     to_be_resample = []
    #     for i, seq in enumerate(cur_tokens):
    #         most_token_dict = {}
    #         most_token = None
    #         most_token_num = -1
    #         for j, token in enumerate(seq):
    #             token = int(token)
    #             if token == self.pad_id:
    #                 continue
    #             if token not in most_token_dict:
    #                 most_token_dict[token] = [j]
    #             else:
    #                 most_token_dict[token].append(j)
    #             if len(most_token_dict[token]) > most_token_num:
    #                 most_token = token
    #                 most_token_num = len(most_token_dict[token])
    #         if most_token_num > len(seq) * 0.25:
    #             to_be_resample.append(i)
                
    #     # the top-k selection can be done in two ways: stochastic by injecting Gumbel noise or deterministic
    #     if topk_mode.startswith("stochastic"):
    #         noise_scale = float(topk_mode.replace("stochastic", ""))
    #         lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=True, temp=noise_scale * rate)
    #     elif topk_mode == "deterministic":
    #         lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=False)
    #         if len(to_be_resample) > 0:
    #             noise_scale = 1.5
    #             #print(lowest_k_mask[to_be_resample[0]])
    #             lowest_k_mask[to_be_resample] = topk_masking(_scores_for_topk[to_be_resample], cutoff_len[to_be_resample], 
    #                                                          stochastic=True, temp=noise_scale * rate)
    #     else:
    #         raise NotImplementedError

    #     # Various choices to generate v_t := [v1_t, v2_t].
    #     # Note that
    #     #   v1_t governs the outcomes of tokens where b_t = 1,
    #     #   v2_t governs the outcomes of tokens where b_t = 0.

    #     # #### the `uncond` mode ####
    #     # In our reparameterized decoding,
    #     # both v1_t and v2_t can be fully determined by the current token scores .

    #     # #### the `cond` mode ####
    #     # However, we can also impose some conditional constraints on v1_t so that
    #     # the decoding can be performed in a more conservative manner.
    #     # For example, we can set v1_t = 0 only when
    #     # (the newly output tokens are the same as previous denoised results, AND
    #     # the current token score becomes lower, AND
    #     # the current token score is not in the top-k share among all tokens).
    #     if condition == "cond":
    #         not_v1_t = (cur_tokens == output_tokens) & (cur_scores < output_scores) & lowest_k_mask
    #     elif condition == "uncond":
    #         not_v1_t = lowest_k_mask
    #     else:
    #         raise NotImplementedError

    #     # for b_t = 0, the token is set to noise if it is in the lowest k scores.
    #     not_v2_t = lowest_k_mask

    #     last_mask_position = xt_neq_x0
    #     masked_to_noise = (~xt_neq_x0 & not_v1_t) | (xt_neq_x0 & not_v2_t)
    #     if isinstance(noise, torch.Tensor):
    #         output_tokens.masked_scatter_(masked_to_noise, noise[masked_to_noise])
    #     elif isinstance(noise, (int, float)):
    #         output_tokens.masked_fill_(masked_to_noise, noise)
    #     else:
    #         raise NotImplementedError("noise should be either a tensor or a scalar")
    #     output_scores.masked_fill_(masked_to_noise, -math.inf)

    #     masked_to_x0 = xt_neq_x0 & ~not_v2_t
    #     output_tokens.masked_scatter_(masked_to_x0, cur_tokens[masked_to_x0])
    #     output_scores.masked_scatter_(masked_to_x0, cur_scores[masked_to_x0])
    #     assert ((masked_to_x0 & last_mask_position) == masked_to_x0).all()
    #     # b_{t} = (b_{t+1} & u_t) | v_t
    #     # For convenience, save the NOT of b_t for the next iteration
    #     # NOT_b_{t} = (NOT_b_{t+1} | not_v1_t) & not_v2_t
    #     #
    #     # # When condition is 'uncond', the not_v1_t is equal to not_v2_t, the new_xt_neq_x0 is always equal to not_v1/v2_t
    #     new_xt_neq_x0 = (xt_neq_x0 | not_v1_t) & not_v2_t
    #     assert (new_xt_neq_x0 == not_v2_t).all()
    #     return new_xt_neq_x0, output_tokens, output_scores

    def _reparam_decoding(
        self,
        output_tokens,
        output_scores,
        cur_tokens,
        cur_scores,
        decoding_strategy,
        xt_neq_x0,
        type_ids,
        non_special_sym_mask,
        t,
        max_step,
    ):
        def _reparam_process(
            output_tokens,
            output_scores,
            cur_tokens,
            cur_scores,
            xt_neq_x0,
            noise,
            non_special_sym_mask,
        ):
            """This function is used to perform reparameterized decoding.

            output_tokens: [B, N]
            output_scores: [B, N]
            cur_tokens: [B, N]
            cur_scores: [B, N]
            xt_neq_x0: equivalent to not_b_t [B, N]
            non_special_sym_mask: [B, N]
            noise: either [B, N] or scalar (if using the mask noise)
            """

            # decoding_strategy needs to take the form of "reparam-<conditioning>-<topk_mode>-<schedule>"
            _, condition, topk_mode, schedule = decoding_strategy.split("-")

            # first set the denoising rate according to the schedule
            if schedule == "linear":
                rate = 1 - t / max_step
            elif schedule == "cosine":
                rate = np.cos(t / max_step * np.pi * 0.5)
            else:
                raise NotImplementedError

            # compute the cutoff length for denoising top-k positions
            cutoff_len = (
                non_special_sym_mask.sum(1, keepdim=True).type_as(
                    output_scores
                )
                * rate
            ).long()
            # set the scores of special symbols to a large value so that they will never be selected
            _scores_for_topk = cur_scores.masked_fill(
                ~non_special_sym_mask, 1000.0
            )

            # the top-k selection can be done in two ways: stochastic by injecting Gumbel noise or deterministic
            if topk_mode.startswith("stochastic"):
                noise_scale = float(topk_mode.replace("stochastic", ""))
                lowest_k_mask = topk_masking(
                    _scores_for_topk,
                    cutoff_len,
                    stochastic=True,
                    temp=noise_scale * rate,
                )
            elif topk_mode == "deterministic":
                lowest_k_mask = topk_masking(
                    _scores_for_topk, cutoff_len, stochastic=False
                )

            elif topk_mode == "positionprior":
                lowest_k_mask_1 = topk_masking_prior(
                    _scores_for_topk, cutoff_len, stochastic=False
                )
                lowest_k_mask_2 = topk_masking_prior(
                    _scores_for_topk, cutoff_len, stochastic=False
                )
                lowest_k_mask = lowest_k_mask_1 | lowest_k_mask_2
            else:
                raise NotImplementedError

            # Various choices to generate v_t := [v1_t, v2_t].
            # Note that
            #   v1_t governs the outcomes of tokens where b_t = 1,
            #   v2_t governs the outcomes of tokens where b_t = 0.

            # #### the `uncond` mode ####
            # In our reparameterized decoding,
            # both v1_t and v2_t can be fully determined by the current token scores .

            # #### the `cond` mode ####
            # However, we can also impose some conditional constraints on v1_t so that
            # the decoding can be performed in a more conservative manner.
            # For example, we can set v1_t = 0 only when
            # (the newly output tokens are the same as previous denoised results, AND
            # the current token score becomes lower, AND
            # the current token score is not in the top-k share among all tokens).
            if condition == "cond":
                not_v1_t = (
                    (cur_tokens == output_tokens)
                    & (cur_scores < output_scores)
                    & lowest_k_mask
                )
            elif condition == "uncond":
                not_v1_t = lowest_k_mask
            else:
                raise NotImplementedError

            # for b_t = 0, the token is set to noise if it is in the lowest k scores.
            not_v2_t = lowest_k_mask

            last_mask_position = xt_neq_x0

            masked_to_noise = (~xt_neq_x0 & not_v1_t) | (xt_neq_x0 & not_v2_t)
            if isinstance(noise, torch.Tensor):
                output_tokens.masked_scatter_(
                    masked_to_noise, noise[masked_to_noise]
                )
            elif isinstance(noise, (int, float)):
                output_tokens.masked_fill_(masked_to_noise, noise)
            else:
                raise NotImplementedError(
                    "noise should be either a tensor or a scalar"
                )
            output_scores.masked_fill_(masked_to_noise, -math.inf)

            masked_to_x0 = xt_neq_x0 & ~not_v2_t
            output_tokens.masked_scatter_(
                masked_to_x0, cur_tokens[masked_to_x0]
            )
            output_scores.masked_scatter_(
                masked_to_x0, cur_scores[masked_to_x0]
            )
            assert ((masked_to_x0 & last_mask_position) == masked_to_x0).all()
            # b_{t} = (b_{t+1} & u_t) | v_t
            # For convenience, save the NOT of b_t for the next iteration
            # NOT_b_{t} = (NOT_b_{t+1} | not_v1_t) & not_v2_t
            #
            # # When condition is 'uncond', the not_v1_t is equal to not_v2_t, the new_xt_neq_x0 is always equal to not_v1/v2_t (?)
            new_xt_neq_x0 = (xt_neq_x0 | not_v1_t) & not_v2_t
            assert (new_xt_neq_x0 == not_v2_t).all()
            return new_xt_neq_x0, output_tokens, output_scores

        aa_position = type_ids.eq(self.aa_type) & non_special_sym_mask
        struct_position = type_ids.eq(self.struct_type) & non_special_sym_mask
        new_xt_neq_x0 = xt_neq_x0.clone()
        new_xt_neq_x0_aa = new_xt_neq_x0.fill_(False)
        new_xt_neq_x0_struct = new_xt_neq_x0.fill_(False)
        if aa_position.any():
            new_xt_neq_x0_aa, output_tokens, output_scores = _reparam_process(
                output_tokens=output_tokens,
                output_scores=output_scores,
                cur_tokens=cur_tokens,
                cur_scores=cur_scores,
                xt_neq_x0=xt_neq_x0 & aa_position,
                noise=self.aa_mask_id,
                non_special_sym_mask=aa_position,
            )
        if struct_position.any():
            (
                new_xt_neq_x0_struct,
                output_tokens,
                output_scores,
            ) = _reparam_process(
                output_tokens=output_tokens,
                output_scores=output_scores,
                cur_tokens=cur_tokens,
                cur_scores=cur_scores,
                xt_neq_x0=xt_neq_x0 & struct_position,
                noise=self.struct_mask_id,
                non_special_sym_mask=struct_position,
            )
        new_xt_neq_x0 = new_xt_neq_x0_aa | new_xt_neq_x0_struct
        return new_xt_neq_x0, output_tokens, output_scores

    def generate(
        self, 
        batch,
        max_iter=None, 
        temperature=1.0, 
        partial_masks=None,
        unmasking_strategy="stochastic1.0",
        sampling_strategy='gumbel_argmax'
    ):
        # tokenizer = tokenizer
        # max_iter = max_iter
        # temperature = temperature
        self.eval()
        max_iter = max_iter
        temperature = temperature

        # 0) encoding
        encoder_out = self.forward_encoder(batch)
        # 1) initialized from all mask tokens, where partial_masks will fix motif
        initial_output_tokens, initial_output_scores = self.initialize_output_tokens(
            batch.get("input_ids"), encoder_out=encoder_out, partial_masks=partial_masks)  #
        prev_decoder_out = dict(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            output_masks=None,
            attentions=None,
            step=0,
            max_step=max_iter,
            history=[initial_output_tokens.clone()],
            temperature=temperature,
            type_ids=self.get_modality_type(initial_output_tokens),
        )

        prev_decoder_out['output_masks'] = self.get_non_special_symbol_mask(
                prev_decoder_out['output_tokens'], partial_masks=partial_masks
            )
        
        history_detail = []
        last_mask = prev_decoder_out["output_masks"].clone()
        
        for step in tqdm(range(max_iter), desc='Decoding'):
            # 2.1: predict
            with torch.no_grad():
                decoder_out = self.forward_decoder(
                    prev_decoder_out=prev_decoder_out,
                    partial_masks=partial_masks,
                    sampling_strategy=sampling_strategy,
                    go_label=batch.get('go_label', None),
                    ipr_label=batch.get('ipr_label', None),
                    seq_cond=batch.get('seq_cond', None),
                    ec_label=batch.get('ec_label', None),

                )

            output_tokens = decoder_out['output_tokens']
            output_scores = decoder_out['output_scores']

            # 2.2: re-mask skeptical parts of low confidence
            non_special_sym_mask = self.get_non_special_symbol_mask(
                prev_decoder_out['output_tokens'], partial_masks=partial_masks
            )
            
            (
                output_masks,
                result_tokens,
                result_scores,
            ) = self._reparam_decoding(
                output_tokens=prev_decoder_out["output_tokens"].clone(),
                output_scores=prev_decoder_out["output_scores"].clone(),
                cur_tokens=output_tokens.clone(),
                cur_scores=output_scores.clone(),
                decoding_strategy=f"reparam-uncond-{unmasking_strategy}-linear",
                xt_neq_x0=prev_decoder_out["output_masks"],
                type_ids=prev_decoder_out["type_ids"].clone(),
                non_special_sym_mask=non_special_sym_mask,
                t=step + 1,
                max_step=max_iter,
            )

            demask_pos = ((last_mask == 1) & (output_masks == 0)).nonzero(as_tuple=True)
            remask_pos = ((last_mask == 0) & (output_masks == 1)).nonzero(as_tuple=True)
            history_detail.append({
                "step": step + 1,
                "tokens": result_tokens.cpu().tolist(),
                "mask": output_masks.cpu().tolist(),
                "demask_pos": [x for x in zip(*[d.cpu().tolist() for d in demask_pos])],
                "remask_pos": [x for x in zip(*[d.cpu().tolist() for d in remask_pos])],
                "pred_tokens": decoder_out["output_tokens"],
            })
            last_mask = output_masks.clone()

            
            # Note Original
            # output_masks, result_tokens, result_scores = self._reparam_decoding(
            #     output_tokens=prev_decoder_out['output_tokens'].clone(),
            #     output_scores=prev_decoder_out['output_scores'].clone(),
            #     cur_tokens=output_tokens.clone(),
            #     cur_scores=output_scores.clone(),
            #     decoding_strategy='reparam-uncond-deterministic-linear',#'reparam-uncond-stochastic1.0-linear'
            #     xt_neq_x0=prev_decoder_out['output_masks'],
            #     non_special_sym_mask=non_special_sym_mask,
            #     t=step + 1,
            #     max_step=max_iter,
            #     noise=self.mask_id,
            # )

            # output_masks, result_tokens, result_scores = self._reparam_decoding(
            #     output_tokens=output_tokens.clone(),#output_tokens,#
            #     output_scores=output_scores.clone(),#output_scores,##
            #     cur_tokens=prev_decoder_out['output_tokens'].clone(),#prev_decoder_out['output_tokens'],##
            #     cur_scores=prev_decoder_out['output_scores'].clone(),#prev_decoder_out['output_scores'],##
            #     decoding_strategy='reparam-uncond-deterministic-linear',#'reparam-uncond-stochastic1.0-linear',#,##
            #     # decoding_strategy='reparam-uncond-deterministic-cosine',
            #     xt_neq_x0=prev_decoder_out['output_masks'],
            #     non_special_sym_mask=non_special_sym_mask,
            #     t=step + 1,
            #     max_step=max_iter,
            #     noise=self.mask_id, # if 'init_pred' not in encoder_out else encoder_out['init_pred'],
            #     mask_811=False
            # )
            prev_decoder_out.update(output_masks=output_masks)
            output_tokens = result_tokens
            output_scores = result_scores

            prev_decoder_out.update(
                output_tokens=output_tokens,
                output_scores=output_scores,
                step=step + 1,
                history=decoder_out['history']
            )

        decoder_out = prev_decoder_out
        return decoder_out['output_tokens'], decoder_out['output_scores']