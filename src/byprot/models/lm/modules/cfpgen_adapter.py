
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from byprot import utils
from byprot.models.lm.model_utils import LoRAConfig, NetConfig, get_net
from byprot.models.lm.dplm import DiffusionProteinLanguageModel

from transformers.models.esm.modeling_esm import EsmAttention, EsmIntermediate, \
                        EsmOutput, EsmSelfOutput, EsmSelfAttention
from transformers import AutoConfig
from copy import deepcopy


logger = utils.get_logger(__name__)



@dataclass
class DPLMWithAdapterConfig:
    mode: str = field(default='adapter')
    num_diffusion_timesteps: int = field(default=100)
    adapter_dropout: float = field(default=0.1)
    encoder_d_model: int = field(default=512)
    dplm_name: str = field(default="")
    net: NetConfig = field(default=NetConfig())
    arch_type: str = field(default="")
    gradient_ckpt: bool = field(default=False)
    rdm_couple: bool = field(default=False)
    cond: dict = field(default_factory=lambda: {})
    lora: dict = field(default_factory=lambda: {})
    name: str = field(default="")
    pretrain: bool = field(default=False)
    pretrained_model_name_or_path: str = field(default="")

    
class DPLMWithConditionalAdatper(nn.Module):
    _default_cfg = DPLMWithAdapterConfig()
    
    @classmethod
    def from_pretrained(cls, cfg):  # load pretrained functional decoder

        train_pnames = []
        dplm_adapter = cls(cfg)

        for pname, param in dplm_adapter.named_parameters():
            if 'adapter' in pname:
                train_pnames.append(pname)
            else:
                param.requires_grad = False

        return dplm_adapter


    def __init__(self, cfg, net=None):
        super().__init__()
        self._update_cfg(cfg)
        
        self.net = get_net(cfg) if net is None else net
        self.tokenizer = self.net.tokenizer
        
        self.mask_id = self.net.mask_id
        self.pad_id = self.net.pad_id
        self.bos_id = self.net.bos_id
        self.eos_id = self.net.eos_id
        self.x_id = self.net.x_id

    def forward(self, batch, encoder_out=None, tokens=None, 
                loss_mask=None, forward_diffusion=False, 
                **kwargs):
        encoder_hidden_states = encoder_out['feats']

        encoder_attention_mask = encoder_out['encoder_attention_mask'] if 'encoder_attention_mask' in encoder_out else batch['prev_tokens'].ne(self.pad_id)

        input_ids = batch['prev_tokens']
        if 'go_type' in batch:
            input_ids = dict(prev_tokens=batch['prev_tokens'], go=batch['go_type'])
        if 'ipr_type' in batch:
            input_ids = dict(prev_tokens=batch['prev_tokens'], go=batch['go_type'], ipr=batch['ipr_type'])
        if 'ec_type' in batch:
            input_ids = dict(prev_tokens=batch['prev_tokens'], go=batch['go_type'], ipr=batch['ipr_type'], ec=batch['ec_type'])

        outputs = self.net(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        return outputs

    def compute_loss(self, batch, weighting='constant', encoder_out=None, tokens=None, label_smoothing=False, return_outputs=False,):
        target = batch['tokens'] if tokens is None else tokens
        partial_masks = torch.zeros_like(target).bool()

        # couple
        t1, t2 = torch.randint(
            1, self.cfg.num_diffusion_timesteps + 1,
            (2 * target.size(0), ),
            device=target.device
        ).chunk(2)

        x_t, t, loss_mask = list(
            self.q_sample_coupled(
                target, t1, t2,
                maskable_mask=self.get_non_special_sym_mask(target, partial_masks)
            ).values()
        )
        # target = target.repeat(2, 1)

        batch['prev_tokens'] = x_t  
        logits = self.forward(batch, encoder_out=encoder_out,
                              loss_mask=loss_mask, forward_diffusion=True)['logits']

        num_timesteps = self.cfg.num_diffusion_timesteps
        weight = {
            "linear": (num_timesteps - (t - 1)),    # num_timesteps * (1 - (t-1)/num_timesteps)
            "constant": num_timesteps * torch.ones_like(t)
        }[weighting][:, None].float() / num_timesteps
        weight = weight.expand(loss_mask.size())

        return logits, batch['tokens'].repeat(2, 1), loss_mask, weight
    
    def _update_cfg(self, cfg):
        # if '_target_' in cfg.denoiser:
        #     cfg.denoiser.pop('_target_')
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
    
    def get_non_special_sym_mask(self, output_tokens, partial_masks=None):
        non_special_sym_mask = (
            output_tokens.ne(self.pad_id) &
            output_tokens.ne(self.bos_id) &
            output_tokens.ne(self.eos_id)
        )
        if partial_masks is not None:
            non_special_sym_mask &= (~partial_masks)
        return non_special_sym_mask
