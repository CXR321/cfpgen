
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


from byprot.models.lm.esm_dplm import EsmForDPLM
from byprot.models.lm.esm_cfpgen import EsmForCFPGEN, EsmForCFPGEN_DPLM2
from byprot.models.lm.esm_cfpgen_if import EsmForCFPGenInverseFolding
from byprot.models.lm.dplm2 import MultimodalDiffusionProteinLanguageModel
from dataclasses import dataclass, field
from transformers import AutoModelForMaskedLM, AutoConfig, AutoTokenizer
import torch
import os
# try:
#     from peft import get_peft_model, LoraConfig, TaskType
# except:
#     pass

try:
    from peft import LoraConfig, TaskType, get_peft_model
    from peft.peft_model import PeftModel
except:
    pass

@dataclass
class CondConfig:
    use_go: bool = True
    go_num: int = 375
    go_drop: float = 0.5
    use_ipr: bool = True
    ipr_num: int = 1154
    ipr_drop: float = 0.5
    use_ec: bool = False
    ec_num: int = 661
    ec_drop: float = 0.0
    use_seq_motif: bool = False
    use_struc_bb: bool = False
    motif_min_len: int = 10
    motif_max_len: int = 30

@dataclass
class NetConfig:
    arch_type: str = "esm"
    name: str = "esm2_t33_650M_UR50D"
    dropout: float = 0.1
    pretrain: bool = False
    pretrained_model_name_or_path: str = ""

@dataclass
class LoRAConfig:
    lora: bool = field(
        default=False
    )
    lora_rank: int = field(
        default=16
    )
    lora_dropout: float = field(
        default=0.1
    )
    lora_target_module: str = field(
        default=""
    )
    modules_to_save: str = field(
        default=""
    )

def get_net_class(arch_type):
    if arch_type == 'esm':
        return EsmForDPLM
    # TODO: dplm will support more architectures, such as Llama
    elif arch_type == 'func_esm':
        return EsmForCFPGEN
    elif arch_type == 'func_esm_dplm2':
        return EsmForCFPGEN_DPLM2
    else:
        raise NotImplementedError

def load_state_dict_with_report(net, state_dict, strict=False):
    load_info = net.load_state_dict(state_dict, strict=strict)

    missing_keys = load_info.missing_keys
    unexpected_keys = load_info.unexpected_keys

    total_keys_in_model = len(list(net.state_dict().keys()))
    loaded_keys = total_keys_in_model - len(missing_keys)

    print(f"Successfully loaded {loaded_keys}/{total_keys_in_model} parameters.")
    # print(f"Missing keys: {missing_keys}")
    # print(f"Unexpected keys: {unexpected_keys}")

    return loaded_keys, missing_keys, unexpected_keys

def get_net(cfg):

    if 'net' not in cfg and cfg.arch_type == 'func_esm_if':
        config = AutoConfig.from_pretrained(f'{cfg.name}')
        if 'cond' in cfg:
            config.update(cfg.cond)
        net = EsmForCFPGenInverseFolding(config)
        pretrained_model_name_or_path = cfg.pretrained_model_name_or_path
        # is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isfile(pretrained_model_name_or_path):
            # load your pretrained MLM from local
            state_dict = torch.load(pretrained_model_name_or_path, map_location='cpu')['state_dict']
            # net.load_state_dict(state_dict, strict=True)
            loaded_keys, missing_keys, unexpected_keys = load_state_dict_with_report(net, state_dict, strict=False)

    else:
        if cfg.net.arch_type == 'esm':
            config = AutoConfig.from_pretrained(f'{cfg.net.name}')
            net = EsmForDPLM(config, dropout=cfg.net.dropout)
        elif cfg.net.arch_type == 'func_esm':
            config = AutoConfig.from_pretrained(f'{cfg.net.name}')
            if 'cond' in cfg:
                config.update(cfg.cond)
            net = EsmForCFPGEN(config, dropout=cfg.net.dropout)
        elif cfg.net.arch_type == 'func_esm_dplm2':
            print(f"net name: {cfg.net.name}")
            config = AutoConfig.from_pretrained(f'{cfg.net.name}')
            # print(f"net complete: {net}")
            config.hidden_dropout_prob = cfg.net.dropout
            config.tie_word_embeddings = False
            config.vocab_size = 8229

            config.use_diff_modulation = getattr(cfg, "use_diff_modulation", False)
            config.use_func_cross_attn = getattr(cfg, "use_func_cross_attn", False)
            config.use_motif_struct_emb = getattr(cfg, "use_motif_struct_emb", False)

            # print(f"config: {config}")
            if 'cond' in cfg:
                config.update(cfg.cond)

            pretrained_net = (
                MultimodalDiffusionProteinLanguageModel.from_pretrained(
                    cfg.net.pretrained_model_name_or_path
                ).net
            )
            # print(f"net complete: {net}")

            net = EsmForCFPGEN_DPLM2(config, dropout=cfg.net.dropout)
            # print(f"net complete: {net}")

            if issubclass(type(pretrained_net), PeftModel):
                pretrained_net = pretrained_net.merge_and_unload()
            pretrained_state_dict = pretrained_net.state_dict()
            result = net.load_state_dict(pretrained_state_dict, strict=False)

            print("Missing keys (in model but not in pretrained):", result.missing_keys)
            print("Unexpected keys (in pretrained but not in model):", result.unexpected_keys)

            # exit()


        else:
            raise NotImplementedError

        # 2-stage training (please refer to our paper for more details.)
        ## stage 1: pretrain a masked language model (MLM) from scratch
        ## stage 2: continue pretrain a diffusion language model based on the pretrained MLM
        # TODO: 本质都是load，先屏蔽了
        # if cfg.net.pretrain:
        #     pretrained_model_name_or_path = cfg.net.pretrained_model_name_or_path
        #     # is_local = os.path.isdir(pretrained_model_name_or_path)
        #     is_local = os.path.isfile(pretrained_model_name_or_path)
        #     if is_local:
        #         # load your pretrained MLM from local
        #         state_dict = torch.load(pretrained_model_name_or_path, map_location='cpu')['state_dict']
        #         # net.load_state_dict(state_dict, strict=True)
        #         loaded_keys, missing_keys, unexpected_keys = load_state_dict_with_report(net, state_dict, strict=False)

        #     else:
        #         # or you can load a pretrained MLM from huggingface
        #         ptrn_net = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path)
        #         net.load_state_dict(ptrn_net.state_dict(), strict=True)
        #         del ptrn_net
            
    # activate lora training if possible
    if cfg.lora.lora:
        # QKVO, MLP
        lora_target_module = cfg.lora.lora_target_module
        modules_to_save = cfg.lora.modules_to_save.split(',')+ [
            'esm.encoder.go_embedder',
            'esm.encoder.ipr_embedder',
            'esm.encoder.layer.*.adaLN_modulation'
        ]

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, 
            target_modules=lora_target_module,
            modules_to_save=modules_to_save,
            inference_mode=False, r=cfg.lora.lora_rank, lora_alpha=32, lora_dropout=cfg.lora.lora_dropout
        )
        net = get_peft_model(net, peft_config)
            
    return net

def topk_masking(scores, cutoff_len, stochastic=False, temp=1.0):
    """
    scores: [b, n]
    cutoff_len: [b, 1]
    stochastic: bool, whether to add noise to select top_k or not
    returns:
        mask: [b, n], with 1 if the token is in top-k lowest scores, 0 otherwise
    """
    if stochastic:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len)
    masking = _scores < cutoff
    return masking

def topk_masking_prior(
    scores, cutoff_len, stochastic=False, temp=1.0, prior_mask=None
):
    """
    scores: [b, n]
    cutoff_len: [b, 1]
    stochastic: bool, whether to add noise to select top_k or not
    returns:
        mask: [b, n], with 1 if the token is in top-k lowest scores, 0 otherwise
    """
    if stochastic:
        gumbel_noise = -torch.log(
            -torch.log(torch.rand_like(scores) + 1e-8) + 1e-8
        )
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(
        dim=-1, index=cutoff_len
    )  # + torch.tensor(1e-10)
    # cutoff_len = k -> select k + 1 tokens
    masking = _scores < cutoff
    return masking

def sample_from_categorical(logits=None, temperature=1.0):
    if temperature:
        dist = torch.distributions.Categorical(logits=logits.div(temperature))
        tokens = dist.sample()
        scores = dist.log_prob(tokens)
    else:
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores

def stochastic_sample_from_categorical(logits=None, temperature=1.0, noise_scale=1.0):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    logits = logits + noise_scale * gumbel_noise
    tokens, scores = sample_from_categorical(logits, temperature)
    # scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores

def top_k_top_p_filtering(logits, top_k=0, top_p=0.95, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Basic outline taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    ori_shape = logits.shape
    logits = logits.reshape(-1, ori_shape[-1])
    assert logits.dim() == 2  # [BATCH_SIZE, VOCAB_SIZE]
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    # Replace logits to be removed with -inf in the sorted_logits
    sorted_logits[sorted_indices_to_remove] = filter_value
    # Then reverse the sorting process by mapping back sorted_logits to their original position
    logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))
    logits = logits.reshape(ori_shape) 
    return logits


# def get_struct_tokenizer(
#     model_name_or_path="airkingbd/struct_tokenizer", eval_mode=True
# ):
#     from byprot.models.structok.structok_lfq import VQModel

#     if os.path.exists(model_name_or_path):
#         root_path = f"{model_name_or_path}/.hydra"
#     else:
#         root_path = Path(snapshot_download(repo_id=model_name_or_path, local_dir="./checkpoints"))
#     cfg = load_yaml_config(f"{root_path}/config.yaml")
#     stok = VQModel(**cfg)
#     pretrained_state_dict = torch.load(
#         f"{root_path}/dplm2_struct_tokenizer.ckpt",
#         map_location=torch.device("cpu"),
#     )
#     missing, unexpected = stok.load_state_dict(
#         pretrained_state_dict, strict=False
#     )
#     print(
#         f'Restored from "{model_name_or_path}" with {len(missing)} missing and {len(unexpected)} unexpected keys'
#     )
#     if len(missing) > 0:
#         print(f"Missing Keys: {missing}")
#         print(f"Unexpected Keys: {unexpected}")
#     stok = stok.requires_grad_(False)
#     return stok.train(not eval_mode)
