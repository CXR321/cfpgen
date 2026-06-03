
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


from math import isnan
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchviz import make_dot


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    flag = False
    if target.dim() == lprobs.dim() - 1:
        flag = True
        target = target.unsqueeze(-1)

    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)

    if flag:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, scores: Tensor, target: Tensor, mask=None) -> Tensor:
        """
          scores: [N, ..., C], unnormalized scores
          target: [N, ...]
          mask: [N, ...], where elements with `True` are allowed and `False` are masked-out
        """
        n_tokens = target.numel()
        n_nonpad_tokens = target.ne(self.ignore_index).long().sum()

        bsz, num_classes = scores.shape[0], scores.shape[-1]

        if mask is not None:
            scores = scores[mask]  # [N * len, C]
            target = target[mask]  # [N]
        scores = scores.reshape(-1, num_classes)
        target = target.reshape(-1)

        if self.ignore_index is not None:
            sample_size = target.ne(self.ignore_index).long().sum()
        else:
            sample_size = torch.tensor(target.numel(), device=target.device)

        # smooth_loss = F.cross_entropy(
        #     scores.transpose(1, -1), target,
        #     weight=self.weight,
        #     ignore_index=self.ignore_index, reduction=self.reduction,
        #     label_smoothing=self.label_smoothing)

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs=F.log_softmax(scores, dim=-1),
            target=target,
            epsilon=self.label_smoothing,
            ignore_index=self.ignore_index,
            reduce=True,
        )
        loss_avg = loss / sample_size
        ppl = torch.exp(nll_loss / sample_size)

        logging_output = {
            'nll_loss_sum': nll_loss.data,
            'loss_sum': loss.data,
            'ppl': ppl.data,
            'bsz': bsz,
            'sample_size': sample_size,
            'sample_ratio': sample_size / n_tokens,
            'nonpad_ratio': n_nonpad_tokens / n_tokens
        }
        return loss_avg, logging_output


class Coord2SeqCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, scores: Tensor, target: Tensor, label_mask=None, coord_mask=None, weights=None) -> Tensor:
        """
          scores: [N, L, C], unnormalized scores
          target: [N, L]
          coord_mask: FloatTensor [N, L], where elements with `True` are allowed and `False` are masked-out
        """
        if label_mask is None:
            label_mask = coord_mask

        bsz, num_classes = scores.shape[0], scores.shape[-1]

        n_tokens = target.numel()
        if self.ignore_index is not None:
            sample_size = n_nonpad_tokens = target.ne(self.ignore_index).float().sum()
        else:
            sample_size = n_nonpad_tokens = n_tokens

        # [N, L]
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs=F.log_softmax(scores, dim=-1),
            target=target,
            epsilon=self.label_smoothing,
            ignore_index=self.ignore_index,
            reduce=False,
        )
        if weights is not None:
            loss, nll_loss = loss * weights, nll_loss * weights
        fullseq_loss = loss.sum() / sample_size
        fullseq_nll_loss = nll_loss.sum() / sample_size

        # use coord masked loss for model training,
        # ignoring those position with missing coords (as nan)
        if label_mask is not None:
            label_mask = label_mask.float()
            sample_size = label_mask.sum()  # sample size should be set to valid coordinates
            loss = (loss * label_mask).sum() / sample_size
            nll_loss = (nll_loss * label_mask).sum() / sample_size
        else:
            loss, nll_loss = fullseq_loss, fullseq_nll_loss
        # nll_loss = nll_loss[label_mask] # calculate pesudo-ppl
        ppl = torch.exp(nll_loss)

        logging_output = {
            'nll_loss': nll_loss.data,
            'ppl': ppl.data, # torch.mean(ppl).data,
            'fullseq_loss': fullseq_loss.data,
            'fullseq_nll_loss': fullseq_nll_loss.data,
            'bsz': bsz,
            'sample_size': sample_size,
            'sample_ratio': sample_size / n_tokens,
            'nonpad_ratio': n_nonpad_tokens / n_tokens
        }
        return loss, logging_output


class RDMCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, scores: Tensor, target: Tensor, label_mask=None, weights=None,
                cal_constant_loss=False,
                watch_t1_t2_loss=False,
                ) -> Tensor:
        """
          scores: [N, L, C], unnormalized scores
          target: [N, L]
          coord_mask: FloatTensor [N, L], where elements with `True` are allowed and `False` are masked-out
        """
        bsz, num_classes = scores.shape[0], scores.shape[-1]

        n_tokens = target.numel()
        if self.ignore_index is not None:
            sample_size = n_nonpad_tokens = target.ne(self.ignore_index).float().sum()
        else:
            sample_size = n_nonpad_tokens = n_tokens

        # [N, L]
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs=F.log_softmax(scores, dim=-1),
            target=target,
            epsilon=self.label_smoothing,
            ignore_index=self.ignore_index,
            reduce=False,
        )
        if weights is not None:
            loss, nll_loss = loss * weights, nll_loss * weights
        fullseq_loss = loss.sum() / sample_size
        fullseq_nll_loss = nll_loss.sum() / sample_size

        t1_loss, t2_loss = None, None
        if watch_t1_t2_loss:
            t1_loss, t2_loss = loss.chunk(2)
            t1_mask, t2_mask = label_mask.chunk(2)
            t1_loss = (t1_loss * t1_mask).sum() / (t1_mask.sum())
            t2_loss = (t2_loss * t2_mask).sum() / (t2_mask.sum())
            
        # use coord masked loss for model training,
        # ignoring those position with missing coords (as nan)
        if label_mask is not None:
            label_mask = label_mask.float()
            sample_size = label_mask.sum()  # sample size should be set to valid coordinates
            loss = (loss * label_mask).sum() / sample_size
            nll_loss = (nll_loss * label_mask).sum() / sample_size
        else:
            loss, nll_loss = fullseq_loss, fullseq_nll_loss

        ppl = torch.exp(nll_loss)
        
        logging_output = {
            'nll_loss': nll_loss.data,
            'ppl': ppl.data,
            'fullseq_loss': fullseq_loss.data,
            'fullseq_nll_loss': fullseq_nll_loss.data,
            'bsz': bsz,
            'sample_size': sample_size,
            'sample_ratio': sample_size / n_tokens,
            'nonpad_ratio': n_nonpad_tokens / n_tokens,
            'weight_diff_loss': loss.data
        }
        
        if cal_constant_loss:
            constant_weights = weights.new_ones(size=weights.size())
            constant_loss, _ = label_smoothed_nll_loss(
                lprobs=F.log_softmax(scores, dim=-1),
                target=target,
                epsilon=self.label_smoothing,
                ignore_index=self.ignore_index,
                reduce=False,
            )
            constant_loss = constant_loss * constant_weights
            constant_loss = (constant_loss * label_mask).sum() / sample_size
            logging_output['constant_diff_loss'] = constant_loss.data

        if watch_t1_t2_loss:
            logging_output['weight_diff_t1_loss'] = t1_loss.data
            logging_output['weight_diff_t2_loss'] = t2_loss.data
        
        return loss, logging_output

def get_attn_loss(attn_map_loss):
    # bce_loss_func = nn.BCELoss()
    bce_loss_func = nn.MSELoss()
    struct_attn, aa_attn = attn_map_loss['attn_map'].chunk(2, dim=2)
    struct_attn_segment, aa_attn_segment = attn_map_loss['go_type_segments'].chunk(2, dim=2)
    struct_attn_segment_mask, aa_attn_segment_mask = attn_map_loss['go_type_segments_mask'].chunk(2, dim=2)

    print(f"struct_attn: {struct_attn},\nstruct_attn_segment: {struct_attn_segment},\n")

    with torch.amp.autocast(device_type='cuda', enabled=False):
        struct_attn_loss = bce_loss_func(struct_attn.float(), struct_attn_segment.float())
        aa_attn_loss = bce_loss_func(aa_attn.float(), aa_attn_segment.float())

#     dot = make_dot(struct_attn_loss)

# # 保存为文件（如 .png 或 .pdf），然后你可以打开查看
#     dot.render("loss_graph", view=True)
#     exit()

    # print(f"segment:{struct_attn_segment},\nattn: {struct_attn}")


    # print(f"struct_attn_loss: {struct_attn_loss.shape}")
    # print(f"aa_attn_loss: {aa_attn_loss.shape}")

    struct_attn_sum = (struct_attn_loss * struct_attn_segment_mask).sum(dim=(1, 2))  # Shape: (B)
    struct_mask_sum = struct_attn_segment_mask.sum(dim=(1, 2))                    # Shape: (B)

    # print(f"struct_attn_sum: {struct_attn_sum.shape}")
    # print(f"struct_mask_sum: {struct_mask_sum.shape}")

    struct_attn_loss_per_sample = struct_attn_sum / (struct_attn_segment_mask.shape[1] + 1e-5)

    print(f"struct_attn_sum: {struct_attn_sum}, struct_mask_sum: {struct_mask_sum}, struct_attn_loss_per_sample: {struct_attn_loss_per_sample}")

    # 2. 针对 aa_attn_loss 进行逐样本（Per-Sample）归一化
    aa_attn_sum = (aa_attn_loss * aa_attn_segment_mask).sum(dim=(1, 2))      # Shape: (B)
    aa_mask_sum = aa_attn_segment_mask.sum(dim=(1, 2))                        # Shape: (B)
    # 结果 aa_attn_loss_per_sample 的 shape: (B)

    aa_attn_loss_per_sample = aa_attn_sum / (aa_attn_segment_mask.shape[1] + 1e-5)


    print(f"aa_attn_sum: {aa_attn_sum}, aa_mask_sum: {aa_mask_sum}, aa_attn_loss_per_sample: {aa_attn_loss_per_sample}")

    # 3. 计算最终的 attn_loss
    # 3.1 逐样本加权: (B) * (B) -> (B)
    # struct_weighted_loss = struct_attn_loss_per_sample * attn_map_loss['struct_weight_point']
    # aa_weighted_loss = aa_attn_loss_per_sample * attn_map_loss['aatype_weight_point']

    # 不加权？？
    struct_weighted_loss = struct_attn_loss_per_sample
    aa_weighted_loss = aa_attn_loss_per_sample



    # 3.2 在 Batch 维度上求均值（最终的 Loss 标量）
    attn_loss = (struct_weighted_loss + aa_weighted_loss).mean()

    print(f"attn_loss: {attn_loss}")
    # exit()

    return attn_loss

class StructAARDMCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(
        self,
        scores_dict,
        target_dict,
        label_mask_dict=None,
        weights_dict=None,
        cal_constant_loss=False,
        watch_t1_t2_loss=False,
        hidden_states=None,
        attn_map_loss=None,
    ) -> Tensor:
        """
        scores: [N, L, C], unnormalized scores
        target: [N, L]
        coord_mask: FloatTensor [N, L], where elements with `True` are allowed and `False` are masked-out
        """
        losses = 0
        nll_losses = 0
        logging_output_dict = {}

        def compute(scores, target, label_mask, weights, key=""):
            if len(key) > 0:
                key = f"{key}/"
            bsz, num_classes = scores.shape[0], scores.shape[-1]
            n_tokens = target.numel()
            if self.ignore_index is not None:
                sample_size = n_nonpad_tokens = (
                    target.ne(self.ignore_index).float().sum()
                )
            else:
                sample_size = n_nonpad_tokens = n_tokens
            # [N, L]
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs=F.log_softmax(scores, dim=-1),
                target=target,
                epsilon=self.label_smoothing,
                ignore_index=self.ignore_index,
                reduce=False,
            )
            if weights is not None:
                loss, nll_loss = loss * weights, nll_loss * weights
            fullseq_loss = loss.sum() / sample_size
            fullseq_nll_loss = nll_loss.sum() / sample_size

            t1_loss, t2_loss = None, None
            if watch_t1_t2_loss:
                t1_loss, t2_loss = loss.chunk(2)
                t1_mask, t2_mask = label_mask.chunk(2)
                t1_loss = (t1_loss * t1_mask).sum() / (t1_mask.sum())
                t2_loss = (t2_loss * t2_mask).sum() / (t2_mask.sum())

            # use coord masked loss for model training,
            # ignoring those position with missing coords (as nan)

            if label_mask is not None:
                label_mask = label_mask.float()
                sample_size = max(1, label_mask.sum())
                if len(label_mask.shape) == (len(loss.shape) - 1):
                    # if bit-based modeling,
                    # the loss is in B x L x 13 and label_mask is in B x L
                    label_mask = label_mask[..., None].expand(loss.shape)
                loss = (loss * label_mask).sum() / sample_size
                nll_loss = (nll_loss * label_mask).sum() / sample_size
            else:
                loss, nll_loss = fullseq_loss, fullseq_nll_loss

            ppl = torch.exp(nll_loss)

            logging_output = {
                f"{key}nll_loss": nll_loss.data,
                f"{key}ppl": ppl.data,
                f"{key}fullseq_loss": fullseq_loss.data,
                f"{key}fullseq_nll_loss": fullseq_nll_loss.data,
                f"{key}bsz": bsz,
                f"{key}sample_size": sample_size,
                f"{key}sample_ratio": sample_size / n_tokens,
                f"{key}nonpad_ratio": n_nonpad_tokens / n_tokens,
                f"{key}weight_diff_loss": loss.data,
            }

            if cal_constant_loss:
                constant_weights = weights.new_ones(size=weights.size())
                constant_loss, _ = label_smoothed_nll_loss(
                    lprobs=F.log_softmax(scores, dim=-1),
                    target=target,
                    epsilon=self.label_smoothing,
                    ignore_index=self.ignore_index,
                    reduce=False,
                )
                constant_loss = constant_loss * constant_weights
                constant_loss = (
                    constant_loss * label_mask
                ).sum() / sample_size
                logging_output[f"{key}constant_diff_loss"] = constant_loss.data

            if watch_t1_t2_loss:
                logging_output[f"{key}weight_diff_t1_loss"] = t1_loss.data
                logging_output[f"{key}weight_diff_t2_loss"] = t2_loss.data

            return loss, nll_loss, logging_output

        if type(scores_dict) is not dict:
            loss, nll_loss, logging_output = compute(
                scores_dict, target_dict, label_mask_dict, weights_dict
            )
            return loss, logging_output
        else:
            for k, scores in scores_dict.items():
                loss, nll_loss, logging_output = compute(
                    scores,
                    target_dict[k],
                    label_mask_dict[k],
                    weights_dict[k],
                    k,
                )
                losses += loss
                nll_losses += nll_loss
                logging_output_dict.update(logging_output)
            logging_output_dict["sample_size"] = logging_output[
                f"{k}/sample_size"
            ]
            logging_output_dict["nll_loss"] = nll_losses / len(
                scores_dict.keys()
            )
            logging_output_dict["fullseq_loss"] = logging_output[
                f"{k}/fullseq_loss"
            ]
            logging_output_dict["fullseq_nll_loss"] = logging_output[
                f"{k}/fullseq_nll_loss"
            ]
            logging_output_dict["ppl"] = logging_output[f"{k}/ppl"]
            return losses / len(scores_dict.keys()), logging_output_dict


class ContrastMotifStructAARDMCrossEntropyLoss(nn.CrossEntropyLoss):


    def __init__(self, label_smoothing=0.1, ignore_index=-100, hidden_dim=1280, memory_size=1024, temperature=0.02, scale=0.1, start_step=10000, attn_scale=0, attn_start_step=10000):
        super().__init__(label_smoothing=label_smoothing, ignore_index=ignore_index)
        self.temperature = temperature
        self.memory_size = memory_size
        self.scale = scale
        self.attn_scale = attn_scale

        # 全局 memory bank
        # self.register_buffer("memory_bank", torch.empty(0, hidden_dim))
        # self.register_buffer("memory_labels", torch.empty(0, dtype=torch.long))
        # self.register_buffer("latest_label_bank", torch.zeros(400, hidden_dim))

        self.memory_bank = torch.empty(0, hidden_dim)
        self.memory_labels = torch.empty(0, dtype=torch.long)

        # self.memory_bank.requires_grad = False
        # self.memory_labels.requires_grad = False
        # self.latest_label_bank.requires_grad = False


        # 额外的 per-label 最新样本缓存
        self.latest_label_bank = {}

        self.time = 0

        self.start_time = start_step
        self.attn_start_time = attn_start_step

        self._load_class_weights()

    def _load_class_weights(self):
        try:
            import pickle
            # 1. 从 .pkl 文件加载权重张量
            with open('/AIRvePFS/dair/chenxr-data/repo/cfpgen/go_number_class_weights.pkl', 'rb') as f:
                class_weights_tensor = pickle.load(f)
            
            # 2. 将张量移动到当前设备 (GPU/CPU)
            # 必须使用 .to(self.device) 确保设备一致性
            self.class_weights = class_weights_tensor
            print(f"✅ 成功加载类别权重。权重形状: {self.class_weights.shape}")
            
        except FileNotFoundError:
            print("⚠️ 警告：未找到 go_number_class_weights.pkl 文件，将使用无加权 Cross Entropy Loss。")
            self.class_weights = None # 如果文件不存在，则不使用权重

    @torch.no_grad()
    def _update_memory_bank(self, features, labels):
        """更新全局 memory bank（FIFO）"""
        # 检查是否有nan的features
        if self.memory_bank.numel() == 0:
            self.memory_bank = features.detach()
            self.memory_labels = labels.detach()
        else:
            new_features = torch.cat([self.memory_bank, features.detach()], dim=0)
            new_labels = torch.cat([self.memory_labels, labels.detach()], dim=0)
            if len(new_features) > self.memory_size:
                new_features = new_features[-self.memory_size:]
                new_labels = new_labels[-self.memory_size:]
            self.memory_bank = new_features
            self.memory_labels = new_labels

        if torch.isnan(self.memory_bank).any():
            print(f"memory_bank contains nan: {self.memory_bank}")
            raise ValueError("memory_bank contains nan")

    @torch.no_grad()
    def _update_latest_label_bank(self, features, labels):
        """更新每个 label 的最新样本"""
        for f, l in zip(features, labels):
            # print(f"update label {l} with feature {f}")
            self.latest_label_bank[l.item()] = f.detach()
            # self.latest_label_bank[l.item()].requires_grad = False
            if torch.isnan(self.latest_label_bank[l.item()]).any():
                raise ValueError(f"latest_label_bank[{l.item()}] contains nan")

    def get_motif_hidden_states_and_labels(self, hidden_states, motif_position_and_label_list):
        """
        根据motif位置信息提取对应的hidden states和labels
        
        Args:
            hidden_states: list of [seq_len, hidden_dim], batch中的每个序列的hidden states
            motif_position_and_label_list: list of dict, 每个dict包含motif的位置和标签信息
        
        Returns:
            motif_hidden_states: [total_motifs, hidden_dim]
            motif_labels: [total_motifs]
        """
        all_motif_states = []
        all_motif_labels = []
        
        for i, (hidden_state, motif_dict) in enumerate(zip(hidden_states, motif_position_and_label_list)):
            # hidden_state: [seq_len, hidden_dim]
            for (start, end), label in motif_dict.items():

                if end - start <= 5:
                    print(f"bad start:{start} end:{end} label:{label}")
                    continue
                # 提取motif区域的hidden states
                motif_region = hidden_state[start:end]  # [motif_len, hidden_dim]

                if torch.isnan(motif_region).any():
                    print(f"motif_region contains nan: {motif_region}")
                    print(f"start: {start}, end: {end}, label: {label}")
                    print(f"hidden_state: {hidden_state}")
                    print(f"shape hidden_state: {hidden_state.shape}")
                    raise ValueError("motif_region contains nan")
                
                # 对motif区域进行池化（平均池化）
                motif_embedding = torch.mean(motif_region, dim=0)  # [hidden_dim]

                if torch.isnan(motif_embedding).any():
                    print(f"motif_embedding contains nan: {motif_embedding}")
                    print(f"start: {start}, end: {end}, label: {label}")
                    print(f"motif_region: {motif_region}")
                    raise ValueError("motif_embedding contains nan")
                
                all_motif_states.append(motif_embedding)
                all_motif_labels.append(label)
        
        if len(all_motif_states) == 0:
            # 如果没有motif，返回空tensor
            return torch.tensor([], device=hidden_states[0].device), torch.tensor([], device=hidden_states[0].device, dtype=torch.long)
        
        motif_hidden_states = torch.stack(all_motif_states)  # [total_motifs, hidden_dim]
        motif_labels = torch.tensor(all_motif_labels, device=motif_hidden_states.device, dtype=torch.long)

        if torch.isnan(motif_hidden_states).any():
            print(f"motif_hidden_states contains nan: {motif_hidden_states}")
            raise ValueError("motif_hidden_states contains nan")
        
        return motif_hidden_states, motif_labels


    def forward(
        self,
        scores_dict,
        target_dict,
        label_mask_dict=None,
        weights_dict=None,
        cal_constant_loss=False,
        watch_t1_t2_loss=False,
        hidden_states=None,
        attn_map_loss=None,
    ) -> Tensor:
        """
        scores: [N, L, C], unnormalized scores
        target: [N, L]
        coord_mask: FloatTensor [N, L], where elements with `True` are allowed and `False` are masked-out
        """

        device = scores_dict['struct'][0].device
        self.time += 1

        # attn_map_loss
        if attn_map_loss is not None and self.time >= self.attn_start_time:
            print("start attn loss")
            attn_loss = get_attn_loss(attn_map_loss) * self.attn_scale
            print(f"attn_loss: {attn_loss}")
        else:
            attn_loss = None

        # contrast loss

        motif_head_loss = None

        if hidden_states['motif_labels_logits'] is None:

            if hidden_states is None:
                raise ValueError("hidden_states should not be None")

            features, labels = self.get_motif_hidden_states_and_labels(hidden_states['struct'], hidden_states['motif'])

            B = features.shape[0]
            contrast_loss_mean = None
            
            if len(labels) != 0:
                if self.time < self.start_time:
                    # print("start update memory bank")
                    self._update_latest_label_bank(features, labels)     
                    self._update_memory_bank(features, labels)
                        
                else:
                    print("start contrast loss")

                    # --- 新增的权重扩展逻辑开始 ---
                    # struct_weight_point: [B]

                    # 1. 统计每个序列的motif数量
                    num_motifs_per_seq = []
                    for motif_dict in hidden_states['motif']:
                        # motif_dict.items() 返回的是 (start, end) : label
                        length = 0
                        for start, end in motif_dict.keys():
                            if end - start > 5:
                                length += 1
                        num_motifs_per_seq.append(length)

                    # 2. 将权重根据每个序列的motif数量进行扩展
                    expanded_weights = []
                    device = features.device # 确保权重张量和features在同一设备上
                    for i, weight in enumerate(hidden_states['struct_weight_point']):
                        # 权重 weight 复制 num_motifs_per_seq[i] 次
                        num_repeats = num_motifs_per_seq[i]
                        if num_repeats > 0:
                            # 使用 repeat 或 full 创建一个 [Mi] 形状的张量
                            expanded_weights.append(
                                torch.full((num_repeats,), weight.item(), dtype=weight.dtype, device=device)
                            )

                    # 3. 拼接得到最终的 motif 权重
                    if len(expanded_weights) > 0:
                        # final_motif_weights 形状为 [total_motifs]
                        final_motif_weights = torch.cat(expanded_weights)
                    else:
                        # 处理 features 为空的情况
                        final_motif_weights = torch.tensor([], device=device, dtype=hidden_states['struct_weight_point'].dtype)
                        
                    # 检查 features 的 batch size 是否和权重匹配（重要）
                    if len(features) != len(final_motif_weights):
                        # 如果 features 不为空，但长度不匹配，则说明逻辑有问题。
                        # 如果 features 为空，则两者长度都为0，是匹配的。
                        if len(features) > 0:
                            raise ValueError(f"Motif features size ({len(features)}) does not match expanded weights size ({len(final_motif_weights)}). Check motif extraction logic.")

                    struct_weight_point = final_motif_weights # 将 [B] 替换为 [total_motifs]
                    # --- 新增的权重扩展逻辑结束 ---

                    latest_feats = []
                    latest_labs = []
                    if self.latest_label_bank:
                        latest_feats.extend(self.latest_label_bank.values())
                        latest_labs.extend(self.latest_label_bank.keys())
                        # for label in labels:
                        #     if label in self.latest_label_bank:
                        #         latest_feats.append(self.latest_label_bank[label])
                        #         latest_labs.append(label)
                        if len(latest_feats) > 0:
                            latest_feats = torch.stack(latest_feats).to(device)
                            latest_labs = torch.tensor(latest_labs, dtype=torch.long, device=device)

                    # 拼接 batch + global memory + latest label memory
                    parts = [features]
                    parts_labels = [labels]

                    # if len(self.memory_bank) > 0:
                    #     parts.append(self.memory_bank)
                    #     parts_labels.append(self.memory_labels)
                    # if len(latest_feats) > 0:
                    #     parts.append(latest_feats)
                    #     parts_labels.append(latest_labs)

                    if len(self.memory_bank) > 0:
                        parts.append(self.memory_bank.detach().to(device))
                        parts_labels.append(self.memory_labels.detach().to(device))
                    if len(latest_feats) > 0:
                        parts.append(latest_feats.detach())
                        parts_labels.append(latest_labs.detach())

                    all_features = torch.cat(parts, dim=0)  # [B+M+L, D]
                    all_labels = torch.cat(parts_labels, dim=0)  # [B+M+L]
                    
                    # use cosine similarity
                    # normalize with eps
                    features = F.normalize(features, dim=-1, eps=1e-4)
                    all_features = F.normalize(all_features, dim=-1, eps=1e-4)

                    sim = torch.matmul(features, all_features.T) / self.temperature

                    # use euclidean distance
                    # dist = torch.cdist(features, all_features, p=2)  # [B, B+M+L]
                    # sim = -dist / self.temperature  # 负距离作为相似度

                    # print(f"sim: {sim}")

                    # exit()

                    # mask
                    label_eq = labels.unsqueeze(1) == all_labels.unsqueeze(0)
                    # self_mask = torch.eye(B, B+len(all_features)-B, device=device, dtype=torch.bool)

                    N = all_features.shape[0]
                    self_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
                    self_mask[torch.arange(B), torch.arange(B)] = True  # mask (i,i) for the first B entries

                    pos_mask = label_eq & ~self_mask   # positive examples excluding self
                    any_pos = pos_mask.any(dim=1)      # which rows have positives
                    den_mask = ~self_mask  # include all except self (positives + negatives)


                    NEG_INF = -1e4
                    sim_den = sim.clone()
                    sim_den[~den_mask] = NEG_INF  # exclude self


                    sim_pos = sim.clone()
                    sim_pos[~pos_mask] = NEG_INF  # non-positive become -inf so logsumexp ignores them

                    # row-wise logsumexp
                    den_logsumexp = torch.logsumexp(sim_den, dim=1)  # [B]
                    num_logsumexp = torch.logsumexp(sim_pos, dim=1)  # [B], if no positives row becomes -inf

                    # compute loss per row: only for those with any positive
                    loss_per_sample = torch.zeros(B, device=device)
                    valid = any_pos
                    # for rows with pos_mask all False, num_logsumexp == -inf, so skip them
                    loss_per_sample[valid] = -(num_logsumexp[valid] - den_logsumexp[valid])

                    weighted_loss_per_sample = loss_per_sample * struct_weight_point

                    # mean over valid rows (avoid dividing by zero)
                    if valid.any():
                        contrast_loss_mean = weighted_loss_per_sample[valid].mean() * self.scale
                    else:
                        contrast_loss_mean = torch.tensor(0.0, device=device, dtype=features.dtype)                

                    self._update_memory_bank(features, labels)
                    self._update_latest_label_bank(features, labels)

                    print(f"contrast_loss_mean: {contrast_loss_mean}")
                    if torch.isnan(contrast_loss_mean):
                        torch.set_printoptions(profile="full", sci_mode=False)
                        print(f"contrast_loss_mean is nan")
                        print(f"valid: {valid}")
                        print(f"loss_per_sample: {loss_per_sample}")
                        print(f"num_logsumexp: {num_logsumexp}")
                        print(f"den_logsumexp: {den_logsumexp}")
                        print(f"num_logsumexp[valid]: {num_logsumexp[valid]}")
                        print(f"den_logsumexp[valid]: {den_logsumexp[valid]}")
                        print(f"sim_pos: {sim_pos}")
                        print(f"sim_pos[valid]: {sim_pos[valid]}")
                        print(f"sim_den: {sim_den}")
                        print(f"sim_den[valid]: {sim_den[valid]}")
                        print(f"sim: {sim}")
                        print(f"sim[valid]: {sim[valid]}")
                        print(f"sim[valid][valid]: {sim[valid][valid]}")
                        print(f"sim[valid][valid].shape: {sim[valid][valid].shape}")
                        print(f"weighted_loss_per_sample: {weighted_loss_per_sample}")
                        print(f"struct_weight_point: {struct_weight_point}")
                        print(loss_per_sample[valid])

                        print(f"features: {features}")
                        print(f"features.shape: {features.shape}")
                        print(f"all_features: {all_features}")
                        print(f"all_features.shape: {all_features.shape}")
                        exit()
        else:
            contrast_loss_mean = None
            
            
            if self.time >= self.start_time:
                print("use motif labels logits")
                logits = hidden_states['motif_labels_logits']
                labels = hidden_states['motif_labels']
                
                device = logits.device

                if len(labels) != 0:
                    num_motif_labels = logits.size(-1)
                    invalid_labels = (labels < 0) | (labels >= num_motif_labels)
                    if invalid_labels.any():
                        bad_labels = labels[invalid_labels].detach().cpu().tolist()
                        raise ValueError(
                            "motif head labels must use the original GO id space "
                            f"[0, {num_motif_labels}); got invalid labels {bad_labels[:10]}"
                        )

                    loss_weights = None
                    if self.class_weights is not None:
                        loss_weights = self.class_weights.to(device=device, dtype=logits.dtype)
                        if loss_weights.numel() != num_motif_labels:
                            if loss_weights.numel() > num_motif_labels:
                                loss_weights = loss_weights[:num_motif_labels]
                            else:
                                raise ValueError(
                                    "motif head class weights must match the original GO label count; "
                                    f"got {loss_weights.numel()} weights for {num_motif_labels} labels"
                                )

                    motif_head_loss = F.cross_entropy(logits, labels, weight=loss_weights) * self.scale
                    contrast_loss_mean = motif_head_loss
                    # 要平均吗？
                    print(f"motif_head_loss: {motif_head_loss}")


        # dplm2 diffusion crossentropy loss start
        losses = 0
        nll_losses = 0
        logging_output_dict = {}

        def compute(scores, target, label_mask, weights, key=""):
            if len(key) > 0:
                key = f"{key}/"
            bsz, num_classes = scores.shape[0], scores.shape[-1]
            n_tokens = target.numel()
            if self.ignore_index is not None:
                sample_size = n_nonpad_tokens = (
                    target.ne(self.ignore_index).float().sum()
                )
            else:
                sample_size = n_nonpad_tokens = n_tokens
            # [N, L]
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs=F.log_softmax(scores, dim=-1),
                target=target,
                epsilon=self.label_smoothing,
                ignore_index=self.ignore_index,
                reduce=False,
            )
            if weights is not None:
                loss, nll_loss = loss * weights, nll_loss * weights
            fullseq_loss = loss.sum() / sample_size
            fullseq_nll_loss = nll_loss.sum() / sample_size

            t1_loss, t2_loss = None, None
            if watch_t1_t2_loss:
                t1_loss, t2_loss = loss.chunk(2)
                t1_mask, t2_mask = label_mask.chunk(2)
                t1_loss = (t1_loss * t1_mask).sum() / (t1_mask.sum())
                t2_loss = (t2_loss * t2_mask).sum() / (t2_mask.sum())

            # use coord masked loss for model training,
            # ignoring those position with missing coords (as nan)

            if label_mask is not None:
                label_mask = label_mask.float()
                sample_size = max(1, label_mask.sum())
                if len(label_mask.shape) == (len(loss.shape) - 1):
                    # if bit-based modeling,
                    # the loss is in B x L x 13 and label_mask is in B x L
                    label_mask = label_mask[..., None].expand(loss.shape)
                loss = (loss * label_mask).sum() / sample_size
                nll_loss = (nll_loss * label_mask).sum() / sample_size
            else:
                loss, nll_loss = fullseq_loss, fullseq_nll_loss

            ppl = torch.exp(nll_loss)

            logging_output = {
                f"{key}nll_loss": nll_loss.data,
                f"{key}ppl": ppl.data,
                f"{key}fullseq_loss": fullseq_loss.data,
                f"{key}fullseq_nll_loss": fullseq_nll_loss.data,
                f"{key}bsz": bsz,
                f"{key}sample_size": sample_size,
                f"{key}sample_ratio": sample_size / n_tokens,
                f"{key}nonpad_ratio": n_nonpad_tokens / n_tokens,
                f"{key}weight_diff_loss": loss.data,
            }

            if cal_constant_loss:
                constant_weights = weights.new_ones(size=weights.size())
                constant_loss, _ = label_smoothed_nll_loss(
                    lprobs=F.log_softmax(scores, dim=-1),
                    target=target,
                    epsilon=self.label_smoothing,
                    ignore_index=self.ignore_index,
                    reduce=False,
                )
                constant_loss = constant_loss * constant_weights
                constant_loss = (
                    constant_loss * label_mask
                ).sum() / sample_size
                logging_output[f"{key}constant_diff_loss"] = constant_loss.data

            if watch_t1_t2_loss:
                logging_output[f"{key}weight_diff_t1_loss"] = t1_loss.data
                logging_output[f"{key}weight_diff_t2_loss"] = t2_loss.data

            return loss, nll_loss, logging_output

        if type(scores_dict) is not dict:
            loss, nll_loss, logging_output = compute(
                scores_dict, target_dict, label_mask_dict, weights_dict
            )

            logging_output_dict["original_loss"] = losses

            if contrast_loss_mean is not None:
                logging_output[f"contrast_loss"] = contrast_loss_mean.data
                if motif_head_loss is not None:
                    logging_output[f"motif_head_loss"] = motif_head_loss.data
                loss += contrast_loss_mean

            if attn_loss is not None:
                logging_output_dict["attn_loss"] = attn_loss.data
                losses += attn_loss        

                
            return loss, logging_output
        else:
            for k, scores in scores_dict.items():
                loss, nll_loss, logging_output = compute(
                    scores,
                    target_dict[k],
                    label_mask_dict[k],
                    weights_dict[k],
                    k,
                )
                losses += loss
                nll_losses += nll_loss
                logging_output_dict.update(logging_output)
            logging_output_dict["sample_size"] = logging_output[
                f"{k}/sample_size"
            ]
            logging_output_dict["nll_loss"] = nll_losses / len(
                scores_dict.keys()
            )
            logging_output_dict["fullseq_loss"] = logging_output[
                f"{k}/fullseq_loss"
            ]
            logging_output_dict["fullseq_nll_loss"] = logging_output[
                f"{k}/fullseq_nll_loss"
            ]
            logging_output_dict["ppl"] = logging_output[f"{k}/ppl"]

            logging_output_dict["original_loss"] = losses / len(scores_dict.keys())

            if contrast_loss_mean is not None:
                logging_output_dict["contrast_loss"] = contrast_loss_mean.data
                if motif_head_loss is not None:
                    logging_output_dict["motif_head_loss"] = motif_head_loss.data
                losses += contrast_loss_mean
            if attn_loss is not None:

                # WANING

                print(f"delete losses just attn loss")
                losses = 0

                logging_output_dict["attn_loss"] = attn_loss.data
                losses += attn_loss
            
            return losses / len(scores_dict.keys()), logging_output_dict
