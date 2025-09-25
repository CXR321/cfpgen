from byprot.models.lm.dplm2 import (
    MultimodalDiffusionProteinLanguageModel as DPLM2,
)
import struct
from datasets import load_dataset
import pickle
import os
import requests
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from biotite.sequence.io import fasta
import torch
import random

# Login using e.g. `huggingface-cli login` to access this dataset

# with open("data-bin/uniprotKB/cfpgen_general_dataset/train_go_terms_emb.pkl", "rb") as f:
#     go_terms_emb = pickle.load(f)

# keys = list(go_terms_emb.keys())
# print(go_terms_emb[keys[0]][0].shape)
# exit()

# with open("data-bin/uniprotKB/cfpgen_general_dataset/train_expanded.pkl", "rb") as f:
#     train_data_expanded = pickle.load(f)

# with open("data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold.pkl", "rb") as f:
#     train_data_expanded = pickle.load(f)

# model = DPLM2.from_pretrained("airkingbd/dplm2_650m")
# model.eval()
# model.to("cuda")

def find_motif_in_aa_seq(aa_seq, motif_segment, seq, name):
    """在 aa_seq 中查找 motif_segment 的准确位置"""
    index = aa_seq.find(motif_segment)
    if index == -1:
        print(f"❌ Motif segment {motif_segment} not found in aa_seq!")
        print(f"aa_seq: {aa_seq}")
        print(f"seq: {seq}")
        print(f"name: {name}")

        # raise ValueError("❌ Motif segment not found in aa_seq!")
        return None, None
    return index, index + len(motif_segment) - 1  # 返回 start, end (0-based)

def find_pfam_in_aaseq(aa_seq, ori_seq, motif_s, motif_e):

    aa_start, aa_end = find_motif_in_aa_seq(ori_seq, aa_seq, "None", "None")

    if aa_start is None or aa_end is None:
        return None, None 

    if motif_s <= aa_start:
        aa_motif_s = aa_start
    elif motif_s > aa_end:
        return None, None
    else:
        aa_motif_s = motif_s

    if motif_e >= aa_end:
        aa_motif_e = aa_end
    elif motif_e < aa_start:
        return None, None
    else:
        aa_motif_e = motif_e


    start = aa_motif_s - aa_start
    end = aa_motif_e - aa_start

    return start, end

# go_terms_emb = {}


# not_find_seq_id = []

# n = 1
# for data in tqdm(train_data_expanded):
#     n+=1
#     ori_seq = data['sequence']
#     aa_seq = data['aa_seq']
#     if len(data['struct_seq']) == 0:
#         continue

#     struct_seq = data['struct_seq'].split(',')

#     aa_start, aa_end = find_motif_in_aa_seq(ori_seq, aa_seq, data['sequence'], data['uniprot_id'])

#     if aa_start is None:
#         print(f"❌ Cannot find {data['uniprot_id']} in aa_seq!")
#         not_find_seq_id.append(data['uniprot_id'])
#         continue

#     for motif_info in data['pfam_motif']:
#         motif_segment = motif_info['seq']
#         # if motif_info['go_term'] not in go_terms_emb:
#         #     go_terms_emb[motif_info['go_term']] = []

#         if motif_info['pfam_id'] not in go_terms_emb:
#             go_terms_emb[motif_info['pfam_id']] = []

#         # 重新计算 motif 在 aa_seq 中的位置
#         # start, end = find_motif_in_aa_seq(aa_seq, motif_segment, data['sequence'], data['uniprot_id'])

#         motif_s = motif_info['start'] - 1
#         motif_e = motif_info['end'] - 1

#         if motif_s <= aa_start:
#             aa_motif_s = aa_start
#         elif motif_s > aa_end:
#             continue
#         else:
#             aa_motif_s = motif_s

#         if motif_e >= aa_end:
#             aa_motif_e = aa_end
#         elif motif_e < aa_start:
#             continue
#         else:
#             aa_motif_e = motif_e

#         if aa_motif_e - aa_motif_s + 1 < 20:
#             print(f"❌ Motif segment too short: {aa_motif_e - aa_motif_s + 1} in {motif_info['pfam_id']} of {data['uniprot_id']}")
#             continue
            


#         # if start is None:
#         #     continue

#         start = aa_motif_s - aa_start
#         end = aa_motif_e - aa_start

#         cls_struct_id = model.tokenizer._token_to_id["<cls_struct>"]
#         eos_struct_id = model.tokenizer._token_to_id["<eos_struct>"]

#         # 提取 struct_seq 对应位置的 tokens
#         struct_tokens = struct_seq[start : end + 1]  # Python 切片是 [start, end+1)
#         struct_tokens = [cls_struct_id] + [model.tokenizer._token_to_id[token] for token in struct_tokens] + [eos_struct_id]

#         input_ids = torch.tensor(struct_tokens).unsqueeze(0).to(model.device)
#         input_mask = torch.ones_like(input_ids).to(model.device)
#         # with torch.no_grad():
#         #     emb = model.net.esm.embeddings(
#         #         input_ids, attention_mask=input_mask
#         #     )
#             # print(emb_mean.shape)
#         with torch.no_grad():
#             res = model(input_ids)
#             emb_mean = res["last_hidden_state"][0][0]
#             # print(emb_mean.shape)
#             emb_mean = emb_mean.cpu().numpy()

#             # emb = res["last_hidden_state"][0][1:-1]
#             # emb_mean = emb.mean(dim=0).cpu().numpy()

#         go_terms_emb[motif_info['pfam_id']].append((emb_mean, motif_info['evalue']))

# # with open("data-bin/uniprotKB/cfpgen_general_dataset/train_go_terms_mean_emb.pkl", "wb") as f:
# #     pickle.dump(go_terms_emb, f)

# with open("data-bin/uniprotKB/cfpgen_general_dataset/train_pfam_cls_emb_pfamMotif.pkl", "wb") as f:
#     pickle.dump(go_terms_emb, f)

# with open("data-bin/uniprotKB/cfpgen_general_dataset/not_find_seq_id.txt", "w") as f:
#     for seq_id in not_find_seq_id:
#         f.write(f"{seq_id}\n")



# random.seed(42)  # 设置随机种子以确保可重复性
# selected_indices = random.sample(range(len(train_data_expanded)), min(500, len(train_data_expanded)))
# selected_data = [train_data_expanded[i] for i in selected_indices]

# go_terms_emb['random'] = []
# for data in tqdm(selected_data):
#     aa_seq = data['aa_seq']
#     struct_seq = data['struct_seq'].split(',')

#     min_length = 30
#     max_length = 200

#     segment_length = random.randint(min_length, min(max_length, len(struct_seq)))
    
#     # 随机生成起始位置
#     max_start = len(struct_seq) - segment_length
#     start = random.randint(0, max_start)
#     end = start + segment_length - 1

#     cls_struct_id = model.tokenizer._token_to_id["<cls_struct>"]
#     eos_struct_id = model.tokenizer._token_to_id["<eos_struct>"]

#     # 提取 struct_seq 对应位置的 tokens
#     struct_tokens = struct_seq[start : end + 1]  # Python 切片是 [start, end+1)
#     struct_tokens = [cls_struct_id] + [model.tokenizer._token_to_id[token] for token in struct_tokens] + [eos_struct_id]

#     input_ids = torch.tensor(struct_tokens).unsqueeze(0).to(model.device)
#     input_mask = torch.ones_like(input_ids).to(model.device)
    
#     with torch.no_grad():
#         res = model(input_ids)
#         emb_mean = res["last_hidden_state"][0][0]
#         emb_mean = emb_mean.cpu().numpy()

#     go_terms_emb['random'].append(emb_mean)

# print(f"成功处理了 {n} 条数据")
# print(f"收集到 {len(go_terms_emb)} 个不同的GO terms")

# with open("data-bin/uniprotKB/cfpgen_general_dataset/train_go_terms_cls_emb_500_random.pkl", "wb") as f:
#     pickle.dump(go_terms_emb, f)

