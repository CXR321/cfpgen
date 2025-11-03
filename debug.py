import torch
import torch.nn.functional as F
print(torch.__version__)  # 输出 PyTorch 版本
print(torch.version.cuda)  # 输出 PyTorch 编译时使用的 CUDA 版本
print(torch.cuda.is_available())  # 检查 CUDA 是否可用（True/False）


# from peft import LoraModel 
# 原始模块
# import torch.nn as nn
# orig_linear = nn.Linear(10, 5)

# # 包装成 ModulesToSaveWrapper
# wrapped = ModulesToSaveWrapper(orig_linear)

# # 访问原始模块
# print(wrapped.base_module)  # 原来的 nn.Linear(10, 5)

# t = torch.tensor([[True, False], [False, True]])
# print(~t)

# scores = torch.tensor([[-float('inf'), -float('inf'), -float('inf')]])
# weights = F.softmax(scores, dim=-1)
# print("Softmax output:", weights)  # 看看是否产生NaN

# def parse_pfam2go(filename="pfam2go.txt"):
#     """
#     解析 pfam2go 文件，返回一个映射字典。
    
#     Args:
#         filename (str): pfam2go 文件的路径
        
#     Returns:
#         dict: 一个字典，格式为 {pfam_accession: [(go_id, go_term)]}
#     """
#     pfam2go_map = {}
    
#     with open(filename, 'r') as file:
#         for line in file:
#             line = line.strip()
#             # 跳过注释行和空行
#             if line.startswith('!') or not line:
#                 continue
                
#             # 解析一行
#             # 示例: Pfam:7tm_1 PF00001 > GO:G protein-coupled receptor activity ; GO:0004930
#             parts = line.split(' > ')
#             if len(parts) < 2:
#                 continue
                
#             # 提取 Pfam 部分
#             pfam_section = parts[0].split()
#             pfam_acc = pfam_section[1] # 获取 7tm_1
                
#             # 提取 GO 部分
#             go_section = parts[1].split(' ; ')
#             go_desc = go_section[0][3:]  # 移除前面的 "GO:"，获取描述
#             go_id = go_section[1]        # 获取 GO:0004930
            
#             # 将映射添加到字典中
#             if pfam_acc not in pfam2go_map:
#                 pfam2go_map[pfam_acc] = []
#             pfam2go_map[pfam_acc].append((go_id, go_desc))
            
#     return pfam2go_map

# dict = parse_pfam2go('pfam2go.txt')

# from pfam2go import pfam2go  
# pfam_list = ['PF12146']  
# # pfam_list = "Flu_M1"  
# data = pfam2go(pfam_list)  

# # print(data)

# pfam_list = ['Flu_M1', '7tm_1', 'Abhydrolase_6', 'FSH1']

# for p in pfam_list:
#     print(p, dict[p])


import pickle
# with open("generation-results-cfpgen_650m/cfpgen_650m_go-ipr_interproscan_eval.pkl", "rb") as f:
#     data = pickle.load(f)

# print(data[0])

path = "data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added_pfamMotif.pkl"
# path = "data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold.pkl"
path = "data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif.pkl"
path = "data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl"
# path = "data-bin/uniprotKB/cfpgen_general_dataset/valid_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl"
# path = "data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold.pkl"


# with open(path, "rb") as f:
#     data = pickle.load(f)

# print(len(data))
# n = 0
# for meta in data:
#     # if meta.get('pfam_motif', None) is None:
#     # print(meta)
#     if len(meta['struct_seq'].split(',')) != len(meta['aa_seq']) and len(meta['struct_seq'])!=0 and meta.get("esmfold_plddt") is None:
#         n+=1
#         # print(meta)
        
# print(n)

# a = []

# # a.extend([1,2,3])
# a = ' '.join(["123", "321"])
# # print(a)
# features = torch.tensor([[1,2,3], [4,5,6], [7,8,9]], dtype=torch.float)
# all_features = features
# a = torch.cdist(features, all_features, p=2)

# # a = [-1e6, -1e6, -1e6]
# # a = torch.tensor(a, dtype=torch.float16)
# c = a - a
# b = torch.logsumexp(a, dim=0)

# print(a)
# print(b)
# print(c)
# go_type_segments = [[(1, 2), None], [(2, 3)]]
# raw_lens = [4, 6]
# seq_len = 10
# batch = {'go_type_segments': []}

# max_len = 3

# batch['go_type_segments'] = []
# batch['go_type_segments_mask'] = []
# for single_p_go_type_segments, raw_len in zip(go_type_segments, raw_lens):
#     single_gt = []
#     single_gt_mask = []
#     for segement in single_p_go_type_segments:
#         if segement is not None:
#             gt = torch.zeros(2*seq_len, dtype=torch.float)
#             gt_mask = torch.zeros(2*seq_len, dtype=torch.bool)
#             start, end = segement
#             gt[start:end] = 1.0
#             gt[start+seq_len:end+seq_len] = 1.0
#             gt_mask[1:raw_len-1] = True
#             gt_mask[1+seq_len:raw_len-1+seq_len] = True
#             single_gt.append(gt)
#             single_gt_mask.append(gt_mask)
#         else:
#             single_gt.append(torch.zeros(2*seq_len, dtype=torch.float))
#             single_gt_mask.append(torch.zeros(2*seq_len, dtype=torch.bool))
#     for _ in range(max_len-len(single_p_go_type_segments)):
#         single_gt.append(torch.zeros(2*seq_len, dtype=torch.float))
#         single_gt_mask.append(torch.zeros(2*seq_len, dtype=torch.bool))
#     batch['go_type_segments'].append(torch.stack(single_gt))
#     batch['go_type_segments_mask'].append(torch.stack(single_gt_mask))
# batch['go_type_segments'] = torch.stack(batch['go_type_segments'])
# batch['go_type_segments_mask'] = torch.stack(batch['go_type_segments_mask'])

# print(batch['go_type_segments'])
# print(batch['go_type_segments_mask'])

a = torch.tensor([[True,False],[True,False]])
print(a.squeeze(-1))