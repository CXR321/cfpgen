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

# a = []
# a.append(torch.tensor([[True,False,True],[True,False,True]]))
# a.append(torch.tensor([[True,False,True],[True,False,True]]))
# a.append(torch.tensor([[True,False,True],[True,False,True]]))
# a.append(torch.tensor([[True,False,True],[True,False,True]]))
# a = torch.tensor([[[True,False],[True,False]],[[True,False],[True,False]]])
# a = torch.zeros(10)
# a = a.repeat(2, 1)

# a = torch.tensor([True,False,True])
# b = torch.tensor([False,True,True])

# a = torch.tensor(1.0, requires_grad=True)
# b = torch.tensor(2.0)


# c = a * b

# c.retain_grad()

# d = a.detach()

# c.backward()

# print(c.grad)
# # print(a.squeeze(-1))
# # print(torch.stack(a, dim=1).shape)
# print(torch.max(a, b))

# import torch

# # 假设 L=400 （根据你之前提供的展平后的长度来估计）
# L = 4
# # 模拟你的原始张量：形状为 (1, L, 3)，且每行都是 [1., 0., 0.]
# attn_tensor = torch.tensor([[[1., 0., 0.] for _ in range(L)]])

# # 核心操作：调换第 1 维和第 2 维
# new_tensor = attn_tensor.permute(0, 2, 1)
# new_tensor = attn_tensor.reshape(1,3,4)
# # 或者使用 transpose 方法（在 PyTorch 中，transpose只接受两个维度索引）
# # new_tensor = att
# # n_tensor.transpose(1, 2)
# print(attn_tensor)
# print(new_tensor)

# import pickle
# from Bio import SeqIO
# from Bio.Seq import Seq
# from Bio.SeqRecord import SeqRecord

# def pkl_to_fasta(pkl_path, output_fasta):
#     print(f"正在将 {pkl_path} 转换为 FASTA...")
#     with open(pkl_path, 'rb') as f:
#         data = pickle.load(f)
    
#     records = []
#     for i, item in enumerate(data):
#         # 兼容不同的数据结构
#         seq_str = item.get('sequence', '') if isinstance(item, dict) else getattr(item, 'sequence', '')
#         # 如果有 ID 就用 ID，没有就用索引
#         uid = item.get('uniprot_id', f'train_{i}') if isinstance(item, dict) else getattr(item, 'uniprot_id', f'train_{i}')
        
#         if seq_str:
#             records.append(SeqRecord(Seq(seq_str), id=str(uid), description=""))
            
#     SeqIO.write(records, output_fasta, "fasta")
#     print(f"转换完成: {output_fasta} (共 {len(records)} 条序列)")

# if __name__ == "__main__":
#     # 你的训练集路径
#     TRAIN_PKL = 'data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
#     pkl_to_fasta(TRAIN_PKL, "train_database.fasta")


import pickle
from tqdm import tqdm

# ================= 配置 =================
TRAIN_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
GO_MAPPING_PATH = 'go_mapping.pkl'

# 你的查询条件集合
QUERY_SET = {'GO:0003954', 'GO:0008137', 'GO:0009055'}

def main():
    # 1. 加载映射文件
    print(f"Loading mapping from {GO_MAPPING_PATH}...")
    with open(GO_MAPPING_PATH, 'rb') as f:
        go_mapping = pickle.load(f)
    # 反转映射: Index -> GO ID string
    index_to_go = {v: k for k, v in go_mapping.items()}

    # 2. 加载训练集
    print(f"Loading training data from {TRAIN_PATH}...")
    with open(TRAIN_PATH, 'rb') as f:
        train_data = pickle.load(f)

    # 3. 搜索交集
    print(f"Searching for overlaps with {QUERY_SET}...")
    
    results = []
    
    for entry in tqdm(train_data, desc="Scanning"):
        # 将当前蛋白的 int 列表转换为 GO ID 集合
        current_go_set = {index_to_go[i] for i in entry['go_f_mapped']}
        
        # 计算交集
        intersection = current_go_set.intersection(QUERY_SET)
        
        # 如果有交集，就记录下来
        if len(intersection) > 0:
            results.append({
                'id': entry['uniprot_id'],
                'matched': intersection,
                'full_labels': current_go_set
            })

    # 4. 输出结果
    print("\n" + "="*60)
    print(f"SEARCH RESULTS")
    print("="*60)
    print(f"Query Conditions: {QUERY_SET}")
    print(f"Total Matches Found: {len(results)}")
    print("-" * 60)
    print(f"{'UniProt ID':<15} | {'Matched Query Terms'}")
    print("-" * 60)

    # 为了防止刷屏，如果数量太多，你可以限制打印数量，这里默认打印前 50 个
    for i, res in enumerate(results):
        matched_str = ', '.join(sorted(list(res['matched'])))
        print(f"{res['id']:<15} | {matched_str}")
        
        # 如果只想看前50个，取消下面两行的注释
        # if i >= 50:
        #     print(f"... and {len(results) - 50} more entries.")
        #     break

if __name__ == '__main__':
    main()