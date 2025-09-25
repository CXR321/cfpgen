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
with open("generation-results-cfpgen_650m/cfpgen_650m_go-ipr_interproscan_eval.pkl", "rb") as f:
    data = pickle.load(f)

print(data[0])