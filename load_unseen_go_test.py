import pickle
import json
import torch
import numpy as np
from tqdm import tqdm
import os

# ================= 路径配置 =================
# 输入文件
CANDIDATES_PATH = '/AIRvePFS/dair/fsq-data/data/Protein/data_dfp/group_candidates.json'
CANDIDATES_PATH = '/AIRvePFS/dair/fsq-data/data/Protein/data_dfp/group_filtered.json'
GO_MAPPING_PATH = 'go_mapping.pkl'               # GO ID -> Index
CLS_EMB_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/train_go_terms_cls_emb.pkl'

# 映射字典路径 (用于反推 Description)
DESC_MAPPING_PATH = './go_id_mapping.pkl'        # 包含 desc2map_dict
STATIC_MAPPING_PATH = './desc2map_dict_statics.pkl' # 包含 desc2map_dict_statics

# 输出文件
OUTPUT_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/generated_candidates_motif_emb.pkl'


def load_resources():
    print("Loading resources...")
    
    # 1. 加载 Candidates
    with open(CANDIDATES_PATH, 'r') as f:
        candidates = json.load(f)

    # 2. 加载 GO ID -> Index 映射
    with open(GO_MAPPING_PATH, 'rb') as f:
        # 假设文件内容是 {GO_ID: Index}
        go_id_to_index = pickle.load(f)

    # 3. 加载 Description -> Index 映射 (用于反向构建 Index -> Description)
    with open(DESC_MAPPING_PATH, 'rb') as f:
        desc2map_dict = pickle.load(f)
    
    with open(STATIC_MAPPING_PATH, 'rb') as f:
        desc2map_dict_statics = pickle.load(f)

    # 4. 加载 Embeddings
    with open(CLS_EMB_PATH, 'rb') as f:
        cls_emb = pickle.load(f)

    return candidates, go_id_to_index, desc2map_dict, desc2map_dict_statics, cls_emb


def build_index_to_desc_map(desc2map_dict, desc2map_dict_statics):
    """
    构建 Index -> Description 的反向映射表。
    优先使用 static 字典，然后使用 dynamic 字典。
    """
    index_to_desc = {}

     # 2. 处理 go_id_mapping 中的 motif_desc_to_id
    if 'motif_desc_to_id' in desc2map_dict:
        for desc, idx in desc2map_dict['motif_desc_to_id'].items():
            # 如果存在冲突，这里会覆盖 (通常 ID 应该是唯一的)
            index_to_desc[idx] = desc   

    # 1. 处理 desc2map_dict_statics
    for desc, idx in desc2map_dict_statics.items():
        index_to_desc[idx] = desc
        

            
    return index_to_desc


def get_embedding_by_desc(desc, cls_emb):
    """
    根据描述获取 embedding，包含你提供的特殊字符串替换逻辑。
    虽然我们是反向查找，但为了匹配 cls_emb 的 key，这里做一次标准化检查。
    """
    # 特殊情况处理（虽然字典里取出来的应该是"清洗后"的 Key，但为了保险照搬你的逻辑）
    if desc == "acetyl-CoA:L-glutamate N-acetyltransferase activity":
        desc = "L-glutamate N-acetyltransferase activity"
    elif desc == "glutamate N-acetyltransferase activity":
        desc = "L-glutamate N-acetyltransferase activity, acting on acetyl-L-ornithine as donor"
    elif desc == "methione N-acyltransferase activity":
        desc = "L-methionine N-acyltransferase activity"
    elif desc == "oxidoreductase activity, acting on NAD(P)H, NAD(P) as acceptor":
        desc = "oxidoreductase activity, acting on NAD(P)H as acceptor"
        
    # 尝试获取 embedding
    if desc in cls_emb:
        return cls_emb[desc]
    else:
        return None


def process_data():
    candidates, go_id_to_index, desc2map_dict, desc2map_dict_statics, cls_emb = load_resources()
    
    # 构建反向映射表: Index -> Description
    index_to_desc = build_index_to_desc_map(desc2map_dict, desc2map_dict_statics)
    
    new_dataset = []
    print(f"Processing {len(candidates)} groups...")

    # for i, item in enumerate(tqdm(candidates)):
    #     go_list = item['group']
        
    #     # 初始化数据结构
    #     entry = {
    #         'uniprot_id': f'GEN_{i:06d}',  # 伪造 ID
    #         'go_numbers': {
    #             'F': go_list,
    #             'C': [],
    #             'P': []
    #         },
    #         'aa_seq': None,
    #         'struct_seq': None,
    #         'motif_mask': None,
    #         'motif_struct_emb': torch.zeros(7, 1280, dtype=torch.float32),
    #         'go_f_mapped': []
    #     }
        
    #     mapped_indices = []
    #     motif_num = 0
        
    #     for go_id in go_list:
    #         # 1. GO ID -> Index
    #         if go_id not in go_id_to_index:
    #             # print(f"Skipping {go_id}: Not in go_mapping")
    #             continue
                
    #         idx = go_id_to_index[go_id]
    #         mapped_indices.append(idx)
            
    #         # 2. Index -> Description -> Embedding
    #         # 只填充前 7 个 motif embedding
    #         if motif_num < 7:
    #             if idx in index_to_desc:
    #                 desc = index_to_desc[idx]
    #                 raw_emb = get_embedding_by_desc(desc, cls_emb)
                    
    #                 if raw_emb is not None:
    #                     try:
    #                         numpy_array = np.array(raw_emb)
    #                         # print(numpy_array.shape)
    #                         # exit()
    #                         # 转换为 tensor 并求均值 (假设 raw_emb 是 list of vectors 或类似结构)
    #                         entry['motif_struct_emb'][motif_num] = torch.mean(torch.from_numpy(numpy_array), dim=0)
    #                         motif_num += 1
    #                     except Exception as e:
    #                         print(f"Error processing tensor for {desc}: {e}")
    #                 else:
    #                     print(f"Warning: No embedding found for desc: {desc} (GO: {go_id})")
    #                     pass
    #             else:
    #                 print(f"Warning: No description found for index {idx} (GO: {go_id})")
    #                 pass

    #     entry['go_f_mapped'] = mapped_indices
    #     new_dataset.append(entry)
    for i, item in enumerate(tqdm(candidates)):
        go_list = item['group']
        
        # --- 步骤 1: 预计算公共数据 (Embeddings 和 Mapping) ---
        # 因为生成的10个样本享有相同的 GO 组合，没必要重复计算 10 次
        
        temp_motif_struct_emb = torch.zeros(7, 1280, dtype=torch.float32)
        temp_go_f_mapped = []
        
        motif_num = 0
        
        for go_id in go_list:
            # 1. GO ID -> Index
            if go_id not in go_id_to_index:
                continue
                
            idx = go_id_to_index[go_id]
            temp_go_f_mapped.append(idx)
            
            # 2. Index -> Description -> Embedding
            if motif_num < 7:
                if idx in index_to_desc:
                    desc = index_to_desc[idx]
                    raw_emb = get_embedding_by_desc(desc, cls_emb)
                    
                    if raw_emb is not None:
                        try:
                            numpy_array = np.array(raw_emb)
                            temp_motif_struct_emb[motif_num] = torch.mean(torch.from_numpy(numpy_array), dim=0)
                            motif_num += 1
                        except Exception as e:
                            print(f"Error processing tensor for {desc}: {e}")
                    else:
                        print(f"Warning: No embedding found for desc: {desc} (GO: {go_id})")
                else:
                    print(f"Warning: No description found for index {idx} (GO: {go_id})")

        # --- 步骤 2: 生成 10 个副本 ---
        for j in range(10):
            entry = {
                # 生成唯一 ID，例如: GEN_000001_0, GEN_000001_1 ...
                'uniprot_id': f'GEN_{i:06d}_{j}', 
                'go_numbers': {
                    'F': go_list,
                    'C': [],
                    'P': []
                },
                'aa_seq': "",
                'struct_seq': "",
                'motif_mask': None,
                # 使用 .clone() 确保内存独立，虽然在这里如果不修改也没事，但为了安全起见
                'motif_struct_emb': temp_motif_struct_emb.clone(), 
                'go_f_mapped': temp_go_f_mapped[:], # 列表浅拷贝
                'sequence': "",
            }
            new_dataset.append(entry)

    # 保存
    print(f"Saving to {OUTPUT_PATH}...")
    print(new_dataset[0])
    print(len(new_dataset))
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(new_dataset, f)
    print("Done.")

if __name__ == "__main__":
    process_data()