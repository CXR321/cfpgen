# import pickle
# from collections import Counter
# from tqdm import tqdm

# # ================= 配置路径 =================
# train_path = 'data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
# go_mapping_path = 'go_mapping.pkl'

# # ================= 1. 加载数据 =================
# print(f"Loading GO mapping from {go_mapping_path}...")
# with open(go_mapping_path, 'rb') as f:
#     go_mapping = pickle.load(f)
# # 反转映射: index -> GO:ID
# index_to_go = {v: k for k, v in go_mapping.items()}

# print(f"Loading training data from {train_path}...")
# with open(train_path, 'rb') as f:
#     train_data = pickle.load(f)

# total_samples = len(train_data)
# print(f"Total Training Samples: {total_samples}")

# # ================= 2. 统计频率 =================
# single_go_counter = Counter()
# combination_counter = Counter()

# print("Counting frequencies...")
# for entry in tqdm(train_data):
#     # 获取索引列表
#     indices = entry['go_f_mapped']
    
#     # 1. 统计单个 GO 标签 (把列表里的每个 ID 都单独计数)
#     single_go_counter.update(indices)
    
#     # 2. 统计组合 (排序并转为 tuple，保证唯一性)
#     # 将 index 转回 GO ID 以便阅读，或者保持 index 后续再转
#     # 这里直接存 GO ID 的 tuple
#     go_ids = [index_to_go[i] for i in indices]
#     combo_tuple = tuple(sorted(go_ids))
#     combination_counter[combo_tuple] += 1

# # ================= 3. 输出 Top 100 单个 GO 标签 =================
# print("\n" + "="*80)
# print(f"{'TOP 100 INDIVIDUAL GO TERMS':^80}")
# print("="*80)
# print(f"{'Rank':<5} | {'GO ID':<15} | {'Count':<10} | {'Frequency':<10}")
# print("-" * 80)

# for rank, (idx, count) in enumerate(single_go_counter.most_common(100), 1):
#     go_id = index_to_go[idx]
#     freq = (count / total_samples) * 100
#     print(f"{rank:<5} | {go_id:<15} | {count:<10} | {freq:.2f}%")

# # ================= 4. 输出 Top 100 GO 组合 =================
# print("\n" + "="*80)
# print(f"{'TOP 100 GO COMBINATIONS':^80}")
# print("="*80)
# print(f"{'Rank':<5} | {'Count':<8} | {'Freq':<7} | {'GO Combination'}")
# print("-" * 80)

# for rank, (combo, count) in enumerate(combination_counter.most_common(100), 1):
#     freq = (count / total_samples) * 100
#     combo_str = ", ".join(combo)
    
#     # 如果组合太长，截断显示以便排版
#     if len(combo_str) > 50:
#         combo_str = combo_str[:47] + "..."
        
#     print(f"{rank:<5} | {count:<8} | {freq:.2f}%  | {combo_str}")

# # ================= 5. (可选) 保存到文件 =================
# # 如果你想把这些结果保存成文本文件，取消下面的注释
# # with open('train_statistics_top100.txt', 'w') as f:
# #     f.write("Top 100 Single GO Terms\n")
# #     for idx, count in single_go_counter.most_common(100):
# #         f.write(f"{index_to_go[idx]}\t{count}\n")
# #     f.write("\nTop 100 Combinations\n")
# #     for combo, count in combination_counter.most_common(100):
# #         f.write(f"{', '.join(combo)}\t{count}\n")
# # print("\nStatistics saved to train_statistics_top100.txt")

# import pickle
# import os

# def find_targets_in_pkl():
#     # ================= 配置路径 =================
#     # 请确保这个路径是正确的
#     test_path = 'data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
#     go_mapping_path = 'go_mapping.pkl' # 如果pkl里存的是数字索引，需要映射表
#     # ===========================================

#     print(f"正在加载数据集: {test_path} ...")
#     if not os.path.exists(test_path):
#         print(f"错误: 找不到文件 {test_path}")
#         return

#     try:
#         with open(test_path, 'rb') as f:
#             data = pickle.load(f)
#         print(f"加载成功，数据集包含 {len(data)} 个样本")
#     except Exception as e:
#         print(f"加载 pickle 失败: {e}")
#         return

#     # 尝试加载映射表 (如果 pkl 里存的是数字索引)
#     index_to_go = None
#     if os.path.exists(go_mapping_path):
#         try:
#             with open(go_mapping_path, 'rb') as f:
#                 go_mapping = pickle.load(f)
#             # 翻转字典: value -> key (假设 mapping 是 GO -> Index)
#             index_to_go = {v: k for k, v in go_mapping.items()}
#             print(f"加载映射表成功，共 {len(index_to_go)} 个标签")
#         except:
#             print("警告: 加载 go_mapping.pkl 失败，假设数据中直接包含 GO ID 字符串")
    
#     # 定义我们要找的 5 个目标组合
#     target_combos = {
#         "1. Rank 8 (Metalloendopeptidase)": {
#             "ids": {'GO:0004477', 'GO:0004488'},
#             "desc": "Protein Degradation (Hydrolysis)"
#         },
#         "2. Rank 11 (Aspartate-tRNA ligase)": {
#             "ids": {'GO:0004070', 'GO:0016597'},
#             "desc": "Protein Translation (Ligation)"
#         },
#         "3. Rank 14 (NADH dehydrogenase)": {
#             "ids": {'GO:0008137', 'GO:0048038', 'GO:0050136'},
#             "desc": "Energy Metabolism (Redox)"
#         },
#         "4. Rank 19 (RNA binding + RNase)": {
#             "ids": {'GO:0003723', 'GO:0004523', 'GO:0030145'},
#             "desc": "RNA Processing (Cleavage)"
#         },
#         "5. Rank 35 (Iron-sulfur cluster)": {
#             "ids": {'GO:0004076', 'GO:0005506', 'GO:0051537', 'GO:0051539'},
#             "desc": "Inorganic Cofactor (Coordination)"
#         }
#     }

#     found_samples = {}
    
#     print("\n开始搜索目标组合...")
    
#     for idx, item in enumerate(data):
#         # 1. 解析样本 ID
#         # 根据您提供的参考代码，key 是 'uniprot_id'
#         sample_id = item.get('uniprot_id', f'idx_{idx}')
        
#         # 2. 解析 GO 列表
#         # 根据参考代码，数据里的 key 是 'go_f_mapped'，且存的是数字索引
#         raw_go_indices = item.get('go_f_mapped', [])
        
#         sample_gos = []
#         if index_to_go:
#             # 如果有映射表，把数字转回 GO:XXXXXX
#             sample_gos = [index_to_go.get(i, str(i)) for i in raw_go_indices]
#         else:
#             # 没映射表，假设就是原始数据
#             sample_gos = raw_go_indices
            
#         sample_go_set = set(sample_gos)
        
#         # 3. 匹配逻辑
#         for name, target in target_combos.items():
#             if name in found_samples:
#                 continue # 已找到
            
#             # 判断是否包含目标组合 (目标必须是样本的子集)
#             # 例如：样本有 [A, B, C, D]，目标是 [A, B]，则 issubset 为 True
#             if target['ids'].issubset(sample_go_set):
#                 found_samples[name] = {
#                     'id': sample_id,
#                     'go_list': sample_gos,
#                     'matched_subset': target['ids']
#                 }
        
#         if len(found_samples) == 5:
#             break

#     # ================= 输出结果 =================
#     print("\n" + "="*60)
#     print("SEARCH RESULTS")
#     print("="*60)
    
#     found_count = 0
#     for name, target in target_combos.items():
#         if name in found_samples:
#             found_count += 1
#             sample = found_samples[name]
#             print(f"\n[√] {name}")
#             print(f"    Function:  {target['desc']}")
#             print(f"    Sample ID: {sample['id']}")
#             print(f"    Target GOs: {target['ids']}")
#             # print(f"    Full GOs:  {sample['go_list']}") 
#         else:
#             print(f"\n[x] {name}")
#             print(f"    Status: Not found in test set")
#             print(f"    Target GOs: {target['ids']}")

#     print("\n" + "-"*60)
#     if found_count == 5:
#         print("Done. All 5 combinations found.")
#     else:
#         print(f"Found {found_count}/5 combinations.")

# if __name__ == "__main__":
#     find_targets_in_pkl()


import pickle
import os

def create_novelty_dataset():
    # ================= 配置路径 (请确认路径正确) =================
    test_path = 'data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
    go_mapping_path = 'go_mapping.pkl'
    output_path = 'novelty.pkl'
    # ========================================================

    # 1. 加载映射表 (Index -> GO ID)
    print(f"正在加载 GO 映射: {go_mapping_path} ...")
    if not os.path.exists(go_mapping_path):
        print(f"错误: 找不到文件 {go_mapping_path}")
        return
    
    with open(go_mapping_path, 'rb') as f:
        go_mapping = pickle.load(f)
    # 翻转字典: value(index) -> key(GO:XXXXX)
    index_to_go = {v: k for k, v in go_mapping.items()}

    # 2. 加载测试集
    print(f"正在加载测试集: {test_path} ...")
    if not os.path.exists(test_path):
        print(f"错误: 找不到文件 {test_path}")
        return

    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    print(f"测试集加载完成，共 {len(test_data)} 条数据。")

    # 3. 定义 5 个目标组合
    target_combos = {
        "metalloendopeptidase": {
            "ids": {'GO:0004477', 'GO:0004488'}, 
            "rank": "Rank 8"
        },
        "tRNA_ligase": {
            "ids": {'GO:0004070', 'GO:0016597'}, 
            "rank": "Rank 11"
        },
        "NADH_dehydrogenase": {
            "ids": {'GO:0008137', 'GO:0048038', 'GO:0050136'}, 
            "rank": "Rank 14"
        },
        "RNA_binding_RNase": {
            "ids": {'GO:0003723', 'GO:0004523', 'GO:0030145'}, 
            "rank": "Rank 19"
        },
        "Iron_sulfur_cluster": {
            "ids": {'GO:0004076', 'GO:0005506', 'GO:0051537', 'GO:0051539'}, 
            "rank": "Rank 35"
        }
    }

    # 用于存储找到的样本 (Key: combo_name, Value: data_entry)
    found_entries = {}

    # 4. 遍历并提取
    print("\n开始搜索并提取目标样本...")
    
    for entry in test_data:
        # 获取当前样本的 GO set (解析 mapping)
        raw_indices = entry.get('go_f_mapped', [])
        current_go_set = set([index_to_go.get(i, str(i)) for i in raw_indices])
        
        # 检查每个目标组合
        for name, target in target_combos.items():
            if name in found_entries:
                continue # 这个类别已经找到代表了，跳过
            
            # 判定条件：样本包含目标的所有 GO (Subset)
            if target['ids'] == (current_go_set):
                print(f"[提取成功] {target['rank']} ({name}) -> ID: {entry.get('uniprot_id', 'Unknown')}")
                found_entries[name] = entry
        
        # 如果找齐了 5 个，提前结束循环
        if len(found_entries) == 5:
            break

    # 5. 组装并保存
    extracted_list = list(found_entries.values())

    repeated_list = []
    for item in extracted_list:
        repeated_list.extend([item] * 30)
    extracted_list = repeated_list
    
    if len(extracted_list) > 0:
        print(f"\n正在保存 {len(extracted_list)} 个样本到 {output_path} ...")
        with open(output_path, 'wb') as f:
            pickle.dump(extracted_list, f)
        print("✅ 完成！文件已生成。")
        
        # 打印一下提取结果的摘要，确认没问题
        print("\n=== novelty.pkl 内容摘要 ===")
        for i, item in enumerate(extracted_list):
            uid = item.get('uniprot_id', 'Unknown')
            indices = item.get('go_f_mapped', [])
            gos = [index_to_go.get(x) for x in indices]
            print(f"Index {i}: ID={uid}, GO_Count={len(gos)}")
            # print(f"    GOs: {gos}") 
    else:
        print("\n⚠️ 警告：没有找到任何符合条件的样本，文件未生成。")

if __name__ == "__main__":
    create_novelty_dataset()