# import pickle
# from collections import Counter
# from tqdm import tqdm

# # 1. Load the Mapping
# print("Loading GO mapping...")
# with open('go_mapping.pkl', 'rb') as f:
#     go_mapping = pickle.load(f)
# # Reverse mapping to get GO ID from Index
# index_to_go = {v: k for k, v in go_mapping.items()}

# # 2. Define paths
# paths = {
#     "Test": "data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl",
#     "Train": "data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl"
# }

# def analyze_combinations(dataset_name, file_path):
#     print(f"\n--- Loading {dataset_name} Data ---")
#     with open(file_path, 'rb') as f:
#         data_list = pickle.load(f)
    
#     combination_counter = Counter()
    
#     print(f"Counting combinations for {len(data_list)} entries...")
    
#     for entry in tqdm(data_list):
#         # Extract indices
#         indices = entry['go_f_mapped']
        
#         # Convert to GO IDs
#         go_ids = [index_to_go[i] for i in indices]
        
#         # KEY STEP: Sort and convert to tuple to make it hashable and order-independent
#         # e.g., ['GO:002', 'GO:001'] becomes ('GO:001', 'GO:002')
#         combo_key = tuple(sorted(go_ids))
        
#         combination_counter[combo_key] += 1

#     return combination_counter, len(data_list)

# # 3. Run Analysis
# train_counts, train_total = analyze_combinations("Train", paths["Train"])
# test_counts, test_total = analyze_combinations("Test", paths["Test"])

# # 4. Display Results
# def print_stats(name, counter, total):
#     print(f"\nResults for {name}:")
#     print(f"Total Samples: {total}")
#     print(f"Unique Combinations: {len(counter)}")
    
#     print(f"Top 10 Most Frequent Combinations:")
#     for combo, count in counter.most_common(10):
#         percentage = (count / total) * 100
#         print(f"  {count:5d} ({percentage:.2f}%) -> {combo}")

# print_stats("Train", train_counts, train_total)
# print_stats("Test", test_counts, test_total)

# # 5. Check for "Zero-Shot" Combinations (Combinations in Test but NOT in Train)
# test_keys = set(test_counts.keys())
# train_keys = set(train_counts.keys())
# unseen_combinations = test_keys - train_keys

# print(f"\n--- Distribution Analysis ---")
# print(f"Number of unique combinations in Test that represent UNSEEN combinations (not in Train): {len(unseen_combinations)}")
# print(f"This represents {len(unseen_combinations)/len(test_keys)*100:.2f}% of the unique combinations in the test set.")


# # not strict unseen
# import pickle
# from collections import Counter
# from tqdm import tqdm

# # ================= 配置路径 =================
# go_mapping_path = 'go_mapping.pkl'
# train_path = 'data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
# test_path = 'data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'

# # ================= 1. 加载映射 =================
# print(f"Loading GO mapping from {go_mapping_path}...")
# with open(go_mapping_path, 'rb') as f:
#     go_mapping = pickle.load(f)
# # 反转映射: index -> GO:ID
# index_to_go = {v: k for k, v in go_mapping.items()}

# # ================= 2. 定义处理函数 =================
# def get_dataset_combinations(file_path, dataset_name):
#     """
#     读取数据集，返回所有样本的 GO 组合列表 (list of tuples)
#     """
#     print(f"Loading {dataset_name} data from {file_path}...")
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
    
#     combinations = []
#     print(f"Processing {dataset_name}...")
#     for entry in tqdm(data):
#         indices = entry['go_f_mapped']
#         # 转换 index 为 GO ID
#         go_ids = [index_to_go[i] for i in indices]
#         # 排序并转为 tuple，保证 hashable 且无序一致
#         combo = tuple(sorted(go_ids))
#         combinations.append(combo)
        
#     return combinations

# # ================= 3. 执行统计 =================

# # --- A. 处理训练集 ---
# train_combos_list = get_dataset_combinations(train_path, "TRAIN")
# # 将训练集组合存为集合 (Set)，用于快速查找
# train_combos_set = set(train_combos_list)

# print(f"\n[Train Stats]")
# print(f"Total Samples: {len(train_combos_list)}")
# print(f"Unique Combinations: {len(train_combos_set)}")

# # --- B. 处理测试集并寻找 Unseen ---
# test_combos_list = get_dataset_combinations(test_path, "TEST")

# unseen_combos_list = [] # 存储具体的 Unseen 样本组合
# seen_count = 0

# for combo in test_combos_list:
#     if combo not in train_combos_set:
#         unseen_combos_list.append(combo)
#     else:
#         seen_count += 1

# # ================= 4. 输出结果 =================

# unseen_count = len(unseen_combos_list)
# total_test = len(test_combos_list)
# unseen_ratio = (unseen_count / total_test) * 100

# print(f"\n" + "="*40)
# print(f"       UNSEEN COMBINATION ANALYSIS")
# print(f"="*40)
# print(f"Total Test Samples:      {total_test}")
# print(f"Seen Combinations:       {seen_count} (出现在训练集中)")
# print(f"Unseen Combinations:     {unseen_count} (仅在测试集中出现)")
# print(f"Unseen Ratio:            {unseen_ratio:.2f}%")

# # 统计 Unseen 组合的具体分布（哪些 Unseen 组合出现得最多？）
# unseen_counter = Counter(unseen_combos_list)

# print(f"\nTop 10 Most Frequent 'Unseen' Combinations in Test:")
# print(f"{'Count':<8} | {'GO Combination'}")
# print("-" * 60)
# for combo, count in unseen_counter.most_common(10):
#     print(f"{count:<8} | {', '.join(combo)}")

# print(f"\nTotal unique unseen combination types: {len(unseen_counter)}")

# # ================= 5. (可选) 保存结果 =================
# # 如果你想把这些 Unseen 的具体数据存下来进一步分析
# # with open('unseen_combinations_stats.pkl', 'wb') as f:
# #     pickle.dump(unseen_counter, f)

import pickle
from collections import defaultdict, Counter
from tqdm import tqdm

# ================= 配置 =================
train_path = 'data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
test_path = 'data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
go_mapping_path = 'go_mapping.pkl'

# ================= 1. 加载数据 =================
print("Loading data...")
with open(go_mapping_path, 'rb') as f:
    go_mapping = pickle.load(f)
index_to_go = {v: k for k, v in go_mapping.items()}

with open(train_path, 'rb') as f:
    train_data = pickle.load(f)

with open(test_path, 'rb') as f:
    test_data = pickle.load(f)

# ================= 2. 构建训练集倒排索引 (Inverted Index) =================
# key: GO term index (int), value: set of training sample indices containing this GO
go_to_train_indices = defaultdict(set)

print(f"Building Inverted Index for {len(train_data)} training samples...")
for train_idx, entry in enumerate(tqdm(train_data)):
    go_indices = entry['go_f_mapped']
    for go_idx in go_indices:
        go_to_train_indices[go_idx].add(train_idx)

# ================= 3. 检测测试集 (Strict Unseen Check) =================
print(f"Checking {len(test_data)} test samples for strict unseen status...")

strict_unseen_combos = []
seen_count = 0

for entry in tqdm(test_data):
    test_go_indices = entry['go_f_mapped']
    
    # 如果测试样本没有 GO 标签（空列表），视具体情况处理，这里跳过或视为 Seen
    if not test_go_indices:
        seen_count += 1
        continue
        
    # 获取该测试样本中每个 GO 对应的训练样本集合列表
    # 例如：[{样本1, 样本3}, {样本2, 样本3}]
    sets_to_intersect = [go_to_train_indices[go_idx] for go_idx in test_go_indices]
    
    # 计算交集
    # 如果交集非空，意味着存在至少一个训练样本，包含了所有的 test_go_indices
    # 即：test_go_indices is subset of that train sample
    intersection = set.intersection(*sets_to_intersect)
    
    if len(intersection) > 0:
        # 这是一个子集 (Seen)
        seen_count += 1
    else:
        # 这是一个 Strictly Unseen 组合
        # 转换为 GO ID 字符串以便统计和阅读
        combo_tuple = tuple(sorted([index_to_go[i] for i in test_go_indices]))
        strict_unseen_combos.append(combo_tuple)

# ================= 4. 统计结果 =================
total_test = len(test_data)
strict_unseen_count = len(strict_unseen_combos)
ratio = (strict_unseen_count / total_test) * 100

print(f"\n" + "="*50)
print(f"       STRICT UNSEEN COMBINATION ANALYSIS")
print(f"       (Condition: Test combo is NOT a subset of ANY Train combo)")
print(f"="*50)
print(f"Total Test Samples:      {total_test}")
print(f"Strictly Seen:           {seen_count}")
print(f"Strictly Unseen:         {strict_unseen_count}")
print(f"Strict Unseen Ratio:     {ratio:.2f}%")

# 统计 Unseen 组合的具体分布
unseen_counter = Counter(strict_unseen_combos)

print(f"\nTop 10 Strict Unseen Combinations:")
print(f"{'Count':<6} | {'GO Combination'}")
print("-" * 80)
for combo, count in unseen_counter.most_common(10):
    # 为了显示简洁，截断过长的输出
    combo_str = ', '.join(combo)
    if len(combo_str) > 70:
        combo_str = combo_str[:67] + "..."
    print(f"{count:<6} | {combo_str}")