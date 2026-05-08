import pickle
from tqdm import tqdm

# ================= 路径配置 =================
# 1. 你刚刚生成的 New Data 路径
NEW_DATA_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/generated_candidates_motif_emb.pkl'

# 2. 原始数据集路径 (必须包含 train/test划分，或者你明确知道它是测试集)
# 注意：这里需要你原始的包含 'split': 'test' 的数据文件
ORIGINAL_DATA_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/test.pkl' 

def check_overlap():
    print(f"Loading New Data from {NEW_DATA_PATH}...")
    with open(NEW_DATA_PATH, 'rb') as f:
        new_data = pickle.load(f)

    print(f"Loading Original Data from {ORIGINAL_DATA_PATH}...")
    with open(ORIGINAL_DATA_PATH, 'rb') as f:
        original_data = pickle.load(f)

    # -------------------------------------------------
    # 1. 提取 Test Set 的指纹 (Signature)
    # -------------------------------------------------
    # 假设原始数据里有 'split' 字段，或者你手动指定 test_data = original_data
    if isinstance(original_data, list) and 'split' in original_data[0]:
        test_data = [x for x in original_data if x['split'] == 'test']
        train_data = [x for x in original_data if x['split'] == 'train']
        print(f"Identified {len(train_data)} Train samples and {len(test_data)} Test samples.")
    else:
        print("Warning: Could not auto-detect split. Assuming WHOLE original file is the comparison target.")
        test_data = original_data
        train_data = []

    # 构建 Test Set 的组合集合 (使用 frozenset 忽略顺序)
    test_combos = set()
    test_vocab = set()
    for item in test_data:
        # 兼容不同的键名，根据你之前的代码，这里用 'go_numbers' -> 'F'
        if 'go_numbers' in item and 'F' in item['go_numbers']:
            gos = item['go_numbers']['F']
        elif 'go_list' in item:
            gos = item['go_list']
        else:
            continue # 无法识别
            
        combo = frozenset(gos)
        test_combos.add(combo)
        test_vocab.update(combo)

    # 构建 Train Set 组合集合 (用于检测 OOD)
    train_combos = set()
    for item in train_data:
         if 'go_numbers' in item and 'F' in item['go_numbers']:
            gos = item['go_numbers']['F']
            train_combos.add(frozenset(gos))

    # -------------------------------------------------
    # 2. 检查 New Data
    # -------------------------------------------------
    new_vocab_unknown = set()
    matches_in_test = 0
    matches_in_train = 0
    is_ood_count = 0
    
    # 你的新数据有10个副本，我们为了统计准确，可以去重来看
    unique_new_entries = {} # Map combo -> first entry
    
    for item in new_data:
        gos = item['go_numbers']['F']
        combo = frozenset(gos)
        unique_new_entries[combo] = item
        
        # 检查单词表
        for go_id in gos:
            if go_id not in test_vocab and go_id not in test_vocab: 
                # 注意：有些标签可能只在 Train 有，Test 没有，这里严格检查是否在 Test 出现过
                pass 
                
    print(f"\n--- Analysis of {len(unique_new_entries)} Unique Combinations in New Data ---")

    for combo in unique_new_entries:
        # Check 1: 是否在 Test Set 出现过？
        if combo in test_combos:
            matches_in_test += 1
        
        # Check 2: 是否在 Train Set 出现过？
        if combo in train_combos:
            matches_in_train += 1
        else:
            is_ood_count += 1
            
    # -------------------------------------------------
    # 3. 输出报告
    # -------------------------------------------------
    print(f"\n[结果报告]")
    print(f"1. 新数据中的唯一组合数: {len(unique_new_entries)}")
    print(f"2. 存在于 Test Set 中的组合数: {matches_in_test} (占比 {matches_in_test/len(unique_new_entries):.2%})")
    print(f"3. 存在于 Train Set 中的组合数: {matches_in_train}")
    print(f"4. OOD 组合数 (未在 Train 中出现): {is_ood_count}")
    
    if matches_in_test == len(unique_new_entries):
        print("\n✅ 结论: 你的新数据完全是 Test Set 的子集 (或完全重合)。")
    elif matches_in_test == 0:
        print("\n⚠️ 结论: 你的新数据与 Test Set 完全不重合 (可能是全新的生成任务)。")
    else:
        print("\nℹ️ 结论: 你的新数据部分包含 Test Set 的组合，部分是新的。")

    if is_ood_count > 0:
        print(f"🌟 发现 {is_ood_count} 个 OOD 样本 (符合 ICML 对 Unseen 组合的描述)。")

if __name__ == "__main__":
    check_overlap()