import pickle
from collections import defaultdict
import os
from tqdm import tqdm

# ================= 配置路径 =================
# 请确保这些路径与你之前的脚本一致
TRAIN_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
TEST_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
GO_MAPPING_PATH = 'go_mapping.pkl'

# 输出文件路径
OUTPUT_PATH = 'test_strict_unseen_repeated_10x.pkl'

# 重复次数
REPEAT_TIMES = 10

def load_pickle(path):
    print(f"Loading {path}...")
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    # 1. 加载数据
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        raise FileNotFoundError("训练集或测试集路径不存在，请检查配置。")
        
    train_data = load_pickle(TRAIN_PATH)
    test_data = load_pickle(TEST_PATH)
    go_mapping = load_pickle(GO_MAPPING_PATH)
    
    # 建立 ID 映射 (Index -> GO Term)
    index_to_go = {v: k for k, v in go_mapping.items()}

    # 2. 构建训练集索引 (用于快速查重)
    print("Building training set indexes for filtration...")
    
    go_to_train_indices = defaultdict(set) # 倒排索引: GO ID -> 包含该 GO 的训练样本下标集合
    train_combos_set = set()               # 精确组合集合
    valid_train_gos = set()                # 训练集中出现过的所有单个 GO ID
    
    for idx, entry in enumerate(tqdm(train_data, desc="Indexing Train")):
        # 获取该样本的所有 GO ID
        go_ids = [index_to_go[i] for i in entry['go_f_mapped']]
        
        # 存入组合集
        combo = tuple(sorted(go_ids))
        train_combos_set.add(combo)
        
        # 存入倒排索引和有效原子集
        for go_id in go_ids:
            go_to_train_indices[go_id].add(idx)
            valid_train_gos.add(go_id)

    # 3. 定义筛选逻辑
    def is_strict_unseen(go_list):
        """
        判断条件:
        1. 所有的单个 GO 必须在训练集中出现过 (Known Atoms)。
        2. 组合本身不能在训练集中出现过 (Unseen Composition)。
        3. 组合不能是训练集中任何一个样本的子集 (Strictly Unseen)。
        """
        if not go_list: 
            return False
            
        # 条件 1: 原子必须已知
        if not all(go_id in valid_train_gos for go_id in go_list):
            return False
            
        # 条件 2: 组合必须未知
        if tuple(sorted(go_list)) in train_combos_set:
            return False
            
        # 条件 3: 不能是子集
        # 逻辑: 找到包含列表中所有 GO 的训练样本。如果有交集，说明是子集。
        sets_to_intersect = [go_to_train_indices[go_id] for go_id in go_list]
        if not sets_to_intersect: 
            return True # 理论上上面已拦截，防御性编程
            
        # 计算交集
        common_indices = set.intersection(*sets_to_intersect)
        if len(common_indices) > 0:
            return False # 是某训练样本的子集
            
        return True

    # 4. 处理测试集并重复
    new_dataset = []
    selected_count = 0
    
    print(f"Filtering test set and repeating {REPEAT_TIMES} times...")
    
    for entry in tqdm(test_data, desc="Processing Test"):
        gt_go_ids = [index_to_go[i] for i in entry['go_f_mapped']]
        
        if is_strict_unseen(gt_go_ids):
            selected_count += 1
            # 重复添加 N 遍
            for _ in range(REPEAT_TIMES):
                # 注意：这里添加的是同一个对象的引用。
                # 如果你需要后续修改每个副本而不影响其他副本，建议使用 copy.deepcopy(entry)
                # 但对于只读/推理任务，直接添加引用即可，节省内存。
                new_dataset.append(entry)

    # 5. 保存结果
    print("\n" + "="*40)
    print("SUMMARY")
    print("="*40)
    print(f"Original Test Size:      {len(test_data)}")
    print(f"Selected Unseen Entries: {selected_count}")
    print(f"Multiplication Factor:   x{REPEAT_TIMES}")
    print(f"Final Dataset Size:      {len(new_dataset)}")
    
    print(f"Saving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(new_dataset, f)
    print(f"new dataset: {new_dataset[:1]}")
    print("Done!")

if __name__ == '__main__':
    main()