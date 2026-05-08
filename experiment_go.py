import pickle
from tqdm import tqdm

# ================= 配置路径 =================
test_path = 'data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
go_mapping_path = 'go_mapping.pkl'

# ================= 你感兴趣的目标组合 (Hardcoded) =================
# 我把你列表里的 Top 组合以及我推荐的 Case Study 都放进去了
target_combinations = [
    # 1. Top Count (26) - 这里的组合看起来像是一个复杂的酶复合物
    tuple(sorted(['GO:0036218', 'GO:0036221', 'GO:0047429'])),
    
    # 2. Count 11 - 晶状体结构 + 蛋白结合 (Lens + Identical protein binding)
    tuple(sorted(['GO:0005212', 'GO:0042802'])),
    
    # 3. Count 8 - RNA结合 + 锌指 (RNA binding + Zinc ion binding)
    tuple(sorted(['GO:0003723', 'GO:0008990', 'GO:0052915', 'GO:0070043'])),

    # 4. Count 5 [推荐 Case] - 兼职蛋白 (Arginase activity + Lens structural)
    tuple(sorted(['GO:0004056', 'GO:0005212'])),
]

# 为了方便打印，建立一个查找表
target_set = set(target_combinations)
results = {combo: [] for combo in target_combinations}

# ================= 1. 加载数据 =================
print("Loading data...")
with open(go_mapping_path, 'rb') as f:
    go_mapping = pickle.load(f)

print(go_mapping)
exit()
index_to_go = {v: k for k, v in go_mapping.items()}

with open(test_path, 'rb') as f:
    test_data = pickle.load(f)

# ================= 2. 扫描并匹配 =================
print(f"Scanning {len(test_data)} test samples...")

for entry in test_data:
    indices = entry['go_f_mapped']
    protein_id = entry['uniprot_id'] # 假设你的数据里字段叫 uniprot_id
    
    # 转换并排序
    go_ids = tuple(sorted([index_to_go[i] for i in indices]))
    
    if go_ids in target_set:
        results[go_ids].append(protein_id)

# ================= 3. 输出结果 =================
print("\n" + "="*60)
print("MATCHED UNIPROT IDs FOR CASE STUDIES")
print("="*60)

for combo in target_combinations:
    ids = results[combo]
    print(f"\nCombination: {', '.join(combo)}")
    print(f"Count: {len(ids)}")
    print("-" * 30)
    
    # 打印 ID，如果是 Jupyter 环境可以直接点击链接
    for pid in ids:
        print(f"{pid}  ->  https://www.uniprot.org/uniprotkb/{pid}")