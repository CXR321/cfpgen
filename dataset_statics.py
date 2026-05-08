from load_all_train_data import load_all_pfam_emb_data
from collections import Counter

def get_mapped_tuple(data_entry):
    """提取 go_f_mapped 并转为可哈希的元组，并进行排序确保 [0, 1] 和 [1, 0] 视为同一种组合"""
    if 'go_f_mapped' in data_entry:
        # 排序非常重要，防止顺序不同导致统计错误
        return tuple(sorted(data_entry['go_f_mapped']))
    return tuple()

def analyze_combinations():
    print("正在加载数据...")
    # 1. 加载所有数据
    train_data = load_all_pfam_emb_data("train")
    test_data = load_all_pfam_emb_data("test")
    
    print(f"数据加载完成。训练集: {len(train_data)} 条, 测试集: {len(test_data)} 条")

    # 2. 统计训练集 (Train)
    train_combinations = set()
    for data in train_data:
        combo = get_mapped_tuple(data)
        train_combinations.add(combo)
    
    # 3. 统计测试集 (Test) 及其频率
    test_combinations_counter = Counter()
    for data in test_data:
        combo = get_mapped_tuple(data)
        test_combinations_counter[combo] += 1
    
    unique_test_combinations = set(test_combinations_counter.keys())

    # 4. 计算 Test 中未见过 (Unseen) 的组合
    unseen_in_test = unique_test_combinations - train_combinations
    
    # 5. 计算 Test 集中每个组合的平均蛋白条数
    # (总条数 / 唯一组合数)
    avg_proteins_per_combo = len(test_data) / len(unique_test_combinations) if unique_test_combinations else 0

    print("-" * 40)
    print("统计结果 (基于 go_f_mapped 组合):")
    print("-" * 40)
    print(f"1. 训练集独特的组合个数: {len(train_combinations)}")
    print(f"2. 测试集独特的组合个数: {len(unique_test_combinations)}")
    print(f"3. 测试集中【未在训练集出现过】的新组合个数: {len(unseen_in_test)}")
    print(f"   (占比: {len(unseen_in_test) / len(unique_test_combinations):.2%})")
    print(f"4. 测试集中每个组合平均包含的蛋白条数: {avg_proteins_per_combo:.2f}")
    
    # 额外：如果你想看看那些未见过的组合长什么样，可以打印前5个
    if unseen_in_test:
        print("\n[示例] 测试集出现的新组合 (前5个):")
        for i, combo in enumerate(list(unseen_in_test)[:5]):
            print(f"  - {list(combo)} (该组合在测试集中出现了 {test_combinations_counter[combo]} 次)")


# if __name__ == "__main__":
#     analyze_combinations()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter

# Set style for ICML paper
sns.set_context("paper", font_scale=1.4)
sns.set_style("ticks")

# ==========================================
# REPLACE THIS BLOCK WITH YOUR DATA LOADING
# ==========================================
# Assuming 'data' is your list of entries, e.g.:
# train_data = [entry for entry in data if entry['split'] == 'train']
# test_data = [entry for entry in data if entry['split'] == 'test']
#
# Each entry should look like: {'go_f_mapped': [1, 5, 20]}
# ==========================================

train_data = load_all_pfam_emb_data("train")
test_data = load_all_pfam_emb_data("test")

# (The following is placeholder code to demonstrate structure if you don't have the lists ready)
train_entries = [d['go_f_mapped'] for d in train_data] 
test_entries = [d['go_f_mapped'] for d in test_data]

# ---------------------------------------------------------
# Plotting Script
# ---------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# --- Plot A: Label Frequency Distribution (Log-Log) ---
# Flatten lists to count individual labels
all_train_labels = [lbl for sublist in train_entries for lbl in sublist]
all_test_labels = [lbl for sublist in test_entries for lbl in sublist]

train_counts = Counter(all_train_labels)
test_counts = Counter(all_test_labels)

# specific num_terms if known, else max ID
num_terms = max(max(train_counts.keys()), max(test_counts.keys())) + 1

df_freq = pd.DataFrame({
    'Term ID': range(num_terms),
    'Train Freq': [train_counts.get(i, 0) for i in range(num_terms)],
    'Test Freq': [test_counts.get(i, 0) for i in range(num_terms)]
})
# Sort by Train Frequency
df_freq = df_freq.sort_values('Train Freq', ascending=False).reset_index(drop=True)

axes[0, 0].plot(df_freq.index, df_freq['Train Freq'], label='Train', linewidth=2, alpha=0.8)
axes[0, 0].plot(df_freq.index, df_freq['Test Freq'], label='Test', linewidth=2, alpha=0.8, linestyle='--')
axes[0, 0].set_yscale('log')
axes[0, 0].set_title('(a) Label Frequency Distribution', fontweight='bold')
axes[0, 0].set_xlabel('Label Index (Sorted by Freq)')
axes[0, 0].set_ylabel('Frequency (Log Scale)')
axes[0, 0].legend()
axes[0, 0].grid(True, which="both", ls="-", alpha=0.2)

# --- Plot B: Number of Labels per Sequence ---
train_lens = [len(x) for x in train_entries]
test_lens = [len(x) for x in test_entries]

sns.histplot(train_lens, discrete=True, color='blue', alpha=0.4, label='Train', ax=axes[0, 1], stat='density', common_norm=False)
sns.histplot(test_lens, discrete=True, color='orange', alpha=0.4, label='Test', ax=axes[0, 1], stat='density', common_norm=False)
axes[0, 1].set_title('(b) Labels per Sequence Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Number of GO Terms')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend()

# --- Plot C: Test Set Composition (OOD vs ID) ---
# Identify unique combinations in train
train_combos = set(tuple(sorted(x)) for x in train_entries)

# Check each test sequence
test_combos = [tuple(sorted(x)) for x in test_entries]
is_ood = [t not in train_combos for t in test_combos]

ood_seq_count = sum(is_ood)
id_seq_count = len(test_entries) - ood_seq_count

bars = axes[1, 0].bar(['Seen (ID)', 'Unseen (OOD)'], [id_seq_count, ood_seq_count], color=['#3498db', '#e74c3c'])
axes[1, 0].set_title('(c) Test Set Composition (Sequences)', fontweight='bold')
axes[1, 0].set_ylabel('Count')
for bar in bars:
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height, f'{height}', ha='center', va='bottom')

# --- Plot D: Co-occurrence Heatmap (Top 15) ---
top_labels = df_freq['Term ID'].head(15).values
cooc_matrix = np.zeros((15, 15))

for lbls in train_entries:
    # Filter for top labels only to speed up
    current = [l for l in lbls if l in top_labels]
    for i in range(len(current)):
        for j in range(i + 1, len(current)):
            idx_i = np.where(top_labels == current[i])[0][0]
            idx_j = np.where(top_labels == current[j])[0][0]
            cooc_matrix[idx_i, idx_j] += 1
            cooc_matrix[idx_j, idx_i] += 1

mask = np.eye(15, dtype=bool)
sns.heatmap(pd.DataFrame(cooc_matrix, index=top_labels, columns=top_labels), 
            mask=mask, ax=axes[1, 1], cmap='viridis', cbar_kws={'label': 'Count'})
axes[1, 1].set_title('(d) Top-15 Label Co-occurrence', fontweight='bold')

plt.tight_layout()
plt.show()
plt.savefig('dataset_statics.png')