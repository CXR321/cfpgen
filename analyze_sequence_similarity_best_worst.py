import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from difflib import SequenceMatcher
import random
from itertools import combinations

# ================= 配置区域 =================
TRAIN_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
# 使用上一各步骤生成的 CSV (确保你先运行了上一个脚本)
# 如果没有 CSV，你需要把上个脚本里生成 selected_df 的逻辑拷过来
INPUT_CSV = 'analysis_top_bottom_go_terms_recall.csv' 
GO_MAPPING_PATH = 'go_mapping.pkl'

# 计算相似度时的采样限制 (防止训练集该词有几千条序列导致计算爆炸)
# 30条序列 => C(30,2) = 435 次比对，速度可接受且统计意义足够
MAX_SEQS_PER_GO = 30 
SEED = 42

from Bio import Align

from Bio import Align

def calculate_sequence_identity(seq1, seq2):
    """
    使用 Biopython 计算序列一致性 (兼容旧版本)。
    通过设置 match_score=1, 其他=0，使得 score 直接等于 matches 数量。
    """
    if not seq1 or not seq2:
        return 0.0
    
    if seq1 == seq2:
        return 1.0

    # 1. 创建比对器
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    
    # 2. 关键：设置打分矩阵
    # 这样设置后，alignment.score 就直接等于 Match 的字符个数
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = 0.0
    aligner.extend_gap_score = 0.0

    # 3. 执行比对
    try:
        alignment = aligner.align(seq1, seq2)[0]
    except IndexError:
        return 0.0
    
    # 4. 计算一致性 (兼容旧写法)
    # alignment.score: 在上述设置下等于 matches 数量
    # alignment.shape[1]: 等于比对的总长度 (alignment length, 包含 gap)
    matches = alignment.score
    total_len = alignment.shape[1]
    
    return matches / total_len if total_len > 0 else 0.0

s1="MGKKAIQFGGGNIGRGFVAEFLHEAGYEVVFIDVVDKIIDALKSTPSYEVTEVSEEGEKTKTITNYRAINSKTNEEDVVKEIGTADVVTCAVGPNVLKFIAPVIAKGIDARTASKPVAVIACENAIGATDTLRGFIEQNTDKDRLSSMSERARFANSAIDRIVPNQPPNAGLNVRIEKFYEWTVEQTPFGEFGHPDIPAIHWVDDLKPYIERKLFTVNTGHATTAYYGHMRGKKMIADALADAEIRQIVHKVLEQTAKLITTKHEITEQEQNEYVDTIVKRMSNPFLEDNVERVGRAPLRKLSRNERFIGPASQLAEKGLPFDALLGSIEMALRFQNVPGDEESAELAKILKEMSAEEATGKLTGLEKHHPLYEPVQNVIAKVQKDSK"
s1="MKAVHFGAGNIGRGFVGLLLHEAGYEVVFADVNAELIEALAAADSYDVRLVGDETVTKTVTGFRALNSATAEDELVGEIATADVVTTAVGPRILRFVAPVIARGLGARASGAPRLVVMACENAIGATDLLAAELMDGLQDGDTRDELAARAVFANTAVDRIVPAQDPASGVDVTVEAFFEWVVDRSPFAGAEPPIPGAHFVDSLGPYIERKLFTVNTGHATTAYTGFLEGAATISEAIAKPSVLATVEAVLEETSAALTAKHGLDPEELAEYRAKILNRFRNPHLVDEVTRVGREPLRKLGRNDRFIGPASDYARYVGGAPSALLAAVGAALRFDLPGDSQSAELQQLLRKTEPGGIVTEVMGVEPGDALYEPLVAVVRTAQG"
s2="MKAVHFGAGNIGRGFIGKLLADNGIEVTFADVNQPVIDALNARHSYEVNVVGENAQTDVVKNVAGINSMQEPEKVVEAIATADLVTTAVGPNILPIIAPLIAKGIVRRHETNDRPLNIIACENMVRGTTQLKGAVFDHLPEEHKAWVEEHVGFVDSAVDRIVPPSASEDILAVTVETFSEWIVDKTQFKGTLPNIPGMELTDNLMAFVERKLFTLNTGHAITAYLGQLAGHKTIRDAILDPQIRATVKGAMEESGSVLIKRYGFDREKHAAYIEKIIARFENPYLSDEVERVAGETIRKLGPNERLTKPLAGILEYDLPHDKLVEAYNSL"
s2 = "MRAVHFGAGNIGRGFIGETLAANGFGIDFVDVNDTIISALNQRGEYDIELAVPGQKKIHVENVDGINNGEHPEKVVEAIKTTDLVTTAIGPKILKFIAPLIADGIKARKDADNKQTLDVIACENMIGGSQHLKEEVYSHLDDDVKAFADEYIGFPNAAVDRIVPLQKHDDPLLVSVEEFKEWVVDESQMKNPSLKLETVHYAPDLEPYIERKLFSVNTGHATVAYTGKALGYETIGEAIKDEKVLTQLKNTLKEIRSLLLAKWDFQEKELEEYHDKIIGRFENPYISDEIARVGRTPIRKLGYDERFIRPIRELKARGLSYTTLMETVGKIYHFDEPKDDESQKLQKMLKDEDLKDVIVETTGLKDDADLVAEIAASVKAAD"

print(f"Align: {calculate_sequence_identity(s1, s2)}")
exit(0)

def get_intra_class_similarity(seq_list, max_samples=30):
    """
    计算集合内部的平均两两相似度
    """
    n = len(seq_list)
    if n < 2:
        return 0.0, 0.0 # 无法计算
    
    # 随机采样
    if n > max_samples:
        random.seed(SEED)
        seq_list = random.sample(seq_list, max_samples)
    
    similarities = []
    # 使用 combinations 生成所有不重复对子
    for s1, s2 in combinations(seq_list, 2):
        sim = calculate_sequence_identity(s1, s2)
        similarities.append(sim)
        
    return np.mean(similarities), np.std(similarities)

# ================= 主流程 =================
def main():
    # 1. 加载目标 GO IDs
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"找不到 {INPUT_CSV}，请先运行上一个'analyze_best_worst_terms_filtered.py'脚本生成该文件。")
    
    print(f"Loading target GO terms from {INPUT_CSV}...")
    df_targets = pd.read_csv(INPUT_CSV)
    
    # 区分 Best 和 Worst 组
    # 假设前5个是 Best，后5个是 Worst (根据上一个脚本的逻辑)
    # 或者我们可以根据 d_rec_ours 值来判断：正值大的是Best，负值大的是Worst
    # 为了保险，重新排一下序
    df_targets = df_targets.sort_values(by=['d_rec_ours'], ascending=False)
    
    # 提取 ID 列表
    # 假设 CSV 里有 10 行，前 5 行 Best，后 5 行 Worst
    best_gos = df_targets.head(5)['go_id'].tolist()
    worst_gos = df_targets.tail(5)['go_id'].tolist()
    
    target_go_set = set(best_gos + worst_gos)
    print(f"Targets identified:\n  Best: {best_gos}\n  Worst: {worst_gos}")

    # 2. 加载训练集并提取序列
    print("Loading Training Data and Indexing Sequences...")
    with open(GO_MAPPING_PATH, 'rb') as f:
        index_to_go = {v: k for k, v in pickle.load(f).items()}
        
    with open(TRAIN_PATH, 'rb') as f:
        train_data = pickle.load(f)
        
    # 构建 GO -> List[Sequences]
    go_to_seqs = defaultdict(list)
    
    for entry in tqdm(train_data):
        # 确保 entry 里有序列字段 (通常是 'sequence' 或 'seq')
        seq = entry.get('sequence', entry.get('seq', ''))
        if not seq: continue
        
        # 获取该样本的所有 GO
        gos = [index_to_go[i] for i in entry['go_f_mapped']]
        
        # 只收集我们在意的 GO
        for go in gos:
            if go in target_go_set:
                go_to_seqs[go].append(seq)

    # 3. 计算相似度
    print(f"Calculating Intra-class Sequence Similarity (Max {MAX_SEQS_PER_GO} seqs per term)...")
    
    results = []
    
    # Helper to process a list of GOs
    def process_group(go_list, group_label):
        for go_id in go_list:
            seqs = go_to_seqs.get(go_id, [])
            count = len(seqs)
            
            if count < 2:
                print(f"Warning: {go_id} has < 2 sequences in train. Skipping.")
                continue
                
            mean_sim, std_sim = get_intra_class_similarity(seqs, MAX_SEQS_PER_GO)
            
            # 获取 CSV 中的其他信息用于画图
            row = df_targets[df_targets['go_id'] == go_id].iloc[0]
            name = row['name']
            
            results.append({
                'go_id': go_id,
                'name': name,
                'group': group_label,
                'train_count': count,
                'mean_similarity': mean_sim,
                'std_similarity': std_sim
            })
            print(f"  {go_id} ({group_label}): n={count}, Mean Sim={mean_sim:.4f}")

    process_group(best_gos, 'Best (High Delta)')
    process_group(worst_gos, 'Worst (Low Delta)')
    
    df_res = pd.DataFrame(results)

    # 4. 绘图
    print("Plotting...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 保持 CSV 中的顺序
    # 我们需要在 df_res 中按照 best_gos + worst_gos 的顺序排序
    df_res['go_id'] = pd.Categorical(df_res['go_id'], categories=best_gos + worst_gos, ordered=True)
    df_res = df_res.sort_values('go_id')
    
    x_pos = np.arange(len(df_res))
    
    # 颜色区分
    colors = ['#1f77b4' if g == 'Best (High Delta)' else '#d62728' for g in df_res['group']]
    
    # 绘制柱状图 (带误差棒)
    bars = ax.bar(x_pos, df_res['mean_similarity'], yerr=df_res['std_similarity'], 
                  align='center', alpha=0.8, color=colors, capsize=5, edgecolor='black')
    
    # 装饰
    ax.set_ylabel('Avg Pairwise Sequence Identity (Train Set)', fontsize=12)
    ax.set_title('Does Sequence Conservation explain Model Performance?\n(Intra-class Similarity of Training Samples)', fontsize=14)
    ax.set_xticks(x_pos)
    
    # 标签格式化
    def fmt_label(row):
        return f"{row['go_id']}\n(n={row['train_count']})"
    
    ax.set_xticklabels(df_res.apply(fmt_label, axis=1), rotation=0, fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # 添加分割线
    mid_point = 4.5
    ax.axvline(mid_point, color='gray', linestyle='--')
    ax.text(mid_point - 0.5, ax.get_ylim()[1]*0.95, 'Best Terms', color='#1f77b4', fontweight='bold', ha='right')
    ax.text(mid_point + 0.5, ax.get_ylim()[1]*0.95, 'Worst Terms', color='#d62728', fontweight='bold', ha='left')
    
    # 计算组间均值
    avg_best = df_res[df_res['group'] == 'Best (High Delta)']['mean_similarity'].mean()
    avg_worst = df_res[df_res['group'] == 'Worst (Low Delta)']['mean_similarity'].mean()
    
    print(f"\nSummary:")
    print(f"Average Similarity of BEST terms:  {avg_best:.4f}")
    print(f"Average Similarity of WORST terms: {avg_worst:.4f}")
    
    # 在图上标注组均值
    ax.axhline(avg_best, color='#1f77b4', linestyle=':', linewidth=2, alpha=0.7)
    ax.axhline(avg_worst, color='#d62728', linestyle=':', linewidth=2, alpha=0.7)

    plt.tight_layout()
    plt.savefig('analysis_sequence_similarity_best_worst.png', dpi=300)
    print("Plot saved to analysis_sequence_similarity_best_worst.png")
    plt.show()

if __name__ == '__main__':
    main()