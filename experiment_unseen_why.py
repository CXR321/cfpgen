import os
import pickle
import random
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
from tqdm import tqdm
from src.byprot.utils.ontology import Ontology
import matplotlib.pyplot as plt
import seaborn as sns

random.seed(42)

# ================= 配置 =================
TRAIN_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
TEST_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
GO_MAPPING_PATH = 'go_mapping.pkl'

# 预测结果 TSV (建议使用 nondup 版本)
# TSV_PATH = './generation-results-dplm2-goonly-unseen-all/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut_nondup_preds_mf.tsv'
# TSV_PATH = './generation-results-dplm2-goonly-alldata-dm-ca-me-scale-0.2-weight-headclloss-2.0_sn-pn-11wstep/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut_preds_mf.tsv'
TSV_PATH = './generation-results-dplm2-goonly-unseen-all/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut_nondup_preds_mf.tsv'

# GO Ontology 文件
# GO_OBO_PATH = 'go-basic.obo'
GO_OBO_PATH = 'data/go.obo' 

# 采样设置
CONFIDENCE_THRESHOLD = 0.0 

# ================= 1. GO Ontology 解析器 =================
# def load_go_graph(obo_path):
#     print(f"Loading GO Ontology from {obo_path}...")
#     if not os.path.exists(obo_path):
#         raise FileNotFoundError(f"请下载 go-basic.obo 文件并放置在 {obo_path}")
        
#     G = nx.Graph() # 无向图计算距离
#     with open(obo_path, 'r') as f:
#         current_id = ""
#         for line in f:
#             line = line.strip()
#             if line.startswith("id: GO:"):
#                 current_id = line.split("id: ")[1]
#                 G.add_node(current_id)
#             elif line.startswith("is_a:"):
#                 parent_id = line.split("is_a: ")[1].split(" ! ")[0]
#                 if current_id:
#                     G.add_edge(current_id, parent_id)
#             # elif line.startswith("relationship: part_of"):
#             #     parent_id = line.split("relationship: part_of ")[1].split(" ! ")[0]
#             #     if current_id:
#             #         G.add_edge(current_id, parent_id)
#     return G

# def get_semantic_distance(G, go1, go2):
#     if go1 == go2: return 0
#     try:
#         return nx.shortest_path_length(G, source=go1, target=go2)
#     except (nx.NetworkXNoPath, nx.NodeNotFound):
#         return None # 返回 None 表示无法计算

# ================= 2. 数据加载 =================
def load_data():
    with open(GO_MAPPING_PATH, 'rb') as f:
        go_mapping = pickle.load(f)
    index_to_go = {v: k for k, v in go_mapping.items()}
    
    with open(TRAIN_PATH, 'rb') as f:
        train_data = pickle.load(f)
        
    with open(TEST_PATH, 'rb') as f:
        test_data = pickle.load(f)
        
    return index_to_go, train_data, test_data

# ================= 3. 主分析逻辑 =================
def main():
    # --- Load Resources ---
    # go_graph = load_go_graph(GO_OBO_PATH)
    ontology = Ontology(GO_OBO_PATH)

    # print(get_semantic_distance(go_graph, "GO:0005212", "GO:0042802"))

    # exit()

    index_to_go, train_data, test_data = load_data()
    
    # --- Index Training Data ---
    print("Indexing Training Data...")
    go_to_train_indices = defaultdict(set)
    train_combos_set = set()
    valid_train_gos = set()
    train_go_counts = Counter()
    
    for idx, entry in enumerate(tqdm(train_data, desc="Indexing")):
        go_ids = [index_to_go[i] for i in entry['go_f_mapped']]
        train_combos_set.add(tuple(sorted(go_ids)))
        for go in go_ids:
            valid_train_gos.add(go)
            go_to_train_indices[go].add(idx)
            train_go_counts[go] += 1

    # --- Load Predictions ---
    # print(f"Loading Predictions from {TSV_PATH}...")
    # try:
    #     df = pd.read_csv(TSV_PATH, sep='\t', header=None, names=['raw_id', 'go_id', 'score'])
    # except:
    #     df = pd.read_csv(TSV_PATH, sep=r'\s+', header=None, names=['raw_id', 'go_id', 'score'])
    
    # df['clean_id'] = df['raw_id'].apply(lambda x: x.replace('SEQUENCE_ID=', '').split('_L=')[0])
    # df = df[df['score'] >= CONFIDENCE_THRESHOLD]
    # pred_dict = df.groupby('clean_id')['go_id'].apply(set).to_dict()

# --- Load Predictions ---
    print(f"Loading Predictions from {TSV_PATH}...")
    try:
        df = pd.read_csv(TSV_PATH, sep='\t', header=None, names=['raw_id', 'go_id', 'score'])
    except:
        df = pd.read_csv(TSV_PATH, sep=r'\s+', header=None, names=['raw_id', 'go_id', 'score'])
    
    # 过滤低置信度
    df = df[df['score'] >= CONFIDENCE_THRESHOLD]

    # 定义提取 clean_id 的函数 (保持你原来的逻辑)
    def get_clean_id(raw_str):
        return raw_str.replace('SEQUENCE_ID=', '').split('_L=')[0]

    # [关键修改]
    # 1. 先按 raw_id (也就是具体的某一次生成) 聚合 GO Terms
    #    假设 raw_id 能够区分那 10 条不同的生成结果
    raw_id_groups = df.groupby('raw_id')['go_id'].apply(set).to_dict()

    # 2. 将 raw_id 的结果归并到 clean_id 下，形成 List[Set] 结构
    #    结构: { 'P12345': [ {GO:1, GO:2}, {GO:1, GO:3}, ... ] }
    pred_dict = defaultdict(list)
    
    print("Grouping predictions by Clean ID...")
    for raw_id, go_set in raw_id_groups.items():
        clean_id = get_clean_id(raw_id)
        pred_dict[clean_id].append(go_set)

    # 打印一下检查数据是否正确
    sample_key = next(iter(pred_dict))
    print(f"Sample ID: {sample_key}, Count of generated samples: {len(pred_dict[sample_key])} (Expected ~10)")

    # --- Identify Strict Unseen Targets ---
    print("Identifying Strict Unseen targets...")
    unseen_entries = []
    
    def is_strict_unseen_combo(go_list):
        if not go_list: return False
        if tuple(sorted(go_list)) in train_combos_set: return False
        sets_to_intersect = [go_to_train_indices[go_id] for go_id in go_list if go_id in go_to_train_indices]
        if len(sets_to_intersect) != len(go_list): return True 
        if len(set.intersection(*sets_to_intersect)) > 0: return False 
        return True

    for entry in test_data:
        pid = entry['uniprot_id']
        gt_go_ids = [index_to_go[i] for i in entry['go_f_mapped']]
        
        if is_strict_unseen_combo(gt_go_ids) and all(go in valid_train_gos for go in gt_go_ids):
            unseen_entries.append({'id': pid, 'gt': set(gt_go_ids)})

    # for entry in test_data:
    #     unseen_entries.append({'id': entry['uniprot_id'], 'gt': set(index_to_go[i] for i in entry['go_f_mapped'])})

    print(f"Found {len(unseen_entries)} strict unseen targets.")

    # 1. 先收集所有符合条件的候选数据 (这一步逻辑不变，只是存入 candidates)
    candidates = unseen_entries
    # 2. 按 GT 分组并进行随机下采样
    gt_groups = defaultdict(list)
    for item in candidates:
        # set 是不可哈希的，必须转为 frozenset 才能作为字典的 Key
        gt_key = frozenset(item['gt'])
        gt_groups[gt_key].append(item)

    unseen_entries = []
    max_per_gt = 1  # 设定的阈值

    for gt_key, items in gt_groups.items():
        if len(items) > max_per_gt:
            # 如果超过10条，随机抽取10条
            unseen_entries.extend(random.sample(items, max_per_gt))
        else:
            # 不足或刚好10条，全部保留
            unseen_entries.extend(items)

    print(f"Filtered down to {len(unseen_entries)} targets after capping same GT at {max_per_gt}.")


    # ================= 4. 逐个分析样本 (计算距离) =================
    analysis_results = []

    print("Analyzing distances & metrics...")
    # for entry in tqdm(unseen_entries):
    #     pid = entry['id']
    #     gt_set = entry['gt']
    #     pred_set = pred_dict.get(pid, set())
        
    #     # --- Match Status ---
    #     intersection = gt_set.intersection(pred_set)
    #     if gt_set.issubset(pred_set):
    #         match_status = 'Exact Match'
    #     elif len(intersection) > 0:
    #         match_status = 'Partial Match'
    #     else:
    #         match_status = 'Fail'
        
    #     # === [核心修改] 计算集合内部平均距离 ===
    #     def calc_intra_set_dist(go_set):
    #         go_list = list(go_set)
    #         if len(go_list) < 2: return np.nan # 单标签没有距离，返回 NaN 以便忽略
            
    #         dists = []
    #         for i in range(len(go_list)):
    #             for j in range(i+1, len(go_list)):
    #                 d = get_semantic_distance(go_graph, go_list[i], go_list[j])
    #                 if d is not None and d < 100: dists.append(d)
            
    #         return np.mean(dists) if dists else np.nan

    #     gt_intra_dist = calc_intra_set_dist(gt_set)
    #     pred_intra_dist = calc_intra_set_dist(pred_set)
        
    #     # === [新增] 计算 GT 和 Pred 两个集合之间的平均距离 ===
    #     # (Average Linkage: 两个集合所有可能的两两配对的平均距离)
    #     set_to_set_dist = np.nan
    #     if len(gt_set) > 0 and len(pred_set) > 0:
    #         pair_dists = []
    #         for g in gt_set:
    #             for p in pred_set:
    #                 d = get_semantic_distance(go_graph, g, p)
    #                 if d is not None and d < 100: pair_dists.append(d)
    #         if pair_dists: set_to_set_dist = np.mean(pair_dists)

    #     # === H3: Frequency ===
    #     freqs = [train_go_counts[go] for go in gt_set]
    #     avg_freq = np.mean(freqs) if freqs else 0

    #     analysis_results.append({
    #         'match_status': match_status,
    #         'num_labels': len(gt_set),
    #         'num_pred_labels': len(pred_set),
    #         'GT_Intra_Dist': gt_intra_dist,       # 真值内部距离
    #         'Pred_Intra_Dist': pred_intra_dist,   # 预测值内部距离
    #         'GT_Pred_Inter_Dist': set_to_set_dist,# 真值与预测值之间的距离
    #         'Avg_Train_Freq': avg_freq
    #     })
# 辅助函数：计算集合内部距离 (保持不变)
    def calc_intra_set_dist(go_set):
        go_list = list(go_set)
        if len(go_list) < 2: return np.nan
        dists = []
        for i in range(len(go_list)):
            for j in range(i+1, len(go_list)):
                # d = get_semantic_distance(go_graph, go_list[i], go_list[j])
                d = ontology.get_semantic_distance(go_list[i], go_list[j])
                if d is not None and d < 100: dists.append(d)
        return np.mean(dists) if dists else np.nan

    for entry in tqdm(unseen_entries):
        pid = entry['id']
        gt_set = entry['gt']
        
        # [关键修改] 获取该 ID 对应的所有预测结果列表 (通常是 10 个 set)
        pred_sets_list = pred_dict.get(pid, [])
        
        # 如果没有预测结果 (可能在这个 batch 没生成)，跳过或记录 Fail
        if not pred_sets_list:
            continue # 或者你可以记录一条全空的记录

        # [新增循环] 遍历这 10 个生成结果
        for sample_idx, pred_set in enumerate(pred_sets_list):
            
            # --- Match Status ---
            intersection = gt_set.intersection(pred_set)
            if len(pred_set) == 0:
                match_status = 'Empty Pred' # 预测为空
            elif gt_set.issubset(pred_set):
                match_status = 'Exact Match'
            elif len(intersection) > 0:
                match_status = 'Partial Match'
            else:
                match_status = 'Fail'
            
            # 1. GT 内部距离 (对于同一个 GT，这 10 次计算是一样的，但为了对齐 DataFrame 方便直接算)
            gt_intra_dist = calc_intra_set_dist(gt_set)
            
            # 2. Pred 内部距离
            pred_intra_dist = calc_intra_set_dist(pred_set)
            
            # 3. GT 和 Pred 之间的平均距离 (Set-to-Set)
            set_to_set_dist = np.nan
            if len(gt_set) > 0 and len(pred_set) > 0:
                pair_dists = []
                for g in gt_set:
                    for p in pred_set:
                        # d = get_semantic_distance(go_graph, g, p)
                        d = ontology.get_semantic_distance(g, p)
                        if d is not None and d < 100: pair_dists.append(d)
                if pair_dists: set_to_set_dist = np.mean(pair_dists)

            depths = []
            for go in gt_set:
                depths.append(ontology.get_depth(go))
            avg_depth = np.mean(depths) if depths else 0

            # 4. Frequency
            freqs = [train_go_counts[go] for go in gt_set]
            avg_freq = np.mean(freqs) if freqs else 0

            analysis_results.append({
                'id': pid,                    # 记录 ID
                'sample_idx': sample_idx,     # [新增] 记录这是第几个样本 (0-9)
                'match_status': match_status,
                'num_labels': len(gt_set),
                'num_pred_labels': len(pred_set),
                'GT_Intra_Dist': gt_intra_dist,
                'Pred_Intra_Dist': pred_intra_dist,
                'GT_Pred_Inter_Dist': set_to_set_dist,
                'Avg_Train_Freq': avg_freq,
                'Avg_Depth': avg_depth,
            })


# ================= 5. 统计与输出 =================
    from scipy import stats  # 需要引入 scipy 进行统计检验

    df_res = pd.DataFrame(analysis_results)
    
    print("\n" + "="*100)
    print("ANALYSIS REPORT: Semantic Distances & Statistical Significance")
    print("="*100)
    
    # 定义要对比的指标
    metrics = {
        'GT_Intra_Dist': 'GT Internal Dist (Lower = Tighter Semantics)',
        'Pred_Intra_Dist': 'Pred Internal Dist (Is model hallucinating?)',
        'GT_Pred_Inter_Dist': 'GT-Pred Set Distance (Lower = Closer)',
        'Avg_Train_Freq': 'Avg Train Frequency',
        'num_labels': 'Avg Label Count',
        'Avg_Depth': 'Avg Depth',
    }
    
    # 定义组别名称
    groups = ['Exact Match', 'Partial Match', 'Fail']

    # 打印表头
    # 格式：指标 | 组1均值 | 组2均值 | 组3均值 | 整体P值 | 显著性标记
    header = f"{'Metric':<40} | {'Exact':<8} | {'Partial':<8} | {'Fail':<8} | {'Kruskal-P':<10} | {'Pairwise Sig'}"
    print(header)
    print("-" * 110)
    
    significance_results = []

    for col, desc in metrics.items():
        # 1. 提取各组数据 (自动去除 NaN，这对于距离计算很重要)
        data_exact = df_res[df_res['match_status'] == 'Exact Match'][col].dropna()
        data_partial = df_res[df_res['match_status'] == 'Partial Match'][col].dropna()
        data_fail = df_res[df_res['match_status'] == 'Fail'][col].dropna()
        
        # 计算均值用于展示
        m_exact = data_exact.mean() if len(data_exact) > 0 else 0
        m_partial = data_partial.mean() if len(data_partial) > 0 else 0
        m_fail = data_fail.mean() if len(data_fail) > 0 else 0
        
        # 2. 整体差异检验 (Kruskal-Wallis H Test)
        # 非参数检验，不要求正态分布
        try:
            stat, p_global = stats.kruskal(data_exact, data_partial, data_fail)
        except ValueError:
            p_global = 1.0 # 数据不足时
        
        # 3. 两两差异检验 (Mann-Whitney U Test)
        # 只有当整体差异显著时，看两两差异才有意义，但为了展示我们都算出来
        def get_p_val(d1, d2):
            if len(d1) < 2 or len(d2) < 2: return 1.0
            try:
                _, p = stats.mannwhitneyu(d1, d2, alternative='two-sided')
                return p
            except: return 1.0

        p_ep = get_p_val(data_exact, data_partial) # Exact vs Partial
        p_pf = get_p_val(data_partial, data_fail)  # Partial vs Fail
        p_ef = get_p_val(data_exact, data_fail)    # Exact vs Fail
        
        # 4. 格式化输出
        # 将 P 值转为易读格式 (<0.001, <0.01, <0.05)
        def fmt_p(p):
            if p < 0.001: return "***"
            elif p < 0.01: return "**"
            elif p < 0.05: return "*"
            else: return "ns" # not significant
        
        pairwise_str = f"E-P:{fmt_p(p_ep)} P-F:{fmt_p(p_pf)} E-F:{fmt_p(p_ef)}"
        
        # 打印行
        print(f"{desc:<40} | {m_exact:<8.2f} | {m_partial:<8.2f} | {m_fail:<8.2f} | {p_global:<10.2e} | {pairwise_str}")
        
        # 保存详细统计结果
        significance_results.append({
            'metric': col,
            'mean_exact': m_exact,
            'mean_partial': m_partial,
            'mean_fail': m_fail,
            'p_kruskal_global': p_global,
            'p_exact_partial': p_ep,
            'p_partial_fail': p_pf,
            'p_exact_fail': p_ef
        })

    print("-" * 110)
    print("Legend: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("Tests: Global = Kruskal-Wallis; Pairwise = Mann-Whitney U")
    print("Pairwise Key: E-P (Exact vs Partial), P-F (Partial vs Fail), E-F (Exact vs Fail)")
    
    # 保存结果
    df_res.to_csv('analysis_semantic_distances_raw.csv', index=False)
    pd.DataFrame(significance_results).to_csv('analysis_statistical_significance.csv', index=False)
    print("\nSaved raw data to 'analysis_semantic_distances_raw.csv'")
    print("Saved statistical tests to 'analysis_statistical_significance.csv'")



## ================= 6. 可视化：分布小提琴图 (Violin Plots) =================
    print("\nGenerating Distribution Violin Plots...")
    
    text_scale = 1.35
    sns.set_theme(style="whitegrid", context="paper", font_scale=2 * text_scale)
    plt.rcParams['font.family'] = 'serif'
    
    # 颜色方案
    custom_palette = ["#4c72b0", "#dd8452", "#c44e52"] 
    x_order = ['Exact Match', 'Partial Match', 'Fail']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8)) 
    
    plot_configs = [
        ('GT_Intra_Dist', 'GT Intra-Set Distance', 'Semantic Distance'),
        ('Pred_Intra_Dist', 'Pred Intra-Set Distance', 'Semantic Distance'),
        ('Avg_Depth', 'Mean GO Term Depth', 'Hierarchical Depth')
    ]

    for ax, (col, title, ylabel) in zip(axes, plot_configs):
        plot_data = df_res.dropna(subset=[col])
        
        sns.violinplot(
            data=plot_data, 
            x='match_status', 
            y=col, 
            ax=ax, 
            order=x_order, 
            palette=custom_palette, 
            linewidth=1.0,
            cut=0 
        )

        if col == 'GT_Intra_Dist':
            ax.set_ylim(0, 10)
        elif col == 'Pred_Intra_Dist':
            ax.set_ylim(1, 6)
        elif col == 'Avg_Depth':
            ymin = float(np.floor(plot_data[col].min()))
            ymax = float(np.ceil(plot_data[col].max()))
            if ymin == ymax:
                ax.set_ylim(ymin - 1, ymax + 1)
                ax.set_yticks([ymin - 1, ymin, ymax, ymax + 1])
            else:
                step = 1 if (ymax - ymin) <= 10 else 2
                ax.set_ylim(ymin, ymax)
                ax.set_yticks(np.arange(ymin, ymax + step, step))

        ax.set_title(title, fontsize=int(round(20 * text_scale)), weight='bold', pad=int(round(15 * text_scale)))
        ax.set_xlabel('') 
        ax.set_ylabel(ylabel, fontsize=int(round(20 * text_scale)))
        ax.tick_params(axis='both', labelsize=int(round(18 * text_scale)))
        ax.set_xticklabels(['Exact\nMatch', 'Partial\nMatch', 'Fail'])

    plt.tight_layout()
    
    save_path = 'analysis_violin_distributions_bold_large.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Violin plots saved to '{save_path}'")
    
    plt.show()

# (End of main function)

if __name__ == '__main__':
    main()
