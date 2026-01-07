import os
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from tqdm import tqdm
from itertools import combinations
from src.byprot.utils.ontology import Ontology

# ================= Configuration =================
TRAIN_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
TEST_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
GO_MAPPING_PATH = 'go_mapping.pkl'
GO_OBO_PATH = 'data/go.obo' 

# Models Config
MODELS = {
    'Ours': './generation-results-dplm2-goonly-alldata-dm-ca-me-scale-0.2-weight-headclloss-2.0_sn-pn-11wstep/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut_preds_mf.tsv',
    'Baseline': './generation-results-cfpgen_650m/cfpgen_650m_go_preds_mf.tsv', 
    'Reference': '/AIRvePFS/dair/chenxr-data/repo/cfpgen/data-bin/uniprotKB/cfpgen_general_dataset/test_preds_mf.tsv'
}

CONFIDENCE_THRESHOLD = 0.0

# ================= 2. Helper Functions =================
def load_predictions(tsv_path):
    print(f"Loading {tsv_path}...")
    try:
        df = pd.read_csv(tsv_path, sep='\t', header=None, names=['raw_id', 'go_id', 'score'])
    except:
        df = pd.read_csv(tsv_path, sep=r'\s+', header=None, names=['raw_id', 'go_id', 'score'])
    
    def get_clean_id(raw):
        if 'SEQUENCE_ID=' in raw:
            return raw.replace('SEQUENCE_ID=', '').split('_L=')[0]
        return raw 
    df['clean_id'] = df['raw_id'].apply(get_clean_id)
    df = df[df['score'] >= CONFIDENCE_THRESHOLD]
    
    # Build pid -> [set(go), set(go)] list (supports multiple sampling)
    pid_to_preds = defaultdict(list)
    for raw_id, group in df.groupby('raw_id'):
        pid = group['clean_id'].iloc[0]
        pid_to_preds[pid].append(set(group['go_id']))
    return pid_to_preds

def calculate_sample_metrics_official(gt_set, pred_set, ontology):
    """
    Calculate single sample metrics with Ancestor Propagation
    """
    # 1. Propagate
    gt_prop = set()
    for go in gt_set: gt_prop.update(ontology.get_ancestors(go))
    
    pred_prop = set()
    # Assuming pred_set is already raw IDs, propagate them
    # for go in pred_set: pred_prop.update(ontology.get_ancestors(go))
    pred_prop = pred_set
    
    # 2. Intersection Filter
    intersection = len(gt_prop & pred_prop)
    len_gt = len(gt_prop)
    len_pred = len(pred_prop)
    
    # F1
    if len_gt + len_pred == 0:
        f1 = 0.0 
    else:
        f1 = 2 * intersection / (len_gt + len_pred)
    
    # Recall
    recall = intersection / len_gt if len_gt > 0 else 0.0
    
    return f1, recall

def get_model_metrics_for_sample(pid, preds_map, gt_set, ontology):
    """Helper to get avg metrics for a specific model on a specific sample"""
    preds_list = preds_map.get(pid, [])
    if not preds_list:
        return 0.0, 0.0 # Missing prediction treated as 0
    
    f1s, recs = [], []
    for p_set in preds_list:
        f, r = calculate_sample_metrics_official(gt_set, p_set, ontology)
        f1s.append(f)
        recs.append(r)
    return np.mean(f1s), np.mean(recs)

# ================= 3. Main Logic =================
def main():
    ontology = Ontology(GO_OBO_PATH)
    
    # --- Load Data ---
    print("Loading Mapping & Data...")
    with open(GO_MAPPING_PATH, 'rb') as f:
        index_to_go = {v: k for k, v in pickle.load(f).items()}
    
    with open(TRAIN_PATH, 'rb') as f:
        train_data = pickle.load(f)
        
    with open(TEST_PATH, 'rb') as f:
        test_data = pickle.load(f)

    # --- 1. Build Training Stats (Frequency, IDF, Co-occurrence) ---
    print("Computing Training Stats...")
    train_counts = defaultdict(int)
    train_pair_counts = defaultdict(int)
    total_train = len(train_data)
    
    for entry in tqdm(train_data):
        gos = sorted(list(set([index_to_go[i] for i in entry['go_f_mapped']])))
        for go in gos: train_counts[go] += 1
        if len(gos) > 1:
            for pair in combinations(gos, 2):
                train_pair_counts[pair] += 1
                
    idf_dict = {go: np.log10(total_train / (c + 1)) for go, c in train_counts.items()}
    max_idf = np.log10(total_train)

    # --- 2. Pre-calculate Test Sample Attributes (X-axis variables) ---
    print("Calculating Test Sample Attributes...")
    sample_attrs = {} # pid -> dict
    
    # Pre-calc Depth
    all_test_gos = set()
    for entry in test_data:
        all_test_gos.update([index_to_go[i] for i in entry['go_f_mapped']])
    go_depth_map = {go: ontology.get_depth(go) for go in all_test_gos}

    for entry in tqdm(test_data):
        pid = entry['uniprot_id']
        gt_set = set([index_to_go[i] for i in entry['go_f_mapped']])
        if not gt_set: continue
        
        # A. Semantic Difficulty
        diff = ontology.calculate_set_difficulty(list(gt_set))
        
        # B. Average Depth
        avg_depth = np.mean([go_depth_map.get(go, 0) for go in gt_set])
        
        # C. Log Avg Frequency
        avg_freq = np.mean([train_counts.get(go, 0) for go in gt_set])
        log_freq = np.log10(avg_freq + 1)
        
        # D. Avg IDF
        avg_idf = np.mean([idf_dict.get(go, max_idf) for go in gt_set])
        
        # E. Co-occurrence Strength
        cooc_score = 0.0
        if len(gt_set) > 1:
            scores = []
            for pair in combinations(sorted(list(gt_set)), 2):
                pair_c = train_pair_counts.get(pair, 0)
                u_c = train_counts.get(pair[0], 0) + train_counts.get(pair[1], 0) - pair_c
                scores.append(pair_c / u_c if u_c > 0 else 0)
            cooc_score = np.mean(scores)
        
        sample_attrs[pid] = {
            'gt': gt_set,
            'sem_dist': diff,
            'depth': avg_depth,
            'log_freq': log_freq,
            'idf': avg_idf,
            'cooc': cooc_score,
            'num_labels': len(gt_set)
        }

    # --- 3. Load Predictions ---
    print("Loading Predictions...")
    ours_preds = load_predictions(MODELS['Ours'])
    base_preds = load_predictions(MODELS['Baseline'])
    ref_preds = load_predictions(MODELS['Reference'])

    # --- 4. Calculate Delta Metrics for BOTH models ---
    print("Calculating Delta Metrics...")
    analysis_data = []

    for pid, attrs in tqdm(sample_attrs.items()):
        gt = attrs['gt']
        
        # Get absolute metrics
        f1_ours, rec_ours = get_model_metrics_for_sample(pid, ours_preds, gt, ontology)
        f1_base, rec_base = get_model_metrics_for_sample(pid, base_preds, gt, ontology)
        f1_ref,  rec_ref  = get_model_metrics_for_sample(pid, ref_preds,  gt, ontology)
            
        # Calculate Deltas (Metric - Reference)
        # Ours vs Ref
        d_f1_ours = f1_ours - f1_ref
        d_rec_ours = rec_ours - rec_ref
        
        # Baseline vs Ref
        d_f1_base = f1_base - f1_ref
        d_rec_base = rec_base - rec_ref
        
        analysis_data.append({
            'pid': pid,
            'd_f1_ours': d_f1_ours,
            'd_rec_ours': d_rec_ours,
            'd_f1_base': d_f1_base,
            'd_rec_base': d_rec_base,
            **attrs
        })

    df = pd.DataFrame(analysis_data)
    df.to_csv('delta_metrics_comparison_raw.csv', index=False)
    
    # # ================= 5. Plotting Function (Comparison with Counts) =================
    # def plot_comparison_delta(df, x_col, x_label, filename, bins=15):
    #     # 创建画布和主轴 (ax1)
    #     fig, ax1 = plt.subplots(figsize=(12, 7))
        
    #     # Data Cleaning
    #     cols = [x_col, 'd_f1_ours', 'd_f1_base', 'd_rec_ours', 'd_rec_base']
    #     df_clean = df.dropna(subset=cols).copy()
        
    #     # Binning
    #     # 为了画柱状图，我们需要把 bin 变成离散的分类
    #     if df_clean[x_col].nunique() > bins:
    #         df_clean['bin'] = pd.cut(df_clean[x_col], bins=bins)
    #         # 聚合计算 mean, sem, 和 count (样本数)
    #         stats = df_clean.groupby('bin', observed=True)[['d_f1_ours', 'd_f1_base', 'd_rec_ours', 'd_rec_base']].agg(['mean', 'sem', 'count']).reset_index()
    #         # 获取 X 轴坐标 (用区间中点)
    #         stats['x'] = stats['bin'].apply(lambda x: x.mid).astype(float)
    #         # 获取 Count (用任一列的 count 即可，例如 d_f1_ours 的 count)
    #         counts = stats['d_f1_ours']['count']
    #     else:
    #         # 离散值直接聚合
    #         stats = df_clean.groupby(x_col)[['d_f1_ours', 'd_f1_base', 'd_rec_ours', 'd_rec_base']].agg(['mean', 'sem', 'count']).reset_index()
    #         stats['x'] = stats[x_col]
    #         counts = stats['d_f1_ours']['count']
        
    #     # --- 第二轴 (ax2): 绘制样本数量 (Bar Chart) ---
    #     ax2 = ax1.twinx() # 创建共享 X 轴的第二 Y 轴
        
    #     # 计算柱子的宽度 (根据数据范围动态调整，防止重叠)
    #     if len(stats['x']) > 1:
    #         width = (stats['x'].max() - stats['x'].min()) / len(stats['x']) * 0.8
    #     else:
    #         width = 0.1 # 默认值
            
    #     ax2.bar(stats['x'], counts, width=width, color='gray', alpha=0.15, label='Sample Count')
    #     ax2.set_ylabel('Number of Samples', color='gray', fontsize=12)
    #     ax2.tick_params(axis='y', labelcolor='gray')
        
    #     # --- 主轴 (ax1): 绘制 Delta 曲线 ---
    #     # Zero Line
    #     ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.6)

    #     # 1. Ours - Reference (Solid Lines)
    #     # Delta F1
    #     ln1 = ax1.plot(stats['x'], stats['d_f1_ours']['mean'], 
    #              marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='Ours $\Delta$ F1')
    #     ax1.fill_between(stats['x'], 
    #                      stats['d_f1_ours']['mean'] - stats['d_f1_ours']['sem'], 
    #                      stats['d_f1_ours']['mean'] + stats['d_f1_ours']['sem'], 
    #                      color='#1f77b4', alpha=0.15)
        
    #     # Delta Recall
    #     ln2 = ax1.plot(stats['x'], stats['d_rec_ours']['mean'], 
    #              marker='^', linestyle='-', color='#2ca02c', linewidth=2, label='Ours $\Delta$ Recall')
    #     ax1.fill_between(stats['x'], 
    #                      stats['d_rec_ours']['mean'] - stats['d_rec_ours']['sem'], 
    #                      stats['d_rec_ours']['mean'] + stats['d_rec_ours']['sem'], 
    #                      color='#2ca02c', alpha=0.15)

    #     # 2. Baseline - Reference (Dashed/Dotted Lines)
    #     # Delta F1
    #     ln3 = ax1.plot(stats['x'], stats['d_f1_base']['mean'], 
    #              marker='o', linestyle='--', color='#aec7e8', linewidth=2, label='Baseline $\Delta$ F1') 
        
    #     # Delta Recall
    #     ln4 = ax1.plot(stats['x'], stats['d_rec_base']['mean'], 
    #              marker='^', linestyle='--', color='#98df8a', linewidth=2, label='Baseline $\Delta$ Recall') 

    #     # 设置标签和标题
    #     ax1.set_title(f'Performance Gain over Reference: Ours vs Baseline\n(vs {x_label})', fontsize=14)
    #     ax1.set_xlabel(x_label, fontsize=12)
    #     ax1.set_ylabel('Delta Score (Model - Reference)', fontsize=12)
        
    #     # 合并图例 (Lines + Bar)
    #     lines, labels = ax1.get_legend_handles_labels()
    #     lines2, labels2 = ax2.get_legend_handles_labels()
    #     ax1.legend(lines + lines2, labels + labels2, loc='best', fontsize=10)
        
    #     ax1.grid(True, linestyle=':', alpha=0.6)
        
    #     # 把 ax1 放在 ax2 上面 (防止柱状图遮挡网格线)
    #     ax1.set_zorder(ax2.get_zorder() + 1)
    #     ax1.patch.set_visible(False)
        
    #     plt.tight_layout()
    #     plt.savefig(filename, dpi=300)
    #     print(f"Saved {filename}")
    #     plt.show()

# ================= 5. Plotting Function (Comparison with Counts & Filter) =================
    def plot_comparison_delta(df, x_col, x_label, filename, bins=15, min_samples=100):
        # 创建画布和主轴 (ax1)
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # Data Cleaning
        cols = [x_col, 'd_f1_ours', 'd_f1_base', 'd_rec_ours', 'd_rec_base']
        df_clean = df.dropna(subset=cols).copy()
        
        # Binning
        if df_clean[x_col].nunique() > bins:
            df_clean['bin'] = pd.cut(df_clean[x_col], bins=bins)
            # 聚合计算 mean, sem, 和 count (样本数)
            stats = df_clean.groupby('bin', observed=True)[['d_f1_ours', 'd_f1_base', 'd_rec_ours', 'd_rec_base']].agg(['mean', 'sem', 'count']).reset_index()
            
            # --- [新增] 过滤掉样本数不足的桶 ---
            stats = stats[stats['d_f1_ours']['count'] >= min_samples].copy()
            
            if stats.empty:
                print(f"Warning: All bins filtered out for {filename} (min_samples={min_samples})")
                plt.close()
                return

            # 获取 X 轴坐标 (用区间中点)
            stats['x'] = stats['bin'].apply(lambda x: x.mid).astype(float)
            # 获取 Count (用任一列的 count 即可)
            counts = stats['d_f1_ours']['count']
        else:
            # 离散值直接聚合
            stats = df_clean.groupby(x_col)[['d_f1_ours', 'd_f1_base', 'd_rec_ours', 'd_rec_base']].agg(['mean', 'sem', 'count']).reset_index()
            
            # --- [新增] 过滤掉样本数不足的桶 ---
            stats = stats[stats['d_f1_ours']['count'] >= min_samples].copy()
            
            if stats.empty:
                print(f"Warning: All groups filtered out for {filename} (min_samples={min_samples})")
                plt.close()
                return

            stats['x'] = stats[x_col]
            counts = stats['d_f1_ours']['count']
        
        # --- 第二轴 (ax2): 绘制样本数量 (Bar Chart) ---
        ax2 = ax1.twinx() # 创建共享 X 轴的第二 Y 轴
        
        # 计算柱子的宽度 (根据数据范围动态调整，防止重叠)
        if len(stats['x']) > 1:
            width = (stats['x'].max() - stats['x'].min()) / len(stats['x']) * 0.8
        else:
            width = 0.1 # 默认值
            
        ax2.bar(stats['x'], counts, width=width, color='gray', alpha=0.15, label='Sample Count')
        ax2.set_ylabel('Number of Samples', color='gray', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='gray')
        
        # --- 主轴 (ax1): 绘制 Delta 曲线 ---
        # Zero Line
        ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.6)

        # 1. Ours - Reference (Solid Lines)
        # Delta F1
        ln1 = ax1.plot(stats['x'], stats['d_f1_ours']['mean'], 
                 marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='Ours $\Delta$ F1')
        ax1.fill_between(stats['x'], 
                         stats['d_f1_ours']['mean'] - stats['d_f1_ours']['sem'], 
                         stats['d_f1_ours']['mean'] + stats['d_f1_ours']['sem'], 
                         color='#1f77b4', alpha=0.15)
        
        # Delta Recall
        ln2 = ax1.plot(stats['x'], stats['d_rec_ours']['mean'], 
                 marker='^', linestyle='-', color='#2ca02c', linewidth=2, label='Ours $\Delta$ Recall')
        ax1.fill_between(stats['x'], 
                         stats['d_rec_ours']['mean'] - stats['d_rec_ours']['sem'], 
                         stats['d_rec_ours']['mean'] + stats['d_rec_ours']['sem'], 
                         color='#2ca02c', alpha=0.15)

        # 2. Baseline - Reference (Dashed/Dotted Lines)
        # Delta F1
        ln3 = ax1.plot(stats['x'], stats['d_f1_base']['mean'], 
                 marker='o', linestyle='--', color='#aec7e8', linewidth=2, label='Baseline $\Delta$ F1') 
        
        # Delta Recall
        ln4 = ax1.plot(stats['x'], stats['d_rec_base']['mean'], 
                 marker='^', linestyle='--', color='#98df8a', linewidth=2, label='Baseline $\Delta$ Recall') 

        # 设置标签和标题
        ax1.set_title(f'Performance Gain over Reference: Ours vs Baseline\n(vs {x_label})', fontsize=14)
        ax1.set_xlabel(x_label, fontsize=12)
        ax1.set_ylabel('Delta Score (Model - Reference)', fontsize=12)
        
        # 合并图例 (Lines + Bar)
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='best', fontsize=10)
        
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        # 把 ax1 放在 ax2 上面 (防止柱状图遮挡网格线)
        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Saved {filename}")
        plt.show()
        
    # ================= 6. Generate All Plots =================
    
    # 1. Delta vs Semantic Distance
    plot_comparison_delta(df, 'sem_dist', 'Semantic Difficulty (Avg Distance)', 'comp_delta_vs_distance.png')
    
    # 2. Delta vs Depth
    plot_comparison_delta(df, 'depth', 'Annotation Specificity (Avg Depth)', 'comp_delta_vs_depth.png')
    
    # 3. Delta vs Frequency
    plot_comparison_delta(df, 'log_freq', 'Log10(Train Frequency)', 'comp_delta_vs_frequency.png')
    
    # 4. Delta vs IDF
    plot_comparison_delta(df, 'idf', 'Specificity (Avg IDF)', 'comp_delta_vs_idf.png')
    
    # 5. Delta vs Co-occurrence
    df_multi = df[df['num_labels'] > 1]
    plot_comparison_delta(df_multi, 'cooc', 'Co-occurrence Strength (Typicality)', 'comp_delta_vs_cooc.png')

if __name__ == '__main__':
    main()