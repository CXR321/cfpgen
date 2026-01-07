import os
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
from tqdm import tqdm

# ================= 配置 =================
TRAIN_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
TEST_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
GO_MAPPING_PATH = 'go_mapping.pkl'

# 预测结果 TSV (建议使用 nondup 版本)
TSV_PATH = './generation-results-dplm2-goonly-alldata-dm-ca-me-scale-0.2-weight-headclloss-2.0_sn-pn-11wstep/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut_preds_mf.tsv'
# GO Ontology 文件
GO_OBO_PATH = 'go-basic.obo'

# 采样设置 (用于加速H2计算)
TRAIN_SAMPLE_SIZE = 50000 
CONFIDENCE_THRESHOLD = 0.0 

# ================= 1. GO Ontology 解析器 =================
def load_go_graph(obo_path):
    print(f"Loading GO Ontology from {obo_path}...")
    if not os.path.exists(obo_path):
        raise FileNotFoundError(f"请下载 go-basic.obo 文件并放置在 {obo_path}")
        
    G = nx.Graph() # 无向图计算距离
    with open(obo_path, 'r') as f:
        current_id = ""
        for line in f:
            line = line.strip()
            if line.startswith("id: GO:"):
                current_id = line.split("id: ")[1]
                G.add_node(current_id)
            elif line.startswith("is_a:"):
                parent_id = line.split("is_a: ")[1].split(" ! ")[0]
                if current_id:
                    G.add_edge(current_id, parent_id)
            elif line.startswith("relationship: part_of"):
                parent_id = line.split("relationship: part_of ")[1].split(" ! ")[0]
                if current_id:
                    G.add_edge(current_id, parent_id)
    return G

def get_semantic_distance(G, go1, go2):
    if go1 == go2: return 0
    try:
        return nx.shortest_path_length(G, source=go1, target=go2)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return 999 

# ================= 2. 数据加载与预处理 =================
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
    go_graph = load_go_graph(GO_OBO_PATH)
    index_to_go, train_data, test_data = load_data()
    
    # --- Analyze Training Data ---
    print("Analyzing Training Data Stats...")
    train_go_counts = Counter()
    train_go_cooccurence = defaultdict(Counter)
    
    go_to_train_indices = defaultdict(set)
    train_combos_set = set()
    valid_train_gos = set()
    
    for idx, entry in enumerate(tqdm(train_data, desc="Indexing Train")):
        go_ids = [index_to_go[i] for i in entry['go_f_mapped']]
        train_combos_set.add(tuple(sorted(go_ids)))
        
        for go in go_ids:
            train_go_counts[go] += 1
            valid_train_gos.add(go)
            go_to_train_indices[go].add(idx)
            for other_go in go_ids:
                if go != other_go:
                    train_go_cooccurence[go][other_go] += 1

    # --- Load Predictions ---
    print(f"Loading Predictions from {TSV_PATH}...")
    try:
        df = pd.read_csv(TSV_PATH, sep='\t', header=None, names=['raw_id', 'go_id', 'score'])
    except:
        df = pd.read_csv(TSV_PATH, sep=r'\s+', header=None, names=['raw_id', 'go_id', 'score'])
    
    df['clean_id'] = df['raw_id'].apply(lambda x: x.replace('SEQUENCE_ID=', '').split('_L=')[0])
    df = df[df['score'] >= CONFIDENCE_THRESHOLD]
    pred_dict = df.groupby('clean_id')['go_id'].apply(set).to_dict()

    # --- Identify Strict Unseen Targets ---
    print("Identifying Strict Unseen targets...")
    unseen_entries = []
    
    # Helper: Check strictly unseen
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
        
        # 筛选：组合Unseen 且 原子Known
        if is_strict_unseen_combo(gt_go_ids) and all(go in valid_train_gos for go in gt_go_ids):
            unseen_entries.append({
                'id': pid,
                'gt': set(gt_go_ids)
            })
            
    print(f"Found {len(unseen_entries)} strict unseen targets.")

    # ================= 4. 逐个分析样本 =================
    analysis_results = []

    print("Analyzing metrics for each target...")
    for entry in tqdm(unseen_entries):
        pid = entry['id']
        gt_set = entry['gt']
        pred_set = pred_dict.get(pid, set())
        
        # --- 核心修改：区分 Match Status ---
        intersection = gt_set.intersection(pred_set)
        
        if gt_set.issubset(pred_set):
            match_status = 'Exact Match' # 完美匹配
        elif len(intersection) > 0:
            match_status = 'Partial Match' # 部分匹配
        else:
            match_status = 'Fail' # 失败
        
        # --- H1: 内部语义距离 ---
        dists = []
        gt_list = list(gt_set)
        if len(gt_list) > 1:
            for i in range(len(gt_list)):
                for j in range(i+1, len(gt_list)):
                    d = get_semantic_distance(go_graph, gt_list[i], gt_list[j])
                    if d < 100: dists.append(d)
        avg_internal_dist = np.mean(dists) if dists else 0
        
        # --- H2: 近似距离 (简化版: 上下文代理距离) ---
        # 寻找训练集里是否存在这样的结构：A和B在目标里共现。训练集里有(A, C)，且B和C很近。
        proxy_distances = []
        if len(gt_list) > 1:
            # 随机抽样计算，防止卡死
            check_indices = []
            for go_item in gt_list:
                check_indices.extend(list(go_to_train_indices.get(go_item, set()))[:20]) # 每个词只看20个样本
            
            for idx in set(check_indices):
                train_sample_gos = [index_to_go[x] for x in train_data[idx]['go_f_mapped']]
                # 检查该训练样本对 GT 集合的覆盖程度或近似程度
                # 简化计算：计算 GT 中每个词到该训练样本中最近词的距离，然后取平均
                sample_dists = []
                for gt_go in gt_list:
                    min_d = 999
                    for train_go in train_sample_gos:
                        d = get_semantic_distance(go_graph, gt_go, train_go)
                        if d < min_d: min_d = d
                    sample_dists.append(min_d)
                
                # 如果这个训练样本跟目标很像（平均距离小），记录下来
                avg_d = np.mean(sample_dists)
                if avg_d < 100:
                    proxy_distances.append(avg_d)
                    
        # 取所有训练样本中最小的那个“集合距离”，代表最相似的训练样本距离
        min_proxy_dist = np.min(proxy_distances) if proxy_distances else 999

        # --- H3: 频率 ---
        freqs = [train_go_counts[go] for go in gt_set]
        avg_freq = np.mean(freqs) if freqs else 0
        
        # --- H4: 共现倾向 ---
        cooc_counts = []
        for go in gt_set:
            partners = train_go_cooccurence[go]
            total_partners = sum(partners.values())
            total_occurrences = train_go_counts[go]
            avg_partners = total_partners / total_occurrences if total_occurrences > 0 else 0
            cooc_counts.append(avg_partners)
        avg_cooc_size = np.mean(cooc_counts) if cooc_counts else 0

        # --- H5: 错误类型 ---
        fps = pred_set - gt_set
        fns = gt_set - pred_set
        min_fp_fn_dist = None
        if len(fps) > 0 and len(fns) > 0:
            dists_fp_fn = []
            # for fp in fps:
            #     for fn in fns:
            for fn in fns:
                temp_dis = []
                for fp in fps:                    
                    d = get_semantic_distance(go_graph, fp, fn)
                    temp_dis.append(d)
                dists_fp_fn.append(min(temp_dis))
            if dists_fp_fn: min_fp_fn_dist = np.mean(dists_fp_fn)
        
        analysis_results.append({
            'pid': pid,
            'match_status': match_status, # Exact / Partial / Fail
            'num_labels': len(gt_set),
            'H1_internal_dist': avg_internal_dist,
            'H2_nearest_train_dist': min_proxy_dist,
            'H3_avg_freq': avg_freq,
            'H4_avg_partner_size': avg_cooc_size,
            'min_fp_fn_dist': min_fp_fn_dist
        })

    # ================= 5. 统计与输出 =================
    df_res = pd.DataFrame(analysis_results)
    
    print("\n" + "="*80)
    print("ANALYSIS REPORT: Strict Unseen Generalization Factors")
    print("="*80)
    
    # 统计数量
    status_counts = df_res['match_status'].value_counts()
    print("Distribution of Outcomes:")
    print(status_counts)
    print(f"Exact Match Rate:   {len(df_res[df_res['match_status']=='Exact Match']) / len(df_res) * 100:.2f}%")
    print(f"Partial Match Rate: {len(df_res[df_res['match_status']=='Partial Match']) / len(df_res) * 100:.2f}%")
    
    # 定义要对比的指标
    cols_map = {
        'H1_internal_dist': 'H1: Internal Dist (Smaller=Cohesive)',
        'H2_nearest_train_dist': 'H2: Nearest Train Dist (Smaller=Similar)',
        'H3_avg_freq': 'H3: Train Frequency (Larger=Common)',
        'H4_avg_partner_size': 'H4: Avg Partners (Larger=Social)',
        'num_labels': 'Label Count'
    }
    
    print("\n" + "-"*100)
    print(f"{'Feature':<40} | {'Exact Match':<12} | {'Partial Match':<12} | {'Fail':<12}")
    print("-" * 100)
    
    for col, desc in cols_map.items():
        # 计算三组的均值
        mean_exact = df_res[df_res['match_status'] == 'Exact Match'][col].mean()
        mean_partial = df_res[df_res['match_status'] == 'Partial Match'][col].mean()
        mean_fail = df_res[df_res['match_status'] == 'Fail'][col].mean()
        
        print(f"{desc:<40} | {mean_exact:<12.4f} | {mean_partial:<12.4f} | {mean_fail:<12.4f}")
        
    print("-" * 100)
    
    # H5 单独统计 (只针对 Partial 组，因为 Exact 没有 FP/FN，Fail 可能也没有 FP)
    print("\n[H5] Semantic Error Analysis (Partial Match Only):")
    partial_group = df_res[df_res['match_status'] == 'Partial Match']
    if not partial_group.empty:
        semantic_errors = partial_group[partial_group['min_fp_fn_dist'] <= 2].shape[0]
        total = partial_group.shape[0]
        print(f"Partial matches with semantic close errors (Dist<=2): {semantic_errors}/{total} ({semantic_errors/total*100:.1f}%)")
        print("  -> 这意味着在“部分匹配”的案例中，有多少是预测了语义非常接近的词。")

    # H5 单独统计 (只针对 fail 组)
    print("\n[H5] Semantic Error Analysis (Fail Only):")
    partial_group = df_res[df_res['match_status'] == 'Fail']
    if not partial_group.empty:
        semantic_errors = partial_group[partial_group['min_fp_fn_dist'] <= 2].shape[0]
        total = partial_group.shape[0]
        print(f"Fail with semantic close errors (Dist<=2): {semantic_errors}/{total} ({semantic_errors/total*100:.1f}%)")
        print("  -> 这意味着在“Fail”的案例中，有多少是预测了语义非常接近的词。")

    df_res.to_csv('analysis_unseen_split_results.csv', index=False)
    print("\nSaved detailed results to 'analysis_unseen_split_results.csv'")

if __name__ == '__main__':
    main()