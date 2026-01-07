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
# TSV_PATH = './generation-results-dplm2-goonly-unseen-all/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut_nondup_preds_mf.tsv'
TSV_PATH = './generation-results-dplm2-goonly-alldata-dm-ca-me-scale-0.2-weight-headclloss-2.0_sn-pn-11wstep/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut_preds_mf.tsv'

# GO Ontology 文件
GO_OBO_PATH = 'go-basic.obo'

# 采样设置
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
        return None # 返回 None 表示无法计算

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
    go_graph = load_go_graph(GO_OBO_PATH)

    print(get_semantic_distance(go_graph, "GO:0005212", "GO:0042802"))

    exit()

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
    
    # def is_strict_unseen_combo(go_list):
    #     if not go_list: return False
    #     if tuple(sorted(go_list)) in train_combos_set: return False
    #     sets_to_intersect = [go_to_train_indices[go_id] for go_id in go_list if go_id in go_to_train_indices]
    #     if len(sets_to_intersect) != len(go_list): return True 
    #     if len(set.intersection(*sets_to_intersect)) > 0: return False 
    #     return True

    # for entry in test_data:
    #     pid = entry['uniprot_id']
    #     gt_go_ids = [index_to_go[i] for i in entry['go_f_mapped']]
        
    #     if is_strict_unseen_combo(gt_go_ids) and all(go in valid_train_gos for go in gt_go_ids):
    #         unseen_entries.append({'id': pid, 'gt': set(gt_go_ids)})

    for entry in test_data:
        unseen_entries.append({'id': entry['uniprot_id'], 'gt': set(index_to_go[i] for i in entry['go_f_mapped'])})

    print(f"Found {len(unseen_entries)} strict unseen targets.")




    # ================= 4. 逐个分析样本 (计算距离) =================
    analysis_results = []

    print("Analyzing distances & metrics...")
    for entry in tqdm(unseen_entries):
        pid = entry['id']
        gt_set = entry['gt']
        pred_set = pred_dict.get(pid, set())
        
        # --- Match Status ---
        intersection = gt_set.intersection(pred_set)
        if gt_set.issubset(pred_set):
            match_status = 'Exact Match'
        elif len(intersection) > 0:
            match_status = 'Partial Match'
        else:
            match_status = 'Fail'
        
        # === [核心修改] 计算集合内部平均距离 ===
        def calc_intra_set_dist(go_set):
            go_list = list(go_set)
            if len(go_list) < 2: return np.nan # 单标签没有距离，返回 NaN 以便忽略
            
            dists = []
            for i in range(len(go_list)):
                for j in range(i+1, len(go_list)):
                    d = get_semantic_distance(go_graph, go_list[i], go_list[j])
                    if d is not None and d < 100: dists.append(d)
            
            return np.mean(dists) if dists else np.nan

        gt_intra_dist = calc_intra_set_dist(gt_set)
        pred_intra_dist = calc_intra_set_dist(pred_set)
        
        # === [新增] 计算 GT 和 Pred 两个集合之间的平均距离 ===
        # (Average Linkage: 两个集合所有可能的两两配对的平均距离)
        set_to_set_dist = np.nan
        if len(gt_set) > 0 and len(pred_set) > 0:
            pair_dists = []
            for g in gt_set:
                for p in pred_set:
                    d = get_semantic_distance(go_graph, g, p)
                    if d is not None and d < 100: pair_dists.append(d)
            if pair_dists: set_to_set_dist = np.mean(pair_dists)

        # === H3: Frequency ===
        freqs = [train_go_counts[go] for go in gt_set]
        avg_freq = np.mean(freqs) if freqs else 0

        analysis_results.append({
            'match_status': match_status,
            'num_labels': len(gt_set),
            'num_pred_labels': len(pred_set),
            'GT_Intra_Dist': gt_intra_dist,       # 真值内部距离
            'Pred_Intra_Dist': pred_intra_dist,   # 预测值内部距离
            'GT_Pred_Inter_Dist': set_to_set_dist,# 真值与预测值之间的距离
            'Avg_Train_Freq': avg_freq
        })

    # ================= 5. 统计与输出 =================
    df_res = pd.DataFrame(analysis_results)
    
    print("\n" + "="*80)
    print("ANALYSIS REPORT: Semantic Distances in Unseen Generalization")
    print("="*80)
    
    # 定义要对比的指标
    # 注意：我们这里只统计有值的行 (dropna)
    metrics = {
        'GT_Intra_Dist': 'GT Internal Dist (Lower = Tighter Semantics)',
        'Pred_Intra_Dist': 'Pred Internal Dist (Is model hallucinating diverse terms?)',
        'GT_Pred_Inter_Dist': 'GT-Pred Set Distance (Lower = Semantically Close)',
        'Avg_Train_Freq': 'Avg Train Frequency',
        'num_labels': 'Avg Label Count'
    }
    
    print(f"{'Metric':<50} | {'Exact Match':<12} | {'Partial Match':<12} | {'Fail':<12}")
    print("-" * 100)
    
    for col, desc in metrics.items():
        # 分组计算均值 (自动忽略 NaN)
        m_exact = df_res[df_res['match_status'] == 'Exact Match'][col].mean()
        m_partial = df_res[df_res['match_status'] == 'Partial Match'][col].mean()
        m_fail = df_res[df_res['match_status'] == 'Fail'][col].mean()
        
        print(f"{desc:<50} | {m_exact:<12.4f} | {m_partial:<12.4f} | {m_fail:<12.4f}")
        
    print("-" * 100)
    print("注: Internal Dist 仅统计了标签数 >= 2 的样本。单标签样本已排除。")
    
    # 保存结果
    df_res.to_csv('analysis_semantic_distances.csv', index=False)
    print("\nDetailed CSV saved to 'analysis_semantic_distances.csv'")

if __name__ == '__main__':
    main()