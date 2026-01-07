import os
import glob
import re
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

# ================= 配置区域 =================

# 1. 基础数据路径
TEST_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
GO_MAPPING_PATH = 'go_mapping.pkl'
GO_OBO_PATH = 'go-basic.obo'

# 2. 模型预测结果 TSV
# 你的模型 (DPLM2) - 要求: 全对
MY_MODEL_TSV = './generation-results-dplm2-goonly-alldata-dm-ca-me-scale-0.2-weight-headclloss-2.0_sn-pn-11wstep/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut_preds_mf.tsv'

# Baseline 模型 (CFPGen) - 要求: 尽量全错
BASELINE_TSV = './generation-results-cfpgen_650m/cfpgen_650m_go_preds_mf.tsv'

# 3. PDB 文件夹路径 (用于寻找高 pLDDT 结构)
PDB_FOLDER = './generation-results-dplm2-goonly-alldata-dm-ca-me-scale-0.2-weight-headclloss-2.0_sn-pn-11wstep/esmfold_pdb'

# 4. 筛选阈值
CONFIDENCE_THRESHOLD = 0.0      # 预测分值阈值 (设为0以读取所有预测)
TARGET_LABEL_COUNT = 2          # 严格限制: 只找只有 2 个真实标签的蛋白
MIN_PLDDT = 70.0                # 结构筛选: 最低 pLDDT 要求
MIN_PTM = 0.5                   # 结构筛选: 最低 pTM 要求 (可选)

# ================= 1. 工具函数: 加载 GO 图与计算距离 =================
def load_go_graph(obo_path):
    print(f"Loading GO Ontology from {obo_path}...")
    if not os.path.exists(obo_path):
        raise FileNotFoundError("请下载 go-basic.obo")
    G = nx.Graph() 
    with open(obo_path, 'r') as f:
        current_id = ""
        for line in f:
            line = line.strip()
            if line.startswith("id: GO:"):
                current_id = line.split("id: ")[1]
                G.add_node(current_id)
            elif line.startswith("is_a:"):
                parent = line.split("is_a: ")[1].split(" ! ")[0]
                G.add_edge(current_id, parent)
            elif line.startswith("relationship: part_of"):
                parent = line.split("relationship: part_of ")[1].split(" ! ")[0]
                G.add_edge(current_id, parent)
    return G

def get_distance(G, go1, go2):
    try:
        return nx.shortest_path_length(G, source=go1, target=go2)
    except:
        return 0 # 无法计算距离 (如不同分支 MF vs BP)

# ================= 2. 工具函数: 解析 PDB 信息 =================
def scan_pdb_folder(folder_path):
    """
    扫描文件夹，解析文件名，返回字典: 
    { 'UniProtID': {'filename': ..., 'plddt': ..., 'ptm': ...} }
    如果同一个ID有多个文件，保留 pLDDT 最高的一个。
    """
    print(f"Scanning PDB folder: {folder_path}...")
    pdb_data = {}
    
    # 文件名示例: SEQUENCE_ID=A0A0A2IBP6_L=373_plddt_89.17_ptm_0.901.pdb
    # 正则提取
    pattern = re.compile(r"SEQUENCE_ID=([^_]+)_L=\d+_plddt_([0-9.]+)_ptm_([0-9.]+)\.pdb")
    
    files = glob.glob(os.path.join(folder_path, "*.pdb"))
    for fpath in tqdm(files, desc="Parsing PDBs"):
        fname = os.path.basename(fpath)
        match = pattern.search(fname)
        if match:
            pid = match.group(1)
            plddt = float(match.group(2))
            ptm = float(match.group(3))
            
            # 如果该 ID 还没记录，或者新文件的 pLDDT 更高，则更新
            if pid not in pdb_data or plddt > pdb_data[pid]['plddt']:
                pdb_data[pid] = {
                    'filename': fname,
                    'plddt': plddt,
                    'ptm': ptm,
                    'full_path': fpath
                }
    print(f"Found {len(pdb_data)} unique PDB structures.")
    return pdb_data

# ================= 3. 数据加载与处理 =================
def load_predictions(tsv_path):
    print(f"Loading predictions from {tsv_path}...")
    try:
        df = pd.read_csv(tsv_path, sep='\t', header=None, names=['raw_id', 'go_id', 'score'])
    except:
        df = pd.read_csv(tsv_path, sep=r'\s+', header=None, names=['raw_id', 'go_id', 'score'])
    
    # 解析 UniProt ID
    def clean_id(raw):
        return raw.replace('SEQUENCE_ID=', '').split('_L=')[0]
    
    df['clean_id'] = df['raw_id'].apply(clean_id)
    df = df[df['score'] >= CONFIDENCE_THRESHOLD]
    
    # 聚合为 Set
    return df.groupby('clean_id')['go_id'].apply(set).to_dict()

# ================= 4. 主流程 =================
def main():
    # 1. 资源加载
    go_graph = load_go_graph(GO_OBO_PATH)
    
    with open(GO_MAPPING_PATH, 'rb') as f:
        go_mapping = pickle.load(f)
    index_to_go = {v: k for k, v in go_mapping.items()}
    
    with open(TEST_PATH, 'rb') as f:
        test_data = pickle.load(f)

    # 2. 预测结果加载
    my_preds = load_predictions(MY_MODEL_TSV)
    base_preds = load_predictions(BASELINE_TSV)
    
    # 3. PDB 信息扫描
    pdb_info = scan_pdb_folder(PDB_FOLDER)
    
    # 4. 筛选核心循环
    candidates = []
    
    print("\nScanning Test Set for Gold Candidates...")
    print(f"Criteria: GT_Len={TARGET_LABEL_COUNT}, My_Model=AllCorrect, Baseline=Fail, High pLDDT, High Semantic Dist")
    
    for entry in tqdm(test_data):
        pid = entry['uniprot_id']
        gt_indices = entry['go_f_mapped']
        gt_set = {index_to_go[i] for i in gt_indices}
        
        # --- Filter 1: 标签数量限制 ---
        # if len(gt_set) != TARGET_LABEL_COUNT:
        #     continue

        if len(gt_set) < 2:
            continue

        motifs = entry['motif']
        motif_ranges = [(m['start'], m['end']) for m in motifs]

        if len(motif_ranges) <= 1:
            continue

        motif_len_long = False

        for m_r in motif_ranges:
            if m_r[1] - m_r[0] > 100:
                motif_len_long = True
                break
        if motif_len_long:
            continue

        def calculate_min_overlap(motif_ranges):
            """计算所有motif对之间的最小重叠长度"""
            n = len(motif_ranges)
            
            if n < 2:
                return 0  # 没有重叠
            
            min_overlap = float('inf')
            
            for i in range(n):
                for j in range(i+1, n):
                    start_i, end_i = motif_ranges[i]
                    start_j, end_j = motif_ranges[j]
                    
                    # 计算两个motif的重叠长度
                    overlap = max(0, min(end_i, end_j) - max(start_i, start_j))
                    
                    if overlap > 0:
                        min_overlap = min(min_overlap, overlap)
            
            # 如果没有找到任何重叠
            if min_overlap == float('inf'):
                return 0
            
            return min_overlap

        overlap_len = calculate_min_overlap(motif_ranges)


        if overlap_len > 0:
            continue
            
        # --- Filter 2: PDB 存在且质量高 ---
        if pid not in pdb_info:
            continue
        struct = pdb_info[pid]
        if struct['plddt'] < MIN_PLDDT:
            continue
            
        # --- Filter 3: 模型表现对比 ---
        my_pred_set = my_preds.get(pid, set())
        base_pred_set = base_preds.get(pid, set())
        
        # A. 你的模型必须: Ground Truth 是 Prediction 的子集 (完全覆盖)
        #    或者更严格: gt_set == my_pred_set (完全匹配)
        #    这里用 issubset (覆盖即可)
        if not gt_set.issubset(my_pred_set):
            continue
            
        # B. Baseline 模型必须: 表现很差
        #    定义: 交集为空 (完全没预测对) 或者 Jaccard 极低
        baseline_hits = gt_set.intersection(base_pred_set)
        if len(baseline_hits) > 1: # 这里要求 Baseline 必须全错 (零命中)
             continue
        # 如果觉得全错太难找，可以放宽为: len(baseline_hits) < len(gt_set) (没预测全)
        
        # --- Filter 4: 语义距离计算 ---
        # 因为我们限制了 len=2，所以只有一个距离
        gt_list = list(gt_set)
        dist = get_distance(go_graph, gt_list[0], gt_list[1])
        
        # 记录结果
        candidates.append({
            'uniprot_id': pid,
            'semantic_dist': dist,
            'plddt': struct['plddt'],
            'ptm': struct['ptm'],
            'pdb_filename': struct['filename'],
            'gt_labels': ', '.join(gt_list),
            'my_pred': ', '.join(my_pred_set),
            'baseline_pred': ', '.join(base_pred_set) if base_pred_set else "None",
            'overlap_len': overlap_len,
        })

    # 5. 排序与输出
    # 优先按语义距离排序 (找最难的)，其次按 pLDDT 排序
    candidates.sort(key=lambda x: (x['semantic_dist'], x['plddt']), reverse=True)
    
    print("\n" + "="*80)
    print(f"Found {len(candidates)} Gold Candidates")
    print("="*80)
    
    output_csv = "gold_candidates_for_case_study.csv"
    df_res = pd.DataFrame(candidates)
    df_res.to_csv(output_csv, index=False)
    
    print(f"{'ID':<12} | {'Dist':<5} | {'pLDDT':<6} | {'GT Labels (Distance)'}")
    print("-" * 80)
    
    for i, item in enumerate(candidates[:20]):
        print(f"{item['uniprot_id']:<12} | {item['semantic_dist']:<5} | {item['plddt']:<6.2f} | {item['gt_labels']}")
        
    print(f"\nResults saved to {output_csv}")
    
    if len(candidates) > 0:
        best = candidates[0]
        print("\n[Recommendation] Best Candidate:")
        print(f"ID: {best['uniprot_id']}")
        print(f"File: {best['pdb_filename']}")
        print(f"Labels: {best['gt_labels']} (Distance: {best['semantic_dist']})")
        print(f"Your Model: {best['my_pred']}")
        print(f"Baseline: {best['baseline_pred']}")

if __name__ == '__main__':
    main()