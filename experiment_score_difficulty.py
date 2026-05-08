# from src.byprot.utils.ontology import Ontology
# # import os
# # import pickle
# # import pandas as pd
# # import numpy as np
# # import networkx as nx
# # import matplotlib.pyplot as plt
# # from collections import defaultdict
# # from tqdm import tqdm
# # from sklearn.preprocessing import MultiLabelBinarizer
# # from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
# # import re

# # # ================= 配置区域 =================
# # # 1. 文件路径
# # TEST_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
# # GO_MAPPING_PATH = 'go_mapping.pkl'
# # GO_OBO_PATH = 'go-basic.obo'

# # # 预测结果 TSV (支持包含 _L=..._180 等后缀的ID)
# # # TSV_PATH = './generation-results-dplm2-goonly-unseen-all/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut_nondup_preds_mf.tsv'
# # TSV_PATH = './generation-results-dplm2-goonly-alldata-dm-ca-me-scale-0.2-weight-headclloss-2.0_sn-pn-11wstep/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut_preds_mf.tsv'
# # # TSV_PATH = './generation-results-cfpgen_650m/cfpgen_650m_go_preds_mf.tsv'
# # # NAME = 'cfpgen_650m'
# # NAME = 'ours'


# # # 2. 难度分桶设置 (Semantic Distance Intervals)
# # # 将难度分为: [0, 2), [2, 4), [4, 6), ...
# # # 0 通常是单标签或完全重叠， >12 通常是非常不相关的多功能
# # BINS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 100]
# # CONFIDENCE_THRESHOLD = 0.0

# # # 图表输出路径
# # PLOT_OUTPUT = f'f1_vs_semantic_difficulty_{NAME}.png'
# # CSV_OUTPUT = f'f1_vs_semantic_difficulty_{NAME}.csv'

# # # ================= 1. GO Ontology (模拟 src.byprot.utils.ontology) =================
# # class SimpleOntology:
# #     def __init__(self, obo_path):
# #         print(f"Loading GO Ontology from {obo_path}...")
# #         if not os.path.exists(obo_path):
# #             raise FileNotFoundError(f"请下载 go-basic.obo 文件并放置在 {obo_path}")
        
# #         self.G = nx.DiGraph() # 有向图: Child -> Parent
# #         self.undirected_G = nx.Graph() # 无向图: 用于计算语义距离
        
# #         with open(obo_path, 'r') as f:
# #             current_id = ""
# #             for line in f:
# #                 line = line.strip()
# #                 if line.startswith("id: GO:"):
# #                     current_id = line.split("id: ")[1]
# #                     self.G.add_node(current_id)
# #                     self.undirected_G.add_node(current_id)
# #                 elif line.startswith("is_a:"):
# #                     parent_id = line.split("is_a: ")[1].split(" ! ")[0]
# #                     if current_id:
# #                         self.G.add_edge(current_id, parent_id)
# #                         self.undirected_G.add_edge(current_id, parent_id)
# #                 elif line.startswith("relationship: part_of"):
# #                     parent_id = line.split("relationship: part_of ")[1].split(" ! ")[0]
# #                     if current_id:
# #                         self.G.add_edge(current_id, parent_id)
# #                         self.undirected_G.add_edge(current_id, parent_id)

# #     def get_ancestors(self, go_id):
# #         """获取所有祖先节点 (包括自己)"""
# #         if go_id not in self.G:
# #             return {go_id}
# #         # nx.descendants 在 Child->Parent 图中实际上是获取 Parents (祖先)
# #         ancestors = nx.descendants(self.G, go_id)
# #         ancestors.add(go_id)
# #         return list(ancestors)

# #     def calculate_set_difficulty(self, go_ids):
# #         """计算集合难度 (平均语义距离)"""
# #         if len(go_ids) < 2: return 0.0
# #         dists = []
# #         for i in range(len(go_ids)):
# #             for j in range(i+1, len(go_ids)):
# #                 u, v = go_ids[i], go_ids[j]
# #                 try:
# #                     if u in self.undirected_G and v in self.undirected_G:
# #                         d = nx.shortest_path_length(self.undirected_G, source=u, target=v)
# #                         dists.append(d)
# #                     else:
# #                         dists.append(50.0) # 惩罚
# #                 except nx.NetworkXNoPath:
# #                     dists.append(50.0) # 惩罚
# #         return np.mean(dists) if dists else 0.0

# # # ================= 2. 核心评估逻辑 (复现参考代码) =================
# # def compute_metrics_official(gt_list, pred_list, ontology):
# #     """
# #     完全复现你提供的 calculate_metrics 逻辑:
# #     1. Ancestor Propagation
# #     2. Label Intersection Filtering (unique_go)
# #     3. MultiLabelBinarizer
# #     4. Metrics Calculation
# #     """
# #     if len(gt_list) == 0:
# #         return None

# #     # 1. 标签传播 (Propagate to Ancestors)
# #     # 注意：为了防止内存爆炸和加速，我们先做去重，但逻辑上对每个样本都要做
# #     expanded_gt_list = []
# #     for gos in gt_list:
# #         new_gos = set()
# #         for go in gos:
# #             new_gos.update(ontology.get_ancestors(go))
# #         expanded_gt_list.append(new_gos)
    
# #     expanded_pred_list = []
# #     for gos in pred_list:
# #         new_gos = set()
# #         for go in gos:
# #             new_gos.update(ontology.get_ancestors(go))
# #         expanded_pred_list.append(new_gos)

# #     # 2. 标签空间过滤 (Unique GO Intersection)
# #     unique_go_gt = set()
# #     for go_set in expanded_gt_list:
# #         unique_go_gt.update(go_set)
    
# #     unique_go_pred = set()
# #     for go_set in expanded_pred_list:
# #         unique_go_pred.update(go_set)
    
# #     # === 关键步骤：只保留 GT 和 Pred 的交集 ===
# #     unique_go = unique_go_gt & unique_go_pred
    
# #     if len(unique_go) == 0:
# #         return {
# #             'f1_mic': 0.0, 'f1_mac': 0.0, 
# #             'precision_mic': 0.0, 'recall_mic': 0.0,
# #             'aupr_mic': 0.0, 'auc_mic': 0.0
# #         }

# #     final_pred_list = []
# #     for go_set in expanded_pred_list:
# #         final_pred_list.append([ele for ele in go_set if ele in unique_go])
        
# #     final_gt_list = []
# #     for go_set in expanded_gt_list:
# #         final_gt_list.append([ele for ele in go_set if ele in unique_go])

# #     # 3. 二值化 (MultiLabelBinarizer)
# #     mlb = MultiLabelBinarizer()
# #     # 必须用 all_go_terms fit，保证列对齐
# #     mlb.fit(final_gt_list + final_pred_list) 
    
# #     y_true_binary = mlb.transform(final_gt_list)
# #     y_pred_binary = mlb.transform(final_pred_list)

# #     # 4. 计算指标
# #     # F1 / Precision / Recall
# #     f1_mic = f1_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
# #     f1_mac = f1_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
    
# #     prec_mic = precision_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
# #     recall_mic = recall_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)

# #     # AUC / AUPR (需要 try-catch 防止单一类别导致报错)
# #     try:
# #         auc_mic = roc_auc_score(y_true_binary, y_pred_binary, average='micro')
# #         aupr_mic = average_precision_score(y_true_binary, y_pred_binary, average='micro')
# #     except ValueError:
# #         auc_mic = 0.0
# #         aupr_mic = 0.0

# #     return {
# #         'f1_mic': f1_mic,
# #         'f1_mac': f1_mac,
# #         'precision_mic': prec_mic,
# #         'recall_mic': recall_mic,
# #         'auc_mic': auc_mic,
# #         'aupr_mic': aupr_mic
# #     }

# # # ================= 3. 主流程 =================
# # def main():
# #     # --- Load Ontology ---
# #     ontology = SimpleOntology(GO_OBO_PATH)
    
# #     # --- Load Mappings & Data ---
# #     print("Loading Data mappings...")
# #     with open(GO_MAPPING_PATH, 'rb') as f:
# #         go_mapping = pickle.load(f)
# #     index_to_go = {v: k for k, v in go_mapping.items()}
    
# #     with open(TEST_PATH, 'rb') as f:
# #         test_data = pickle.load(f)
    
# #     # --- Pre-calculate Difficulty for Test Set ---
# #     print("Calculating difficulty for Ground Truth...")
# #     test_info = {} 
# #     for entry in tqdm(test_data):
# #         pid = entry['uniprot_id']
# #         gt_gos = [index_to_go[i] for i in entry['go_f_mapped']]
# #         diff = ontology.calculate_set_difficulty(gt_gos)
# #         test_info[pid] = {
# #             'gt': set(gt_gos),
# #             'difficulty': diff
# #         }

# #     # --- Load Predictions ---
# #     print(f"Loading Predictions from {TSV_PATH}...")
# #     try:
# #         df = pd.read_csv(TSV_PATH, sep='\t', header=None, names=['raw_id', 'go_id', 'score'])
# #     except:
# #         df = pd.read_csv(TSV_PATH, sep=r'\s+', header=None, names=['raw_id', 'go_id', 'score'])
    
# #     # ID Parsing (适配 _L=..._180 格式)
# #     def extract_clean_id(raw):
# #         # 兼容参考代码中的正则逻辑，也可以用 split
# #         if 'SEQUENCE_ID=' in raw:
# #             # 提取 P12345
# #             temp = raw.replace('SEQUENCE_ID=', '')
# #             return temp.split('_L=')[0]
# #         return raw

# #     df['clean_id'] = df['raw_id'].apply(extract_clean_id)
# #     df = df[df['score'] >= CONFIDENCE_THRESHOLD]

# #     # Group by Instance (raw_id)
# #     # 参考代码逻辑是：common_uids = [uid for uid in preds.keys() if uid in gts]
# #     # 我们这里需要保留每一次生成的样本作为独立评估实例
# #     instance_groups = df.groupby('raw_id')
    
# #     preds_map = defaultdict(list) # pid -> [set(gos), set(gos)...]
    
# #     for raw_id, group in instance_groups:
# #         pid = group['clean_id'].iloc[0]
# #         if pid in test_info:
# #             pred_set = set(group['go_id'])
# #             preds_map[pid].append(pred_set)

# #     # ================= 4. 分桶评估 =================
# #     print("\nStarting Bin Evaluation (Official Metrics Logic)...")
    
# #     bin_results = []
    
# #     for i in range(len(BINS) - 1):
# #         low, high = BINS[i], BINS[i+1]
# #         bin_label = f"{low}-{high}" if high < 100 else f">{low}"
        
# #         # 1. 找到该难度区间内的所有 PID
# #         target_pids = [
# #             pid for pid, info in test_info.items() 
# #             if low <= info['difficulty'] < high
# #         ]
        
# #         if not target_pids:
# #             continue
            
# #         # 2. 收集该区间内所有的 (GT, Pred) 对子
# #         # 你的数据是一对多 (1 GT vs 10 Predictions)
# #         # 评估时，我们将所有的预测实例展开，视为独立的测试样本进行 Micro/Macro 计算
        
# #         batch_gt_list = []
# #         batch_pred_list = []
        
# #         for pid in target_pids:
# #             gt_set = test_info[pid]['gt']
# #             predictions = preds_map.get(pid, [])
            
# #             # 如果该蛋白没有预测 (被过滤了)，通常应该算作空预测
# #             if not predictions:
# #                 # 这种情况下，GT有值，Pred为空
# #                 batch_gt_list.append(gt_set)
# #                 batch_pred_list.append(set())
# #             else:
# #                 for pred_set in predictions:
# #                     batch_gt_list.append(gt_set)
# #                     batch_pred_list.append(pred_set)
        
# #         # 3. 使用官方逻辑计算该 Bin 的指标
# #         print(f"Evaluating Bin [{bin_label}] with {len(batch_gt_list)} samples...")
# #         scores = compute_metrics_official(batch_gt_list, batch_pred_list, ontology)
        
# #         if scores:
# #             bin_results.append({
# #                 'bin_label': bin_label,
# #                 'num_samples': len(batch_gt_list),
# #                 'f1_micro': scores['f1_mic'],
# #                 'f1_macro': scores['f1_mac'],
# #                 'precision_micro': scores['precision_mic'],
# #                 'recall_micro': scores['recall_mic'],
# #                 'aupr_micro': scores['aupr_mic']
# #             })
# #             print(f"  -> F1 Micro: {scores['f1_mic']:.4f}, Macro: {scores['f1_mac']:.4f}")

# #     # ================= 5. 绘图 =================
# #     df_res = pd.DataFrame(bin_results)
# #     df_res.to_csv(CSV_OUTPUT, index=False)
    
# #     plt.figure(figsize=(10, 6))
# #     x_pos = range(len(df_res))
    
    

# #     plt.plot(x_pos, df_res['f1_micro'], 'o-', linewidth=2, label='F1 (Micro)', color='#1f77b4')
# #     plt.plot(x_pos, df_res['f1_macro'], 's--', linewidth=2, label='F1 (Macro)', color='#ff7f0e')
# #     # plt.plot(x_pos, df_res['aupr_micro'], '^:', linewidth=2, label='AUPR (Micro)', color='green')
    
# #     # 双轴显示样本数
# #     ax2 = plt.gca().twinx()
# #     ax2.bar(x_pos, df_res['num_samples'], color='gray', alpha=0.15, label='Num Samples')
# #     ax2.set_ylabel('Number of Prediction Instances')
    
# #     plt.xticks(x_pos, df_res['bin_label'])
# #     plt.title('Performance vs. Semantic Difficulty (Official Metrics)')
# #     plt.xlabel('Semantic Difficulty (Avg Distance)')
# #     plt.ylabel('Score')
# #     plt.legend(loc='upper right')
# #     plt.grid(True, linestyle='--', alpha=0.5)
    
# #     plt.tight_layout()
# #     plt.savefig(PLOT_OUTPUT, dpi=300)
# #     print(f"\nPlot saved to {PLOT_OUTPUT}")
# #     plt.show()

# # if __name__ == '__main__':
# #     main()


# # up: single

# import os
# import pickle
# import pandas as pd
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# from collections import defaultdict
# from tqdm import tqdm
# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.metrics import f1_score, recall_score

# # ================= 配置区域 =================
# # 1. 通用数据路径
# TRAIN_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl' # 新增训练集路径
# TEST_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'


# GO_MAPPING_PATH = 'go_mapping.pkl'
# GO_OBO_PATH = 'data/go.obo'

# # 2. 模型配置 (Name -> Path)
# MODELS = {
#     'Ours': './generation-results-dplm2-goonly-alldata-dm-ca-me-scale-0.2-weight-headclloss-2.0_sn-pn-11wstep/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut_preds_mf.tsv',
#     'CFPGen (Baseline)': './generation-results-cfpgen_650m/cfpgen_650m_go_preds_mf.tsv',
#     'Reference (Ground Truth)': '/AIRvePFS/dair/chenxr-data/repo/cfpgen/data-bin/uniprotKB/cfpgen_general_dataset/test_preds_mf.tsv',
# }

# # 3. 评估参数
# BINS = [0, 1, 2,3, 4, 5,6, 7,8,9, 10, 11,12]
# CONFIDENCE_THRESHOLD = 0.0

# # 图表输出路径
# PLOT_F1 = 'comparison_f1_vs_difficulty.png'
# PLOT_RECALL = 'comparison_recall_vs_difficulty.png'
# PLOT_DIST = 'comparison_sample_distribution.png' # 新增分布图路径


# def compute_metrics_official(gt_list, pred_list, ontology):
#     if len(gt_list) == 0: return None
    
#     # Propagate
#     expanded_gt = []
#     for gos in gt_list:
#         s = set()
#         for go in gos: s.update(ontology.get_ancestors(go))
#         # s.update(gos)

#         expanded_gt.append(s)
    
#     expanded_pred = []
#     for gos in pred_list:
#         s = set()
#         # for go in gos: s.update(ontology.get_ancestors(go))
#         s.update(gos)
#         expanded_pred.append(s)

#     # Intersection Filter
#     u_gt = set().union(*expanded_gt)
#     u_pred = set().union(*expanded_pred)
#     unique_go = u_gt & u_pred
    
#     if len(unique_go) == 0:
#         return {'f1_mic': 0.0, 'f1_mac': 0.0, 'rec_mic': 0.0, 'rec_mac': 0.0}

#     final_pred = [[e for e in s if e in unique_go] for s in expanded_pred]
#     final_gt = [[e for e in s if e in unique_go] for s in expanded_gt]

#     # Binarize
#     mlb = MultiLabelBinarizer()
#     mlb.fit(final_gt + final_pred)
#     y_true = mlb.transform(final_gt)
#     y_pred = mlb.transform(final_pred)

#     return {
#         'f1_mic': f1_score(y_true, y_pred, average='micro', zero_division=0),
#         'f1_mac': f1_score(y_true, y_pred, average='macro', zero_division=0),
#         'rec_mic': recall_score(y_true, y_pred, average='micro', zero_division=0),
#         'rec_mac': recall_score(y_true, y_pred, average='macro', zero_division=0)
#     }

# # ================= 2. Main Logic =================
# def main():
#     obo_path = GO_OBO_PATH
#     ontology = Ontology(obo_path, with_rels=True)
    
#     # --- Load Mappings ---
#     print("Loading GO Mapping...")
#     with open(GO_MAPPING_PATH, 'rb') as f:
#         index_to_go = {v: k for k, v in pickle.load(f).items()}
    
#     # --- Process Test Data ---
#     print(f"Loading Test Data from {TEST_PATH}...")
#     with open(TEST_PATH, 'rb') as f:
#         test_data = pickle.load(f)
    
#     test_info = {} 
#     test_difficulties = []
#     print("Calculating difficulty for Test Set...")
#     for entry in tqdm(test_data):
#         pid = entry['uniprot_id']
#         gt_gos = [index_to_go[i] for i in entry['go_f_mapped']]
#         diff = ontology.calculate_set_difficulty(gt_gos)
#         test_info[pid] = {'gt': set(gt_gos), 'difficulty': diff}
#         test_difficulties.append(diff)

#     # --- Process Train Data (新增部分) ---
#     print(f"Loading Train Data from {TRAIN_PATH}...")
#     with open(TRAIN_PATH, 'rb') as f:
#         train_data = pickle.load(f)
        
#     train_difficulties = []
#     print("Calculating difficulty for Train Set (for distribution plot)...")
#     for entry in tqdm(train_data):
#         gt_gos = [index_to_go[i] for i in entry['go_f_mapped']]
#         diff = ontology.calculate_set_difficulty(gt_gos)
#         train_difficulties.append(diff)

#     # --- Pre-calculate Bin Counts for Distribution Plot ---
#     bin_labels = []
#     train_counts = []
#     test_counts = []

#     for i in range(len(BINS) - 1):
#         low, high = BINS[i], BINS[i+1]
#         label = f"{low}-{high}"
#         bin_labels.append(label)
        
#         # Count items in this bin range
#         n_train = sum(low <= d < high for d in train_difficulties)
#         n_test = sum(low <= d < high for d in test_difficulties)
        
#         train_counts.append(n_train)
#         test_counts.append(n_test)

#     # --- Process Each Model for Metrics ---
#     all_results = {} # name -> DataFrame
    
#     for model_name, tsv_path in MODELS.items():
#         print(f"\nProcessing Model for Metrics: {model_name} ...")
        
#         try:
#             df = pd.read_csv(tsv_path, sep='\t', header=None, names=['raw_id', 'go_id', 'score'])
#         except:
#             df = pd.read_csv(tsv_path, sep=r'\s+', header=None, names=['raw_id', 'go_id', 'score'])
            
#         def get_clean_id(raw):
#             if 'SEQUENCE_ID=' in raw:
#                 return raw.replace('SEQUENCE_ID=', '').split('_L=')[0]
#             return raw 

#         df['clean_id'] = df['raw_id'].apply(get_clean_id)
#         df = df[df['score'] >= CONFIDENCE_THRESHOLD]
        
#         instance_groups = df.groupby('raw_id')
#         pid_to_instances = defaultdict(list)
        
#         for raw_id, group in instance_groups:
#             pid = group['clean_id'].iloc[0]
#             if pid in test_info:
#                 pid_to_instances[pid].append(set(group['go_id']))
                
#         bin_res = []
#         # Use the same bin labels we established earlier
#         for i, label in enumerate(bin_labels):
#             low, high = BINS[i], BINS[i+1]
            
#             target_pids = [p for p, info in test_info.items() if low <= info['difficulty'] < high]
            
#             if not target_pids: continue
            
#             batch_gt, batch_pred = [], []
#             for pid in target_pids:
#                 gt_set = test_info[pid]['gt']
#                 preds = pid_to_instances.get(pid, [])
#                 if not preds: 
#                     batch_gt.append(gt_set)
#                     batch_pred.append(set())
#                 else:
#                     for p_set in preds:
#                         batch_gt.append(gt_set)
#                         batch_pred.append(p_set)
            
#             scores = compute_metrics_official(batch_gt, batch_pred, ontology)
#             if scores:
#                 bin_res.append({
#                     'bin_label': label,
#                     'f1_micro': scores['f1_mic'],
#                     'f1_macro': scores['f1_mac'],
#                     'rec_micro': scores['rec_mic'],
#                     'rec_macro': scores['rec_mac']
#                 })
        
#         all_results[model_name] = pd.DataFrame(bin_res)

#     # ================= 3. Plotting Helpers =================
    
#     # Helper 1: Plot Metrics (Line Chart)
#     def plot_metric(metric_micro, metric_macro, y_label, output_file):
#         print(f"\nGenerating {y_label} Plot...")
#         plt.figure(figsize=(12, 7))
#         colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
        
#         first_df = list(all_results.values())[0]
#         # Use common bin labels for X-axis ordering
#         x_labels = bin_labels 
#         x_map = {lbl: i for i, lbl in enumerate(x_labels)}

#         for i, (name, df) in enumerate(all_results.items()):
#             c = colors[i % len(colors)]
            
#             # Align data to common X-axis based on labels
#             df_sorted = df.set_index('bin_label').reindex(x_labels).reset_index()
#             x = range(len(x_labels))
            
#             # Only plot existing data points (drop NaNs from reindex)
#             mask = df_sorted[metric_micro].notna()
            
#             plt.plot(np.array(x)[mask], df_sorted.loc[mask, metric_micro], marker='o', linestyle='-', linewidth=2, color=c, label=f'{name} (Micro)')
#             plt.plot(np.array(x)[mask], df_sorted.loc[mask, metric_macro], marker='^', linestyle='--', linewidth=1.5, alpha=0.7, color=c, label=f'{name} (Macro)')

#         plt.title(f'Ours vs. Baseline: {y_label} by Semantic Difficulty', fontsize=14)
#         plt.xlabel('Semantic Difficulty (Avg Shortest Path Distance)', fontsize=12)
#         plt.ylabel(y_label, fontsize=12)
        
#         plt.xticks(range(len(x_labels)), x_labels, rotation=0)
#         plt.legend(loc='upper right', fontsize=10, frameon=True, shadow=True)
#         plt.grid(True, linestyle=':', alpha=0.6)
#         plt.ylim(0, 1.0)
#         plt.tight_layout()
#         plt.savefig(output_file, dpi=300)
#         print(f"Plot saved to {output_file}")
#         plt.show()

#     # Helper 2: Plot Distribution (Bar Chart) - 新增函数
#     def plot_sample_distribution(labels, train_c, test_c, output_file):
#         print(f"\nGenerating Sample Distribution Plot...")
#         plt.figure(figsize=(12, 7))
        
#         x = np.arange(len(labels))
#         width = 0.35  # Bar width

#         fig, ax = plt.subplots(figsize=(12, 7))
#         rects1 = ax.bar(x - width/2, train_c, width, label='Train Set', color='#1f77b4', alpha=0.7)
#         rects2 = ax.bar(x + width/2, test_c, width, label='Test Set', color='#ff7f0e', alpha=0.7)

#         # Add some text for labels, title and custom x-axis tick labels, etc.
#         ax.set_ylabel('Number of Samples (Proteins)', fontsize=12)
#         ax.set_xlabel('Semantic Difficulty (Avg Shortest Path Distance)', fontsize=12)
#         ax.set_title('Sample Distribution by Difficulty: Train vs. Test', fontsize=14)
#         ax.set_xticks(x)
#         ax.set_xticklabels(labels)
#         ax.legend(fontsize=12)
#         ax.grid(True, axis='y', linestyle=':', alpha=0.6)

#         # # Add count labels on top of bars
#         # def autolabel(rects):
#         #     for rect in rects:
#         #         height = rect.get_height()
#         #         ax.annotate(f'{height}',
#         #                     xy=(rect.get_x() + rect.get_width() / 2, height),
#         #                     xytext=(0, 3),  # 3 points vertical offset
#         #                     textcoords="offset points",
#         #                     ha='center', va='bottom', fontsize=9)

#         # autolabel(rects1)
#         # autolabel(rects2)

#         fig.tight_layout()
#         plt.savefig(output_file, dpi=300)
#         print(f"Distribution plot saved to {output_file}")
#         plt.show()


#     # ================= 4. Generate Plots =================
#     # Plot 1: F1 Score
#     plot_metric('f1_micro', 'f1_macro', 'F1 Score', PLOT_F1)
    
#     # Plot 2: Recall
#     plot_metric('rec_micro', 'rec_macro', 'Recall', PLOT_RECALL)


#     train_counts = [count / len(train_data) for count in train_counts]
#     test_counts = [count / len(test_data) for count in test_counts]
#     # Plot 3: Sample Distribution (新增)
#     plot_sample_distribution(bin_labels, train_counts, test_counts, PLOT_DIST)



# # ================= 5. 深入探究实验 (Deep Dive Analysis) =================
#     print("\n" + "="*60)
#     print("STARTING DEEP DIVE ANALYSIS")
#     print("="*60)

#     # 这里的分析主要针对 'Ours' 模型，如果你想看 Baseline 也可以改
#     target_model_name = 'Ours'
#     if target_model_name not in MODELS:
#         target_model_name = list(MODELS.keys())[0]
    
#     print(f"Analyzing specific issues for model: {target_model_name}")
#     tsv_path = MODELS[target_model_name]

#     # 加载该模型的预测数据
#     try:
#         df = pd.read_csv(tsv_path, sep='\t', header=None, names=['raw_id', 'go_id', 'score'])
#     except:
#         df = pd.read_csv(tsv_path, sep=r'\s+', header=None, names=['raw_id', 'go_id', 'score'])
    
#     def get_clean_id(raw):
#         if 'SEQUENCE_ID=' in raw:
#             return raw.replace('SEQUENCE_ID=', '').split('_L=')[0]
#         return raw 
#     df['clean_id'] = df['raw_id'].apply(get_clean_id)
#     df = df[df['score'] >= CONFIDENCE_THRESHOLD]
    
#     # 组织预测数据
#     instance_groups = df.groupby('raw_id')
#     pid_to_preds = defaultdict(list)
#     for raw_id, group in instance_groups:
#         pid = group['clean_id'].iloc[0]
#         if pid in test_info:
#             pid_to_preds[pid].append(set(group['go_id']))

#     # --- 实验 1: 0-4 区间 Over-prediction 分析 ---
#     print("\n[Deep Dive 1] Analyzing Bin 0-4 (Low F1, High Recall)...")
    
#     analysis_0_4 = []
#     target_pids_0_4 = [p for p, info in test_info.items() if 0 <= info['difficulty'] < 4]

#     for pid in target_pids_0_4:
#         gt_set = test_info[pid]['gt']
#         preds_list = pid_to_preds.get(pid, [])
        
#         if not preds_list: continue

#         # 对每个样本的多次预测取平均指标
#         for pred_set in preds_list:
#             # 1. 数量比率
#             size_ratio = len(pred_set) / len(gt_set) if len(gt_set) > 0 else len(pred_set)
            
#             # 2. FP 分析
#             fp_set = pred_set - gt_set
#             avg_fp_dist = np.nan
#             if fp_set and gt_set:
#                 # 计算每个 FP 到最近 GT 的距离
#                 dists = []
#                 for fp in fp_set:
#                     min_d = min([ontology.get_semantic_distance(fp, gt) for gt in gt_set])
#                     dists.append(min_d)
#                 avg_fp_dist = np.mean(dists)

#             analysis_0_4.append({
#                 'pid': pid,
#                 'gt_size': len(gt_set),
#                 'pred_size': len(pred_set),
#                 'size_ratio': size_ratio,
#                 'avg_fp_dist': avg_fp_dist
#             })
    
#     df_0_4 = pd.DataFrame(analysis_0_4)
#     print(f"  -> Avg Prediction/GT Size Ratio: {df_0_4['size_ratio'].mean():.2f}")
#     print(f"  -> (如果 > 1.5 说明明显多预测了)")
#     print(f"  -> Avg Distance of False Positives to GT: {df_0_4['avg_fp_dist'].mean():.2f}")
#     print(f"  -> (如果 < 2.0 说明虽然预测错了，但错得很'近'，比如父子关系)")

#     # 画图：预测数量分布
#     plt.figure(figsize=(8, 5))
    
#     plt.hist(df_0_4['size_ratio'], bins=30, color='purple', alpha=0.7, range=(0, 5))
#     plt.axvline(1.0, color='red', linestyle='dashed', linewidth=1)
#     plt.title('Bin 0-4: Ratio of Predicted vs GT Label Count')
#     plt.xlabel('Pred Size / GT Size')
#     plt.ylabel('Frequency')
#     plt.savefig('analysis_0_4_size_ratio.png')
#     plt.show()

#     # --- 实验 2: 4-6 区间 Performance Drop 分析 ---
#     print("\n[Deep Dive 2] Analyzing Bin 4-6 (Performance Drop)...")
    
#     analysis_4_6 = []
#     target_pids_4_6 = [p for p, info in test_info.items() if 4 <= info['difficulty'] < 5]
    
#     # 为了对比，我们也取一下 6-8 区间看看区别
#     target_pids_6_8 = [p for p, info in test_info.items() if 6 <= info['difficulty'] < 8]

#     # 统计 FN (漏掉的词) 的特征
#     fn_namespace_counts = defaultdict(int)
    
#     for pid in target_pids_4_6:
#         gt_set = test_info[pid]['gt']
#         preds_list = pid_to_preds.get(pid, [])
        
#         if not preds_list: # 完全没预测出来
#             for go in gt_set: fn_namespace_counts[ontology.get_namespace(go)] += 1
#             continue

#         for pred_set in preds_list:
#             fn_set = gt_set - pred_set
#             for go in fn_set:
#                 # 统计漏掉的词属于哪个命名空间 (MF/BP/CC)
#                 # 注意：需要你的 ontology 类有 get_namespace 方法，或者直接查 dict
#                 if go in ontology.ont:
#                     ns = ontology.ont[go].get('namespace', 'unknown')
#                     fn_namespace_counts[ns] += 1
    
#     print("  -> Missed Terms (FN) Distribution in Bin 4-6:")
#     total_fns = sum(fn_namespace_counts.values())
#     for ns, count in fn_namespace_counts.items():
#         print(f"     {ns}: {count} ({count/total_fns*100:.1f}%)")
    
#     # 检查训练集覆盖率：4-6 的样本中的 GO 词，在训练集中出现的频率是否显著低于其他组？
#     # 我们利用之前算好的 train_data 统计
#     train_go_flat = []
#     for entry in train_data:
#         train_go_flat.extend([index_to_go[i] for i in entry['go_f_mapped']])
#     from collections import Counter
#     train_go_counter = Counter(train_go_flat)

#     def get_avg_train_freq(pids):
#         freqs = []
#         for pid in pids:
#             gt_set = test_info[pid]['gt']
#             for go in gt_set:
#                 freqs.append(train_go_counter.get(go, 0))
#         return np.mean(freqs) if freqs else 0

#     freq_4_6 = get_avg_train_freq(target_pids_4_6)
#     freq_6_8 = get_avg_train_freq(target_pids_6_8) # 对照组
    
#     print(f"  -> Avg Train Frequency of labels in Bin 4-6: {freq_4_6:.1f}")
#     print(f"  -> Avg Train Frequency of labels in Bin 6-8: {freq_6_8:.1f}")
    
#     if freq_4_6 < freq_6_8:
#         print("  -> 结论暗示: 4-6 区间表现差可能是因为包含的标签本身在训练集中就是长尾(Long-tail)样本。")
#     else:
#         print("  -> 结论暗示: 频率正常，可能是组合模式难以学习。")

#     # 保存 4-6 区间表现最差的 Top 20 样本供人工检查
#     worst_cases = []
#     for pid in target_pids_4_6:
#         gt_set = test_info[pid]['gt']
#         preds_list = pid_to_preds.get(pid, [])
#         if not preds_list:
#             worst_cases.append({'pid': pid, 'f1': 0.0, 'gt': gt_set, 'pred_gt': None})
#             continue
#         # 取平均 F1
#         f1s = []
#         for p_set in preds_list:
#             intersection = len(gt_set & p_set)
#             f1 = 2*intersection / (len(gt_set)+len(p_set)) if (len(gt_set)+len(p_set))>0 else 0
#             f1s.append(f1)
#         worst_cases.append({'pid': pid, 'f1': np.mean(f1s), 'gt': gt_set, 'pred_gt': preds_list})
    
#     worst_cases.sort(key=lambda x: x['f1'])
#     pd.DataFrame(worst_cases).to_csv('analysis_4_6_worst_cases.csv', index=False)
#     print("  -> Saved worst performing examples in 4-6 to 'analysis_4_6_worst_cases.csv'")


# # ================= 6. 统计分布视角深入探究 (Updated with TF-IDF & Recall) =================
#     print("\n" + "="*80)
#     print("STARTING DISTRIBUTION-BASED ANALYSIS (Freq, Co-occurrence, TF-IDF)")
#     print("="*80)

#     # 1. 建立训练集统计特征 (Training Set Statistics)
#     print("Building Training Set Statistics & Computing IDF...")
#     train_label_counts = defaultdict(int)
#     train_pair_counts = defaultdict(int)
#     total_train_docs = len(train_data)
    
#     from itertools import combinations
    
#     # 遍历训练集统计词频 (Document Frequency)
#     for entry in tqdm(train_data, desc="Scanning Train Stats"):
#         go_ids = sorted([index_to_go[i] for i in entry['go_f_mapped']])
        
#         # 统计 DF (Document Frequency)
#         # 注意：这里我们统计的是有多少个蛋白包含了该GO (Binary count per doc)
#         unique_gos = set(go_ids)
#         for go in unique_gos:
#             train_label_counts[go] += 1
            
#         # 统计标签对共现
#         if len(unique_gos) > 1:
#             for pair in combinations(sorted(list(unique_gos)), 2):
#                 train_pair_counts[pair] += 1

#     # === [新增] 计算 IDF 字典 ===
#     # IDF = log( Total_Docs / (DF + 1) )
#     idf_dict = {}
#     for go, df_count in train_label_counts.items():
#         # +1 平滑防止除零
#         idf_dict[go] = np.log10(total_train_docs / (df_count + 1))
    
#     # 对于没见过的词，给予最大 IDF
#     default_max_idf = np.log10(total_train_docs / 1)

#     # 2. 分析测试集样本
#     target_model_name = 'Ours'
#     if target_model_name not in all_results:
#          pass # 假设上下文已加载

#     print(f"\nAnalyzing model '{target_model_name}' performance against Dist stats...")
    
#     # 重新加载预测 (确保数据是最新的)
#     tsv_path = MODELS[target_model_name]
#     try:
#         df = pd.read_csv(tsv_path, sep='\t', header=None, names=['raw_id', 'go_id', 'score'])
#     except:
#         df = pd.read_csv(tsv_path, sep=r'\s+', header=None, names=['raw_id', 'go_id', 'score'])
    
#     def get_clean_id(raw):
#         if 'SEQUENCE_ID=' in raw:
#             return raw.replace('SEQUENCE_ID=', '').split('_L=')[0]
#         return raw 
#     df['clean_id'] = df['raw_id'].apply(get_clean_id)
#     df = df[df['score'] >= CONFIDENCE_THRESHOLD]
    
#     instance_groups = df.groupby('raw_id')
#     pid_to_metrics = defaultdict(list)
    
#     # 计算每个样本的 F1 和 Recall
#     for raw_id, group in instance_groups:
#         pid = group['clean_id'].iloc[0]
#         if pid in test_info:
#             gt_set = test_info[pid]['gt']
#             pred_set = set(group['go_id'])

#             s = set()
#             for go in gt_set: s.update(ontology.get_ancestors(go))
#             gt_set = s
            
#             # # Metric Calculation
#             intersection = len(gt_set & pred_set)
            
#             # # F1
#             f1 = 2 * intersection / (len(gt_set) + len(pred_set)) if (len(gt_set) + len(pred_set)) > 0 else 0.0
            
#             # # Recall
#             rec = intersection / len(gt_set) if len(gt_set) > 0 else 0.0
            
#             # scores = compute_metrics_official([gt_set], [pred_set], ontology)

#             pid_to_metrics[pid].append({'f1': f1, 'recall': rec})

#     # 3. 计算每个 Test 样本的统计指标 (含 TF-IDF)
#     dist_analysis_data = []
    
#     for pid, info in test_info.items():
#         if pid not in pid_to_metrics: continue 
        
#         gt_set = list(info['gt'])
#         if not gt_set: continue
        
#         # --- A. 平均标签频率 (Log Frequency) ---
#         freqs = [train_label_counts.get(go, 0) for go in gt_set]
#         avg_freq = np.mean(freqs)
#         log_avg_freq = np.log10(avg_freq + 1)
        
#         # --- B. 未见对子比例 (Unseen Pair Ratio) ---
#         num_pairs = 0
#         unseen_pairs = 0
#         if len(gt_set) > 1:
#             for pair in combinations(sorted(gt_set), 2):
#                 num_pairs += 1
#                 if train_pair_counts.get(pair, 0) == 0:
#                     unseen_pairs += 1
#             unseen_pair_ratio = unseen_pairs / num_pairs
#         else:
#             unseen_pair_ratio = 0.0
            
#         # --- [新增] C. 组合 TF-IDF (Set TF-IDF) ---
#         # 这里我们将一个蛋白视为一篇文档，计算其标签集合的平均 IDF 值
#         # 平均 IDF 越高，说明这个组合里的词越稀有、越特异 (Specificity)
#         idfs = [idf_dict.get(go, default_max_idf) for go in gt_set]
#         avg_idf = np.mean(idfs)
            
#         # 获取该样本的平均 F1 和 Recall
#         metrics_list = pid_to_metrics[pid]
#         avg_f1 = np.mean([m['f1'] for m in metrics_list])
#         avg_rec = np.mean([m['recall'] for m in metrics_list])


#         # co-currence
#         cooc_strengths = []
#         if len(gt_set) > 1:
#             for pair in combinations(sorted(gt_set), 2):
#                 # 获取 A, B 在训练集共同出现的次数
#                 pair_count = train_pair_counts.get(pair, 0)
                
#                 # 获取 A, B 各自出现的次数
#                 count_a = train_label_counts.get(pair[0], 0)
#                 count_b = train_label_counts.get(pair[1], 0)
                
#                 # 计算 Jaccard Co-occurrence Coefficient
#                 # 公式: Intersection / Union
#                 union_count = count_a + count_b - pair_count
                
#                 if union_count > 0:
#                     strength = pair_count / union_count
#                 else:
#                     strength = 0.0
                
#                 cooc_strengths.append(strength)
            
#             # 取平均值作为该样本的得分
#             avg_cooc_strength = np.mean(cooc_strengths)
#         else:
#             # 单标签没有共现概念，设为 NaN 或者 -1，或者 1.0 (视作完美自洽)
#             # 为了分布图好看，对于单标签通常设为 NaN 并在画图时过滤，或者设为 0
#             # 这里我们设为 0，意味着"没有共现支持"
#             avg_cooc_strength = 0.0
        
#         dist_analysis_data.append({
#             'pid': pid,
#             'log_avg_freq': log_avg_freq,
#             'unseen_pair_ratio': unseen_pair_ratio,
#             'avg_idf': avg_idf,
#             'avg_cooc_strength': avg_cooc_strength,


#             'f1': avg_f1,
#             'recall': avg_rec,
#             'num_labels': len(gt_set)
#         })
        
#     df_dist = pd.DataFrame(dist_analysis_data)
    
#     # ================= 绘图通用函数 (双 Y 轴或双线) =================
#     def plot_dual_metric_trend(df, x_col, x_label, title, filename, bins=100):
#         plt.figure(figsize=(10, 6))
        
#         # 自动分桶
#         if df[x_col].nunique() > bins:
#             df['bin'] = pd.cut(df[x_col], bins=bins)
#             # 取中点作为 X 轴坐标
#             stats = df.groupby('bin', observed=True)[['f1', 'recall']].mean().reset_index()
#             # x_axis = stats['bin'].apply(lambda x: x.mid).astype(float)
#             # 为了处理某些空桶导致的分类类型问题，转换一下
#             stats['x_center'] = stats['bin'].apply(lambda x: x.mid).astype(float)
#             x_axis = stats['x_center']
#             y_f1 = stats['f1']
#             y_rec = stats['recall']
#         else:
#             # 如果本身就是离散的 (比如 Ratio只有几个值)，直接 Groupby
#             stats = df.groupby(x_col)[['f1', 'recall']].mean().reset_index()
#             x_axis = stats[x_col]
#             y_f1 = stats['f1']
#             y_rec = stats['recall']
            
#         # 绘图
        
        
#         plt.plot(x_axis, y_f1, marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='F1 Score')
#         plt.plot(x_axis, y_rec, marker='^', linestyle='--', color='#2ca02c', linewidth=2, label='Recall')
        
#         # 填充区域显示差异
#         plt.fill_between(x_axis, y_f1, y_rec, color='gray', alpha=0.1, label='F1-Recall Gap')
        
#         plt.title(title, fontsize=14)
#         plt.xlabel(x_label, fontsize=12)
#         plt.ylabel('Score', fontsize=12)
#         plt.ylim(0, 1.05)
#         plt.legend()
#         plt.grid(True, linestyle=':', alpha=0.6)
        
#         plt.tight_layout()
#         plt.savefig(filename, dpi=300)
#         print(f"Saved plot: {filename}")
#         plt.show()

#     # ================= 生成图表 =================
    
#     # 1. Metrics vs. Label Frequency
#     # 预期：频率越高，F1 和 Recall 都应该上升
#     plot_dual_metric_trend(
#         df_dist, 
#         'log_avg_freq', 
#         'Log10(Avg Label Frequency in Train)', 
#         'Impact of Label Frequency on F1 & Recall',
#         'analysis_dist_freq_dual.png'
#     )

#     # 2. Metrics vs. Unseen Pair Ratio
#     # 预期：未见对子越多，F1 下降。Recall 可能会因为模型乱猜而保持较高，或者也一起下降。
#     # 只看多标签数据
#     df_multi = df_dist[df_dist['num_labels'] > 1].copy()
#     plot_dual_metric_trend(
#         df_multi, 
#         'unseen_pair_ratio', 
#         'Ratio of Unseen Label Pairs (0=Familiar, 1=Novel)', 
#         'Impact of Compositional Novelty on F1 & Recall',
#         'analysis_dist_pair_ratio_dual.png'
#     )
    
#     # 3. Metrics vs. TF-IDF (Specificity)
#     # 预期：TF-IDF 越高（组合越罕见/特异），F1 通常会下降（因为难），
#     # 但如果 Recall 保持很高而 F1 很低，说明模型在对罕见功能进行“过度覆盖”预测。
#     plot_dual_metric_trend(
#         df_dist, 
#         'avg_idf', 
#         'Average IDF of Label Set (Higher = More Specific/Rare)', 
#         'Impact of Label Specificity (TF-IDF) on F1 & Recall',
#         'analysis_dist_tfidf_dual.png'
#     )

#     # 保存数据
#     df_dist.to_csv('analysis_distribution_tfidf_metrics.csv', index=False)

# # ================= 7. 困难样本分布对比分析 (Hard Sample Distribution Analysis) =================
#     print("\n" + "="*80)
#     print("STARTING HARD SAMPLE (Difficulty 4-6) DISTRIBUTION ANALYSIS")
#     print("="*80)

#     # 1. 识别困难样本 (Difficulty 4-6)
#     # 注意：这里复用之前计算的 test_info
#     target_pids_4_6 = set([p for p, info in test_info.items() if 4 <= info['difficulty'] < 5])
#     print(f" identified {len(target_pids_4_6)} hard samples (Difficulty 4-6).")

#     # 2. 在 df_dist 中标记这些样本
#     # df_dist 是上一步 (第6部分) 生成的包含所有统计指标的 DataFrame
#     if 'pid' not in df_dist.columns:
#         # 防御性编程：如果 df_dist 没有 pid 列，尝试重新构建或报错
#         print("Error: df_dist missing 'pid' column. Make sure Section 6 is run.")
#     else:
#         df_dist['is_hard'] = df_dist['pid'].apply(lambda x: x in target_pids_4_6)
        
#         # 分离出 全集 和 困难子集
#         df_hard = df_dist[df_dist['is_hard'] == True]
        
#         # 定义绘图函数：对比分布
#         def plot_distribution_comparison(col_name, x_label, title, filename, bins=20):
#             plt.figure(figsize=(10, 6))
            
#             # 绘制整体分布 (Overall) - 灰色背景
#             # 使用 density=True 以便对比分布形状，而非绝对数量
#             plt.hist(df_dist[col_name], bins=bins, color='gray', alpha=0.3, label='Overall Test Set', density=True, edgecolor='black', linestyle='--')
            
#             # 绘制困难样本分布 (Hard Samples) - 红色高亮
#             plt.hist(df_hard[col_name], bins=bins, color='#d62728', alpha=0.6, label='Hard Samples (Diff 4-6)', density=True, edgecolor='red')
            
#             # 添加均值线
#             mean_overall = df_dist[col_name].mean()
#             mean_hard = df_hard[col_name].mean()
#             plt.axvline(mean_overall, color='gray', linestyle='dashed', linewidth=1.5, label=f'Mean Overall ({mean_overall:.2f})')
#             plt.axvline(mean_hard, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean Hard ({mean_hard:.2f})')
            
#             plt.title(title, fontsize=14)
#             plt.xlabel(x_label, fontsize=12)
#             plt.ylabel('Density (Normalized Frequency)', fontsize=12)
#             plt.legend()
#             plt.grid(True, linestyle=':', alpha=0.5)
            
#             plt.tight_layout()
#             plt.savefig(filename, dpi=300)
#             print(f"Saved distribution plot: {filename}")
#             plt.show()

#         # --- A. 对比 Label Frequency 分布 ---
#         # 假设：困难样本可能更偏向左侧 (低频)
#         plot_distribution_comparison(
#             'log_avg_freq',
#             'Log10(Avg Label Frequency)',
#             'Distribution Shift: Label Frequency (Overall vs Hard 4-6)',
#             'analysis_hard_dist_freq.png',
#             bins=30
#         )

#         # --- B. 对比 Unseen Pair Ratio 分布 ---
#         # 假设：困难样本可能更偏向右侧 (更多没见过的组合)
#         # 仅针对多标签样本进行对比，因为单标签样本 ratio 默认为 0
#         df_dist_multi = df_dist[df_dist['num_labels'] > 1]
#         df_hard_multi = df_hard[df_hard['num_labels'] > 1]
        
#         plt.figure(figsize=(10, 6))
#         plt.hist(df_dist_multi['unseen_pair_ratio'], bins=50, color='gray', alpha=0.3, label='Overall (Multi-label)', density=True, edgecolor='black', linestyle='--')
#         plt.hist(df_hard_multi['unseen_pair_ratio'], bins=50, color='#d62728', alpha=0.6, label='Hard Samples (Diff 4-6)', density=True, edgecolor='red')
        
#         plt.title('Distribution Shift: Unseen Pair Ratio (Overall vs Hard 4-6)', fontsize=14)
#         plt.xlabel('Ratio of Unseen Label Pairs (0=Familiar, 1=Novel)', fontsize=12)
#         plt.ylabel('Density', fontsize=12)
#         plt.legend()
#         plt.grid(True, linestyle=':', alpha=0.5)
#         plt.tight_layout()
#         plt.savefig('analysis_hard_dist_pair_ratio.png', dpi=300)
#         plt.show()

#         # --- C. 对比 TF-IDF (Specificity) 分布 ---
#         # 假设：困难样本可能更偏向右侧 (更高特异性/信息量)
#         plot_distribution_comparison(
#             'avg_idf',
#             'Avg IDF (Specificity)',
#             'Distribution Shift: Label Specificity (Overall vs Hard 4-6)',
#             'analysis_hard_dist_idf.png',
#             bins=30
#         )

# # --- [新增] D. 对比 Co-occurrence Strength 分布 ---
#         # 假设：困难样本 (4-6) 的共现强度可能较低 (即由“非典型搭档”组成的“散装”功能)
        
#         # 仅针对多标签样本，因为单标签该指标无意义
#         df_dist_multi = df_dist[df_dist['num_labels'] > 1]
#         df_hard_multi = df_hard[df_hard['num_labels'] > 1]

#         # 同样需要处理 bins=50 和局部放大，因为很多 unseen pair 的强度是 0
#         fig, ax = plt.subplots(figsize=(10, 6))
        
#         bins_count = 50
        
#         ax.hist(df_dist_multi['avg_cooc_strength'], bins=bins_count, range=(0, 1), 
#                 color='gray', alpha=0.3, label='Overall (Multi-label)', 
#                 density=True, edgecolor='gray', linestyle='--')
        
#         ax.hist(df_hard_multi['avg_cooc_strength'], bins=bins_count, range=(0, 1), 
#                 color='#d62728', alpha=0.6, label='Hard Samples (Diff 4-6)', 
#                 density=True, edgecolor='red')
        
#         # 添加均值线
#         mean_all_cooc = df_dist_multi['avg_cooc_strength'].mean()
#         mean_hard_cooc = df_hard_multi['avg_cooc_strength'].mean()
#         ax.axvline(mean_all_cooc, color='gray', linestyle='dashed', linewidth=1)
#         ax.axvline(mean_hard_cooc, color='red', linestyle='dashed', linewidth=1)

#         ax.set_title('Distribution Shift: Co-occurrence Strength (Overall vs Hard 4-6)', fontsize=14)
#         ax.set_xlabel('Avg Pairwise Co-occurrence Strength (Jaccard)', fontsize=12)
#         ax.set_ylabel('Density', fontsize=12)
#         ax.legend(loc='upper right')
#         ax.grid(True, linestyle=':', alpha=0.5)
        
#         # # [可选] 添加嵌入子图放大低值区域 (0-0.1)
#         # # 因为对于困难样本，强度可能都集中在接近 0 的地方
#         # from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#         # ax_ins = inset_axes(ax, width="40%", height="40%", loc='center right', borderpad=2)
        
#         # zoom_range = (0, 0.2)
#         # ax_ins.hist(df_dist_multi['avg_cooc_strength'], bins=20, range=zoom_range, color='gray', alpha=0.3, density=True)
#         # ax_ins.hist(df_hard_multi['avg_cooc_strength'], bins=20, range=zoom_range, color='#d62728', alpha=0.6, density=True)
#         # ax_ins.set_title('Zoom: 0 - 0.2', fontsize=10)
#         # ax_ins.grid(True, linestyle=':', alpha=0.3)

#         plt.tight_layout()
#         plt.savefig('analysis_hard_dist_cooc_strength.png', dpi=300)
#         print("Saved distribution plot: analysis_hard_dist_cooc_strength.png")
#         plt.show()

#         # 更新统计报告打印
#         print("\nStatistical Summary (Mean Values) - Updated:")
#         print(f"{'Metric':<25} | {'Overall':<10} | {'Hard (4-6)':<10} | {'Difference'}")
#         print("-" * 65)
#         # 加入新指标
#         cols_to_print = ['log_avg_freq', 'unseen_pair_ratio', 'avg_idf', 'avg_cooc_strength']
#         for col in cols_to_print:
#             # 对于 cooc_strength 只统计多标签样本
#             if col == 'avg_cooc_strength':
#                 m_all = df_dist_multi[col].mean()
#                 m_hard = df_hard_multi[col].mean()
#             else:
#                 m_all = df_dist[col].mean()
#                 m_hard = df_hard[col].mean()
                
#             diff = m_hard - m_all
#             print(f"{col:<25} | {m_all:<10.4f} | {m_hard:<10.4f} | {diff:+.4f}")


# # ================= 8. 基于标签/组合难度的归因分析 (Label/Pair-wise Attribution) =================
#     print("\n" + "="*80)
#     print("STARTING ATTRIBUTION ANALYSIS: Why is Bin 4-6 hard?")
#     print("="*80)

#     # --- Step 1: 全局统计 - 计算每个单标签的 Recall ---
#     print("Step 1: Calculating Global Recall per GO Term...")
    
#     # 统计 GT (分母) 和 Hit (分子)
#     term_gt_counts = defaultdict(int)
#     term_hit_counts = defaultdict(int)
    
#     # 统计 Pair GT 和 Pair Hit
#     pair_gt_counts = defaultdict(int)
#     pair_hit_counts = defaultdict(int)

#     # 遍历所有测试集样本 (不分难度)
#     for pid, info in test_info.items():
#         if pid not in pid_to_preds: continue
        
#         gt_set = info['gt']
#         # 获取该样本的预测合集 (Ours 模型可能有多条预测，这里取并集算 Hit，或者取平均 Recall)
#         # 为了定义明确，这里采用 "Any Hit" (只要预测出的集合里有就算)，或者更严格的 "Instance Avg".
#         # 让我们用 Instance Avg 来平滑：
#         # 对于每个标签，如果它在 GT 里，看它在 N 次预测中出现了几次 (frequency in prediction)
        
#         preds_list = pid_to_preds[pid]
#         n_preds = len(preds_list)
#         if n_preds == 0: continue

        
# # --- 单标签统计 ---
#         for go in gt_set:
#             term_gt_counts[go] += 1
#             # 统计这个 GO 被预测到的次数
#             n_hits = sum(1 for p_set in preds_list if go in p_set)
#             # 贡献给分子的值 = 命中率 (0.0 - 1.0)
#             term_hit_counts[go] += (n_hits / n_preds)

#         # --- 共现对统计 (Co-existing Pairs) ---
#         if len(gt_set) > 1:
#             for pair in combinations(sorted(gt_set), 2):
#                 pair_gt_counts[pair] += 1
#                 # 统计这个 Pair 同时被预测到的次数
#                 n_pair_hits = sum(1 for p_set in preds_list if (pair[0] in p_set and pair[1] in p_set))
#                 pair_hit_counts[pair] += (n_pair_hits / n_preds)

#     # 计算最终 Recall
#     term_recall_map = {}
#     for go, total in term_gt_counts.items():
#         if total > 0:
#             term_recall_map[go] = term_hit_counts[go] / total
    
#     pair_recall_map = {}
#     for pair, total in pair_gt_counts.items():
#         if total > 0:
#             pair_recall_map[pair] = pair_hit_counts[pair] / total
            
#     # 全局平均水平 (Baseline)
#     global_avg_term_recall = np.mean(list(term_recall_map.values()))
#     global_avg_pair_recall = np.mean(list(pair_recall_map.values()))
#     print(f"Global Avg Term Recall: {global_avg_term_recall:.4f}")
#     print(f"Global Avg Pair Recall: {global_avg_pair_recall:.4f}")

#     # --- Step 2: 局部映射 - 分析 Bin 4-6 的成分 ---
#     print("\nStep 2: Mapping Hard Samples (Bin 4-6) to Global Statistics...")
    
#     # 这里的 df_dist 应该包含所有样本，我们需要提取 Bin 4-6 的行
#     # 如果没有 'difficulty' 列，我们需要从 test_info 映射
#     # 假设 df_dist 还在内存里
    
#     # 为了方便绘图，我们构建一个新的 DataFrame，包含每个样本的 "成分难度"
#     composition_analysis = []
    
#     for pid, info in test_info.items():
#         if pid not in pid_to_preds: continue
#         gt_set = info['gt']
#         if not gt_set: continue
        
#         # A. 该样本包含的标签的平均难度 (Avg Term Recall)
#         # 值越低，说明这个蛋白由一堆"难预测"的词组成
#         term_recalls = [term_recall_map.get(go, 0.0) for go in gt_set] # 0.0 if not in map (shouldn't happen for GT)
#         avg_sample_term_recall = np.mean(term_recalls)
        
#         # B. 该样本包含的组合的平均难度 (Avg Pair Recall)
#         if len(gt_set) > 1:
#             pair_recalls = []
#             for pair in combinations(sorted(gt_set), 2):
#                 pair_recalls.append(pair_recall_map.get(pair, 0.0))
#             avg_sample_pair_recall = np.mean(pair_recalls)
#         else:
#             avg_sample_pair_recall = np.nan # 单标签没有组合难度

#         composition_analysis.append({
#             'pid': pid,
#             'difficulty': info['difficulty'],
#             'is_hard_bin': (4 <= info['difficulty'] < 5),
#             'avg_term_difficulty': avg_sample_term_recall, # 注意：这是Recall，越低越难
#             'avg_pair_difficulty': avg_sample_pair_recall
#         })
        
#     df_comp = pd.DataFrame(composition_analysis)
    
#     # 分离组
#     df_comp_hard = df_comp[df_comp['is_hard_bin'] == True]
#     df_comp_others = df_comp[df_comp['is_hard_bin'] == False]
    
#     # ================= 绘图 1: 单标签难度分布对比 =================
#     plt.figure(figsize=(10, 6))
    
#     # 注意：X轴是 Recall，越靠左越难
#     plt.hist(df_comp_others['avg_term_difficulty'], bins=5, color='gray', density=True, alpha=0.3,  label='Other Samples', edgecolor='gray', range=[0,1])
#     plt.hist(df_comp_hard['avg_term_difficulty'], bins=5, color='#d62728', density=True, alpha=0.6, label='Hard Samples (Bin 4-6)', edgecolor='red', range=[0,1])
    
#     plt.axvline(df_comp_others['avg_term_difficulty'].mean(), color='gray', linestyle='--')
#     plt.axvline(df_comp_hard['avg_term_difficulty'].mean(), color='red', linestyle='--')
    
#     plt.title('Attribution: Are Hard Samples composed of Hard-to-Predict Terms?')
#     plt.xlabel('Average Historical Recall of Constituent GO Terms (Lower = Harder Terms)', fontsize=12)
#     plt.ylabel('Density')
#     plt.legend(loc='upper left')
#     plt.savefig('analysis_attrib_term_recall.png', dpi=300)
#     plt.show()
    
#     # ================= 绘图 2: 组合难度分布对比 =================
#     plt.figure(figsize=(10, 6))
    
#     # 过滤掉单标签 (NaN)
#     valid_others = df_comp_others.dropna(subset=['avg_pair_difficulty'])
#     valid_hard = df_comp_hard.dropna(subset=['avg_pair_difficulty'])
    
#     plt.hist(valid_others['avg_pair_difficulty'], bins=5, color='gray', density=True, alpha=0.3, label='Other Samples', edgecolor='gray', range=[0,1])
#     plt.hist(valid_hard['avg_pair_difficulty'], bins=5, color='#d62728', density=True, alpha=0.6, label='Hard Samples (Bin 4-6)', edgecolor='red', range=[0,1])
    
#     plt.axvline(valid_others['avg_pair_difficulty'].mean(), color='gray', linestyle='--')
#     plt.axvline(valid_hard['avg_pair_difficulty'].mean(), color='red', linestyle='--')

#     plt.title('Attribution: Are Hard Samples composed of Hard-to-Predict Pairs?')
#     plt.xlabel('Average Historical Recall of Constituent GO Pairs (Lower = Harder Combinations)', fontsize=12)
#     plt.ylabel('Density')
#     plt.legend(loc='upper left')
#     plt.savefig('analysis_attrib_pair_recall.png', dpi=300)
#     plt.show()

#     # ================= 统计输出 =================
#     print("\nAttribution Statistics (Mean Constituent Recall):")
#     print(f"{'Metric':<30} | {'Others':<10} | {'Hard (4-6)':<10} | {'Diff'}")
#     print("-" * 65)
    
#     m_term_other = df_comp_others['avg_term_difficulty'].mean()
#     m_term_hard = df_comp_hard['avg_term_difficulty'].mean()
#     print(f"{'Avg Term Recall':<30} | {m_term_other:.4f} | {m_term_hard:.4f} | {m_term_hard - m_term_other:.4f}")
    
#     m_pair_other = valid_others['avg_pair_difficulty'].mean()
#     m_pair_hard = valid_hard['avg_pair_difficulty'].mean()
#     print(f"{'Avg Pair Recall':<30} | {m_pair_other:.4f} | {m_pair_hard:.4f} | {m_pair_hard - m_pair_other:.4f}")
    
#     print("-" * 65)
#     print("Interpretation:")
#     print(" - If Diff is NEGATIVE, it means Hard Samples consist of terms/pairs that are globally hard to predict.")
#     print(" - This confirms the difficulty is intrinsic to the label vocabulary, not just the specific sample.")



# # ================= 8. [Updated] 基于标签/组合难度的模型对比 (Attribution & Comparison) =================
#     print("\n" + "="*80)
#     print("STARTING ATTRIBUTION COMPARISON: Ours vs. Reference")
#     print("="*80)

#     # 1. 定义计算 Term/Pair Recall 的通用函数
#     def calculate_granular_recalls(model_preds_map, test_info_map):
#         """
#         计算指定模型预测结果的 单标签Recall 和 标签对Recall
#         """
#         term_gt = defaultdict(int)
#         term_hit = defaultdict(int)
#         pair_gt = defaultdict(int)
#         pair_hit = defaultdict(int)
        
#         for pid, info in test_info_map.items():
#             if pid not in model_preds_map: continue
#             gt_set = info['gt']
#             preds_list = model_preds_map[pid] # List of sets (multiple generations)
#             n_preds = len(preds_list)
#             if n_preds == 0: continue
            
#             # A. Term Level
#             for go in gt_set:
#                 term_gt[go] += 1
#                 n_hits = sum(1 for p in preds_list if go in p)
#                 term_hit[go] += (n_hits / n_preds) # Instance Average Recall
                
#             # B. Pair Level
#             if len(gt_set) > 1:
#                 for pair in combinations(sorted(gt_set), 2):
#                     pair_gt[pair] += 1
#                     n_pair_hits = sum(1 for p in preds_list if (pair[0] in p and pair[1] in p))
#                     pair_hit[pair] += (n_pair_hits / n_preds)
        
#         # Calculate Final Rates
#         term_res = {k: term_hit[k]/term_gt[k] for k in term_gt if term_gt[k] > 0}
#         pair_res = {k: pair_hit[k]/pair_gt[k] for k in pair_gt if pair_gt[k] > 0}
        
#         return term_res, pair_res, term_gt # Return term_gt to filter rare terms if needed

#     # 2. 准备 Reference 模型的预测数据
#     ref_model_name = 'Reference (Ground Truth)'
#     print(f"Loading predictions for {ref_model_name}...")
#     ref_tsv = MODELS[ref_model_name]
    
#     try:
#         df_ref = pd.read_csv(ref_tsv, sep='\t', header=None, names=['raw_id', 'go_id', 'score'])
#     except:
#         df_ref = pd.read_csv(ref_tsv, sep=r'\s+', header=None, names=['raw_id', 'go_id', 'score'])
        
#     df_ref['clean_id'] = df_ref['raw_id'].apply(lambda x: x.replace('SEQUENCE_ID=', '').split('_L=')[0])
#     df_ref = df_ref[df_ref['score'] >= CONFIDENCE_THRESHOLD]
    
#     ref_preds_map = defaultdict(list)
#     for raw_id, group in df_ref.groupby('raw_id'):
#         pid = group['clean_id'].iloc[0]
#         if pid in test_info:
#             ref_preds_map[pid].append(set(group['go_id']))

#     # 3. 准备 Ours 模型的预测数据 (复用之前加载的逻辑，确保 pid_to_preds 存在)
#     # 如果上下文中 pid_to_preds (Ours) 已经有了，直接用。否则这里重新加载一次 Ours
#     if 'Ours' not in MODELS: raise ValueError("Model 'Ours' not defined in MODELS")
#     # (假设 pid_to_preds 已经是 Ours 的预测结果，如果没有请取消注释下面几行)
#     # df_ours = pd.read_csv(MODELS['Ours'], sep='\t', header=None, names=['raw_id', 'go_id', 'score'])
#     # df_ours['clean_id'] = df_ours['raw_id'].apply(lambda x: x.replace('SEQUENCE_ID=', '').split('_L=')[0])
#     # pid_to_preds = defaultdict(list)
#     # for raw_id, group in df_ours.groupby('raw_id'):
#     #     pid = group['clean_id'].iloc[0]
#     #     if pid in test_info:
#     #         pid_to_preds[pid].append(set(group['go_id']))

#     # 4. 计算指标
#     print("Calculating Recall Maps for Ours...")
#     term_rec_ours, pair_rec_ours, term_counts = calculate_granular_recalls(pid_to_preds, test_info)
    
#     print("Calculating Recall Maps for Reference...")
#     term_rec_ref, pair_rec_ref, _ = calculate_granular_recalls(ref_preds_map, test_info)

#     # 5. 对比分析 (Intersection)
#     common_terms = set(term_rec_ours.keys()) & set(term_rec_ref.keys())
#     print(f"Analyzing {len(common_terms)} common GO terms appearing in Test Set GT.")
    
#     comparison_data = []
#     for go in common_terms:
#         comparison_data.append({
#             'go_id': go,
#             'recall_ours': term_rec_ours[go],
#             'recall_ref': term_rec_ref[go],
#             'count': term_counts[go]
#         })
#     df_comp_term = pd.DataFrame(comparison_data)

#     # 6. 绘图：Term Recall Correlation
#     plt.figure(figsize=(8, 8))
#     # 使用 Hexbin 或 Scatter
#     plt.scatter(df_comp_term['recall_ref'], df_comp_term['recall_ours'], alpha=0.1, color='purple', s=10)
    
#     # 添加对角线 (y=x)
#     plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Equal Performance')
    
#     # 拟合线
#     m, b = np.polyfit(df_comp_term['recall_ref'], df_comp_term['recall_ours'], 1)
#     plt.plot(df_comp_term['recall_ref'], m*df_comp_term['recall_ref'] + b, color='red', label=f'Fit: y={m:.2f}x+{b:.2f}')
    
#     plt.title('Difficulty Consistency: Term Recall (Ours vs Reference)', fontsize=14)
#     plt.xlabel('Recall of Reference Model (Ground Truth)', fontsize=12)
#     plt.ylabel('Recall of Ours Model', fontsize=12)
#     plt.legend()
#     plt.grid(True, linestyle=':', alpha=0.5)
#     plt.savefig('analysis_compare_term_recall_scatter.png', dpi=300)
#     plt.show()
    
#     corr_term = df_comp_term['recall_ours'].corr(df_comp_term['recall_ref'])
#     print(f"Term Recall Correlation (Ours vs Ref): {corr_term:.4f}")
#     if corr_term > 0.5:
#         print("  -> Conclusion: High correlation. Hard terms are hard for BOTH models (Intrinsic Difficulty).")
#     else:
#         print("  -> Conclusion: Low correlation. Models struggle with DIFFERENT terms.")

#     # ================= 9. [Rewrite] 样本平均深度与性能关系 (Sample Depth vs Performance) =================
#     print("\n" + "="*80)
#     print("STARTING SAMPLE DEPTH ANALYSIS (Ours Model)")
#     print("="*80)
    
#     sample_depth_data = []
    
#     # 预计算每个 Term 的深度，避免循环中重复计算
#     # 获取所有涉及的 GO term
#     all_involved_gos = set()
#     for info in test_info.values():
#         all_involved_gos.update(info['gt'])
    
#     print(f"Pre-calculating depths for {len(all_involved_gos)} terms...")
#     go_depth_map = {}
#     for go in all_involved_gos:
#         go_depth_map[go] = ontology.get_depth(go)

#     # 分析每个样本
#     for pid, info in test_info.items():
#         if pid not in pid_to_preds: continue
        
#         gt_set = info['gt']
#         if not gt_set: continue
        
#         # 1. 计算该样本的 Average Depth
#         # (深度越深，说明该蛋白的功能标注越具体/细粒度)
#         depths = [go_depth_map.get(go, 0) for go in gt_set]
#         avg_sample_depth = np.mean(depths)

#         s = set()
#         for go in gt_set:
#             s.update(ontology.get_ancestors(go))
#         gt_set = s


#         # 2. 获取 Ours 模型的 F1 和 Recall
#         preds_list = pid_to_preds[pid]
#         f1s = []
#         recs = []
#         for p_set in preds_list:
#             inters = len(gt_set & p_set)
#             f1 = 2 * inters / (len(gt_set) + len(p_set)) if (len(gt_set) + len(p_set)) > 0 else 0
#             rec = inters / len(gt_set) if len(gt_set) > 0 else 0
#             f1s.append(f1)
#             recs.append(rec)
            
#         sample_depth_data.append({
#             'pid': pid,
#             'avg_depth': avg_sample_depth,
#             'f1': np.mean(f1s),
#             'recall': np.mean(recs)
#         })
        
#     df_sample_depth = pd.DataFrame(sample_depth_data)
    
#     # --- 绘图: F1 & Recall 随 平均深度 的变化趋势 ---
#     print("Generating Plot: Sample Performance vs. Average Depth...")
    
#     plt.figure(figsize=(10, 6))
    
#     # 为了看清趋势，将深度进行 Round 或 Binning
#     # 大多数深度在 2-12 之间
#     df_sample_depth['depth_bin'] = df_sample_depth['avg_depth'].round(0) # Round to nearest integer
    
#     stats_depth = df_sample_depth.groupby('depth_bin')[['f1', 'recall']].mean().reset_index()
#     # 过滤掉样本极少的极端深度 (比如 depth > 12)
#     stats_depth = stats_depth[stats_depth['depth_bin'] <= 14]
    
#     # 双线图
#     plt.plot(stats_depth['depth_bin'], stats_depth['f1'], marker='o', label='Avg F1 Score', linewidth=2, color='#1f77b4')
#     plt.plot(stats_depth['depth_bin'], stats_depth['recall'], marker='^', label='Avg Recall', linewidth=2, color='#2ca02c')
    
#     # 背景加散点 (可选，展示分布密度)
#     # plt.scatter(df_sample_depth['avg_depth'], df_sample_depth['f1'], alpha=0.05, color='gray', s=5)
    
#     plt.title('Impact of Annotation Specificity (Avg Depth) on Model Performance', fontsize=14)
#     plt.xlabel('Average GO Term Depth of Sample (Higher = More Specific Function)', fontsize=12)
#     plt.ylabel('Score', fontsize=12)
#     plt.legend()
#     plt.grid(True, linestyle=':', alpha=0.5)
    
#     # 添加样本数量分布在双轴
#     ax2 = plt.gca().twinx()
#     counts = df_sample_depth.groupby('depth_bin')['pid'].count()
#     # 这里的counts需要和stats_depth对齐
#     counts = counts.reindex(stats_depth['depth_bin']).fillna(0)
#     ax2.bar(stats_depth['depth_bin'], counts, color='gray', alpha=0.15, label='Sample Count')
#     ax2.set_ylabel('Number of Samples')
    
#     plt.tight_layout()
#     plt.savefig('analysis_sample_depth_performance.png', dpi=300)
#     plt.show()

#     # 统计相关性
#     corr_depth_f1 = df_sample_depth['avg_depth'].corr(df_sample_depth['f1'])
#     print(f"\nCorrelation (Avg Depth vs F1): {corr_depth_f1:.4f}")
    
#     if corr_depth_f1 < -0.1:
#         print("  -> Trend: Performance drops for deeper (more specific) samples.")
#     elif corr_depth_f1 > 0.1:
#         print("  -> Trend: Performance improves for deeper samples (Model is good at specific terms).")
#     else:
#         print("  -> Trend: Performance is stable across different specificity levels.")

#     # 保存数据
#     df_sample_depth.to_csv('analysis_sample_depth_metrics.csv', index=False)

# # ... (接在 Step 8 的 correlation 计算代码之后) ...
    
#     # ================= [新增] Zero-Recall Overlap Analysis =================
#     print("\n" + "-"*40)
#     print("Zero-Recall Overlap Analysis (Hard Failure Check)")
#     print("-" * 40)
    
#     # 1. 筛选出 Ours 完全没预测对的词 (Recall_Ours == 0)
#     # 注意：df_comp_term 是我们在上面刚刚生成的 DataFrame
#     subset_ours_zero = df_comp_term[df_comp_term['recall_ours'] == 0.0]
#     n_ours_zero = len(subset_ours_zero)
    
#     # 2. 在这部分词中，筛选出 Reference 也完全没预测对的词 (Recall_Ref == 0)
#     subset_both_zero = subset_ours_zero[subset_ours_zero['recall_ref'] == 0.0]
#     n_both_zero = len(subset_both_zero)
    
#     # 3. 统计 Reference 成功但 Ours 失败的 (Ref > 0, Ours == 0)
#     # 这部分是我们需要重点反思的 "Missed Opportunities"
#     subset_ref_success_ours_fail = subset_ours_zero[subset_ours_zero['recall_ref'] > 0.0]
#     n_ref_success = len(subset_ref_success_ours_fail)

#     # 4. 输出统计
#     print(f"Total Terms Analyzed (Common in Test GT): {len(df_comp_term)}")
#     print(f"Terms with Recall=0 in 'Ours':            {n_ours_zero}")
    
#     if n_ours_zero > 0:
#         ratio_both_zero = (n_both_zero / n_ours_zero) * 100
#         ratio_ref_success = (n_ref_success / n_ours_zero) * 100
        
#         print(f"\nOf the {n_ours_zero} terms that 'Ours' completely missed:")
#         print(f"  A. Both Missed (Ref Recall=0):      {n_both_zero} ({ratio_both_zero:.2f}%)")
#         print(f"     -> Interpretation: Intrinsically Hard. Even the Reference failed.")
        
#         print(f"  B. Ref Found (Ref Recall>0):        {n_ref_success} ({ratio_ref_success:.2f}%)")
#         print(f"     -> Interpretation: Model Weakness. Reference found them, but we missed.")
        
#         # 额外：看看在情况 B 中，Reference 是找回了一点点，还是找回得很完美？
#         if n_ref_success > 0:
#             avg_ref_recall_in_missed = subset_ref_success_ours_fail['recall_ref'].mean()
#             print(f"     -> Avg Recall of Reference on these specific terms: {avg_ref_recall_in_missed:.4f}")
    
#     # 5. 保存详细列表供检查
#     # 我们把 Ours=0 但 Ref>0 的词存下来，这是最有价值的 Error Analysis 数据
#     if n_ref_success > 0:
#         out_csv_missed = 'analysis_terms_ours_missed_ref_found.csv'
#         subset_ref_success_ours_fail.sort_values(by='recall_ref', ascending=False).to_csv(out_csv_missed, index=False)
#         print(f"\nSaved list of {n_ref_success} terms (Ours=0, Ref>0) to '{out_csv_missed}'")


# if __name__ == '__main__':
#     main()

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, recall_score
from src.byprot.utils.ontology import Ontology

# ================= Configuration =================
# 1. Paths
TEST_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
GO_MAPPING_PATH = 'go_mapping.pkl'
GO_OBO_PATH = 'data/go.obo'

# 2. Models to Compare
MODELS = {
    'Ours': './generation-results-dplm2-goonly-alldata-dm-ca-me-scale-0.2-weight-headclloss-2.0_sn-pn-11wstep/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut_preds_mf.tsv',
    'Baseline': './generation-results-cfpgen_650m/cfpgen_650m_go_preds_mf.tsv', 
    'Reference': '/AIRvePFS/dair/chenxr-data/repo/cfpgen/data-bin/uniprotKB/cfpgen_general_dataset/test_preds_mf.tsv' # Optional
}

# 3. Binning Settings
# Semantic Difficulty Bins: [0, 2), [2, 4), ...
BINS = [0, 1,2, 3,4, 5,6, 7,8, 9,10, 11,12]
CONFIDENCE_THRESHOLD = 0.0

# 4. Plot Styling (ICML)
plt.rcParams.update({
    'font.family': 'serif',
    # 'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color Palette: Ours (Blue), Baseline (Gray/Red)
COLORS = {'Ours': '#1f77b4', 'Baseline': '#d62728', 'Reference': 'gray'}

# ================= Helper Functions =================

def load_data_and_ontology():
    print("Loading Ontology...")
    ontology = Ontology(GO_OBO_PATH, with_rels=True)
    
    print("Loading Mappings...")
    with open(GO_MAPPING_PATH, 'rb') as f:
        index_to_go = {v: k for k, v in pickle.load(f).items()}
        
    print("Loading Test Data...")
    with open(TEST_PATH, 'rb') as f:
        test_data = pickle.load(f)
        
    return ontology, index_to_go, test_data

def compute_metrics_official(gt_list, pred_list, ontology):
    if len(gt_list) == 0: return None
    
    # 1. Propagate
    expanded_gt = []
    for gos in gt_list:
        s = set()
        for go in gos: s.update(ontology.get_ancestors(go))
        expanded_gt.append(s)
    
    expanded_pred = []
    for gos in pred_list:
        s = set()
        for go in gos: s.update(ontology.get_ancestors(go))
        s.update(gos)
        expanded_pred.append(s)

    # 2. Intersection Filter
    u_gt = set().union(*expanded_gt)
    u_pred = set().union(*expanded_pred)
    unique_go = u_gt & u_pred
    
    if len(unique_go) == 0:
        return {'f1_mic': 0.0, 'f1_mac': 0.0, 'rec_mic': 0.0, 'rec_mac': 0.0}

    final_pred = [[e for e in s if e in unique_go] for s in expanded_pred]
    final_gt = [[e for e in s if e in unique_go] for s in expanded_gt]

    # 3. Metrics
    mlb = MultiLabelBinarizer()
    mlb.fit(final_gt + final_pred)
    y_true = mlb.transform(final_gt)
    y_pred = mlb.transform(final_pred)

    return {
        'f1_mic': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_mac': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'rec_mic': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'rec_mac': recall_score(y_true, y_pred, average='macro', zero_division=0)
    }

def load_predictions(path):
    print(f"Loading {path}...")
    try:
        df = pd.read_csv(path, sep='\t', header=None, names=['raw_id', 'go_id', 'score'])
    except:
        df = pd.read_csv(path, sep=r'\s+', header=None, names=['raw_id', 'go_id', 'score'])
    
    def get_clean_id(raw):
        if 'SEQUENCE_ID=' in raw:
            return raw.replace('SEQUENCE_ID=', '').split('_L=')[0]
        return raw 
    df['clean_id'] = df['raw_id'].apply(get_clean_id)
    df = df[df['score'] >= CONFIDENCE_THRESHOLD]
    
    pid_to_preds = defaultdict(list)
    for raw_id, group in df.groupby('raw_id'):
        pid = group['clean_id'].iloc[0]
        pid_to_preds[pid].append(set(group['go_id']))
    return pid_to_preds

def plot_comparison(all_results, metric_key_micro, metric_key_macro, y_label, title, filename, bin_labels):
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(bin_labels))
    
    for name, df in all_results.items():
        color = COLORS.get(name, 'black')
        
        # Align data
        df_sorted = df.set_index('bin_label').reindex(bin_labels).reset_index()
        mask = df_sorted[metric_key_micro].notna()
        
        # Plot Micro (Solid)
        plt.plot(x[mask], df_sorted.loc[mask, metric_key_micro], 
                 marker='o', markersize=6, linestyle='-', linewidth=2, 
                 color=color, label=f'{name} (Micro)')
        
        # Plot Macro (Dashed)
        plt.plot(x[mask], df_sorted.loc[mask, metric_key_macro], 
                 marker='^', markersize=6, linestyle='--', linewidth=1.5, alpha=0.7,
                 color=color, label=f'{name} (Macro)')

    plt.title(title, pad=15, fontweight='bold')
    plt.xlabel('Semantic Difficulty (Mean Intra-Set Distance)', fontweight='bold')
    plt.ylabel(y_label, fontweight='bold')
    plt.xticks(x, bin_labels)
    plt.ylim(0, 1.0)
    
    # Legend layout
    plt.legend(loc='lower left', frameon=True, framealpha=0.9, ncol=2)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filename}")
    plt.show()

# ================= Main =================

def main():
    ontology, index_to_go, test_data = load_data_and_ontology()
    
    # 1. Calculate Difficulty for Test Set
    print("Calculating Test Set Difficulty...")
    test_info = {}
    for entry in tqdm(test_data):
        pid = entry['uniprot_id']
        gt_gos = [index_to_go[i] for i in entry['go_f_mapped']]
        diff = ontology.calculate_set_difficulty(gt_gos)
        test_info[pid] = {'gt': set(gt_gos), 'difficulty': diff}
        
    # 2. Define Bins
    bin_labels = []
    for i in range(len(BINS) - 1):
        low, high = BINS[i], BINS[i+1]
        label = f"{low}-{high}" if high < 100 else f">{low}"
        bin_labels.append(label)
        
    # 3. Process Each Model
    all_model_results = {}
    
    for model_name, path in MODELS.items():
        preds_map = load_predictions(path)
        
        bin_res = []
        for i, label in enumerate(bin_labels):
            low, high = BINS[i], BINS[i+1]
            
            target_pids = [p for p, info in test_info.items() if low <= info['difficulty'] < high]
            if not target_pids: continue
            
            batch_gt = []
            batch_pred = []
            
            for pid in target_pids:
                gt_set = test_info[pid]['gt']
                # Get all generations for this PID
                p_sets = preds_map.get(pid, [])
                
                if not p_sets:
                    batch_gt.append(gt_set)
                    batch_pred.append(set())
                else:
                    for p_set in p_sets:
                        batch_gt.append(gt_set)
                        batch_pred.append(p_set)
            
            scores = compute_metrics_official(batch_gt, batch_pred, ontology)
            if scores:
                scores['bin_label'] = label
                bin_res.append(scores)
        
        all_model_results[model_name] = pd.DataFrame(bin_res)
        
    # 4. Plotting
    # Plot F1
    plot_comparison(all_model_results, 'f1_mic', 'f1_mac', 'F1 Score', 
                    'Performance vs. Semantic Difficulty (F1)', 
                    'comparison_f1_vs_difficulty.png', bin_labels)
    
    # Plot Recall
    plot_comparison(all_model_results, 'rec_mic', 'rec_mac', 'Recall', 
                    'Performance vs. Semantic Difficulty (Recall)', 
                    'comparison_recall_vs_difficulty.png', bin_labels)

if __name__ == '__main__':
    main()