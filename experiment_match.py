import pandas as pd
import pickle
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import os
import networkx as nx
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score
from src.byprot.utils.ontology import Ontology


# ================= Configuration =================
# 1. Paths
# TSV_PATH = './generation-results-cfpgen_650m_unseen/cfpgen_650m_go_nondup_preds_mf.tsv'
TSV_PATH = './generation-results-dplm2-goonly-unseen-all/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut_nondup_preds_mf.tsv'

# TSV_PATH = './generation-results-cfpgen_650m/cfpgen_650m_go_preds_mf.tsv'
# TSV_PATH = './generation-results-dplm2-goonly-alldata-dm-ca-me-scale-0.2-weight-headclloss-2.0_sn-pn-11wstep/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut_preds_mf.tsv'
TRAIN_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
TEST_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
GO_MAPPING_PATH = 'go_mapping.pkl'
GO_OBO_PATH = 'data/go.obo' 

# 2. Threshold
CONFIDENCE_THRESHOLD = 0.0

# ================= 2. Data Loading & Indexing =================
print("Loading data and building indexes...")

# TODO all metrics cal

# Load Ontology
ontology = Ontology(GO_OBO_PATH)

with open(GO_MAPPING_PATH, 'rb') as f:
    go_mapping = pickle.load(f)
index_to_go = {v: k for k, v in go_mapping.items()}

# Load Train for Strict Unseen Logic
with open(TRAIN_PATH, 'rb') as f:
    train_data = pickle.load(f)

go_to_train_indices = defaultdict(set)
train_combos_set = set()
valid_train_gos = set()

for idx, entry in enumerate(train_data):
    go_ids = [index_to_go[i] for i in entry['go_f_mapped']]
    combo = tuple(sorted(go_ids))
    train_combos_set.add(combo)
    for go_id in go_ids:
        go_to_train_indices[go_id].add(idx)
        valid_train_gos.add(go_id)

def is_strict_unseen_combo(go_list):
    """Determine if a GO list is Strictly Unseen relative to training set"""
    if not go_list: return False
    if tuple(sorted(go_list)) in train_combos_set: return False
    sets_to_intersect = [go_to_train_indices[go_id] for go_id in go_list if go_id in go_to_train_indices]
    if len(sets_to_intersect) != len(go_list): return True 
    if len(set.intersection(*sets_to_intersect)) > 0: return False 
    return True

# ================= 3. Identify Targets =================
print("Identifying Strictly Unseen targets in Test Data...")

with open(TEST_PATH, 'rb') as f:
    test_data = pickle.load(f)

strict_unseen_targets = {} 

# for entry in test_data:
#     pid = entry['uniprot_id']
#     gt_go_ids = [index_to_go[i] for i in entry['go_f_mapped']]
    
#     # Logic: Must be Unseen Combo AND composed of Known Atoms
#     if not is_strict_unseen_combo(gt_go_ids):
#         continue
#     if not all(go_id in valid_train_gos for go_id in gt_go_ids):
#         continue

#     strict_unseen_targets[pid] = set(gt_go_ids)

# print(f"Found {len(strict_unseen_targets)} Strictly Unseen Targets.")

strict_unseen_targets = {}
seen_gt_combos = set()  # 用于记录已经收录过的 GT 组合

for entry in test_data:
    pid = entry['uniprot_id']
    gt_go_ids = [index_to_go[i] for i in entry['go_f_mapped']]
    
    # Logic: Must be Unseen Combo AND composed of Known Atoms
    if not is_strict_unseen_combo(gt_go_ids):
        continue
    if not all(go_id in valid_train_gos for go_id in gt_go_ids):
        continue

    # --- Filter: Keep only 1 unique GT combination ---
    # 将列表转换为 frozenset 以便作为 hash key
    gt_key = frozenset(gt_go_ids)
    
    # 如果这个组合之前没出现过，才添加
    if gt_key not in seen_gt_combos:
        strict_unseen_targets[pid] = set(gt_go_ids)
        seen_gt_combos.add(gt_key)  # 标记为已收录

print(f"Found {len(strict_unseen_targets)} Strictly Unseen Targets (Unique GT combinations).")

# ================= 4. Load Predictions =================
print(f"Loading predictions from {TSV_PATH}...")

try:
    df = pd.read_csv(TSV_PATH, sep='\t', header=None, names=['raw_id', 'go_id', 'score'])
except:
    df = pd.read_csv(TSV_PATH, sep=r'\s+', header=None, names=['raw_id', 'go_id', 'score'])

# Extract clean UniProt ID
def extract_uniprot_id(raw_id):
    # SEQUENCE_ID=P04424_L=200_180 -> P04424
    if 'SEQUENCE_ID=' in raw_id:
        temp = raw_id.replace('SEQUENCE_ID=', '')
        return temp.split('_L=')[0]
    return raw_id

df['uniprot_id'] = df['raw_id'].apply(extract_uniprot_id)
df = df[df['score'] >= CONFIDENCE_THRESHOLD]

# Group by Raw Instance ID (to handle repeats _180, _181...)
instance_groups = df.groupby('raw_id')

# ================= 5. Compute Metrics (Official Logic) =================
print("Computing metrics using Ancestor Propagation...")

all_gt_propagated = []
all_pred_propagated = []
all_raw_ids = []
all_pids = []

# 1. Collect and Propagate Data
all_runs = defaultdict(lambda: {'gt': [], 'pred': [], 'raw_id': [], 'pid': []})

for raw_id, group in tqdm(instance_groups, desc="Propagating"):
    pid = group['uniprot_id'].iloc[0]
    
    if pid in strict_unseen_targets:
        # Extract repeat index
        repeat_idx = raw_id.split('_')[-1] if 'SEQUENCE_ID=' in raw_id else '0'
        
        # Get raw sets
        gt_set_raw = strict_unseen_targets[pid]
        pred_set_raw = set(group['go_id'])
        
        # Propagate Ancestors
        gt_prop = set()
        for go in gt_set_raw:
            gt_prop.update(ontology.get_ancestors(go))
            
        pred_prop = set()
        # for go in pred_set_raw:
        #     pred_prop.update(ontology.get_ancestors(go))
        pred_prop = pred_set_raw
            
        all_gt_propagated.append(gt_prop)
        all_pred_propagated.append(pred_prop)
        all_raw_ids.append(raw_id)
        all_pids.append(pid)
        
        all_runs[repeat_idx]['gt'].append(gt_prop)
        all_runs[repeat_idx]['pred'].append(pred_prop)
    
    

# 2. Intersection Filter (The "Official" Logic)
# Calculate Unique GOs across ALL valid samples
unique_go_gt = set().union(*all_gt_propagated)
unique_go_pred = set().union(*all_pred_propagated)
unique_go = unique_go_gt & unique_go_pred

print(f"Total Unique GOs in intersection space: {len(unique_go)}")

# Filter lists by this intersection
final_gt_list = []
final_pred_list = []

for gt_s, pred_s in zip(all_gt_propagated, all_pred_propagated):
    final_gt_list.append([go for go in gt_s if go in unique_go])
    final_pred_list.append([go for go in pred_s if go in unique_go])

# 3. Calculate Metrics via MultiLabelBinarizer
mlb = MultiLabelBinarizer()
mlb.fit(final_gt_list + final_pred_list)

y_true = mlb.transform(final_gt_list)
y_pred = mlb.transform(final_pred_list)

# Global Metrics (over all runs flattened)
f1_mic_all = f1_score(y_true, y_pred, average='micro', zero_division=0)
f1_mac_all = f1_score(y_true, y_pred, average='macro', zero_division=0)
rec_mic_all = recall_score(y_true, y_pred, average='micro', zero_division=0)
prec_mic_all = precision_score(y_true, y_pred, average='micro', zero_division=0)

# 4. Calculate Metrics Per Run
run_metrics = {'f1_mic': [], 'f1_mac': [], 'rec_mic': [], 'prec_mic': []}

merged_runs = defaultdict(lambda: {'gt': [], 'pred': [], 'raw_id': [], 'pid': []})

# for i in range(0, 10, 2):
#     k1 = str(i)
#     k2 = str(i+1)
#     new_key = f"{k1}_{k2}" # 新的 key 会变成 '0_1', '2_3' 等
    
#     # 将两个 run 的列表直接相加（拼接）
#     merged_runs[new_key]['gt'] = all_runs[k1]['gt'] + all_runs[k2]['gt']
#     merged_runs[new_key]['pred'] = all_runs[k1]['pred'] + all_runs[k2]['pred']
#     merged_runs[new_key]['raw_id'] = all_runs[k1]['raw_id'] + all_runs[k2]['raw_id']
#     merged_runs[new_key]['pid'] = all_runs[k1]['pid'] + all_runs[k2]['pid']

# for run_idx, run_data in merged_runs.items():

#     print(run_data)

#     unique_go = set().union(*run_data['gt']) & set().union(*run_data['pred'])

#     run_gt_list = [[go for go in gt_s if go in unique_go] for gt_s in run_data['gt']]
#     run_pred_list = [[go for go in pred_s if go in unique_go] for pred_s in run_data['pred']]

#     # print(run_gt_list)
#     # print(run_pred_list)

#     mlb = MultiLabelBinarizer()
#     mlb.fit(run_gt_list + run_pred_list)
    
#     y_true_run = mlb.transform(run_gt_list)
#     y_pred_run = mlb.transform(run_pred_list)
    
#     run_metrics['f1_mic'].append(f1_score(y_true_run, y_pred_run, average='micro', zero_division=0))
#     run_metrics['f1_mac'].append(f1_score(y_true_run, y_pred_run, average='macro', zero_division=0))
#     run_metrics['rec_mic'].append(recall_score(y_true_run, y_pred_run, average='micro', zero_division=0))
#     run_metrics['prec_mic'].append(precision_score(y_true_run, y_pred_run, average='micro', zero_division=0))

f1_mic_mean, f1_mic_std = np.mean(run_metrics['f1_mic']), np.std(run_metrics['f1_mic'])
f1_mac_mean, f1_mac_std = np.mean(run_metrics['f1_mac']), np.std(run_metrics['f1_mac'])
rec_mic_mean, rec_mic_std = np.mean(run_metrics['rec_mic']), np.std(run_metrics['rec_mic'])
prec_mic_mean, prec_mic_std = np.mean(run_metrics['prec_mic']), np.std(run_metrics['prec_mic'])

print("\n" + "="*60)
print("METRICS REPORT (Official Logic: Propagation + Intersection)")
print("="*60)
print(f"Overall (Flattened):")
print(f"Micro F1:        {f1_mic_all:.4f}")
print(f"Macro F1:        {f1_mac_all:.4f}")
print(f"Micro Recall:    {rec_mic_all:.4f}")
print(f"Micro Precision: {prec_mic_all:.4f}")
# print("\nPer Run (Mean ± Std):")
# print(f"Micro F1:        {f1_mic_mean:.4f} ± {f1_mic_std:.4f}")
# print(f"Macro F1:        {f1_mac_mean:.4f} ± {f1_mac_std:.4f}")
# print(f"Micro Recall:    {rec_mic_mean:.4f} ± {rec_mic_std:.4f}")
# print(f"Micro Precision: {prec_mic_mean:.4f} ± {prec_mic_std:.4f}")

# ================= 6. Identify Exact Matches =================
# An "Exact Match" in this context is defined as a sample where
# the set of propagated GO terms (filtered by intersection) matches exactly.
# i.e., F1 Score for that specific sample == 1.0

print("\nSearching for EXACT MATCHES ...")

exact_matches = []

gt_combo_stats = defaultdict(lambda: {'total': 0, 'exact': 0})

evaluated_count = 0

for raw_id, group in tqdm(instance_groups, desc="Propagating"):
    pid = group['uniprot_id'].iloc[0]

    if pid not in strict_unseen_targets:
        continue

    # ----- Raw GT / Pred -----
    gt_raw = set(strict_unseen_targets[pid])          # NO propagation
    pred_raw = set(group['go_id'])

    # ----- Intersection Space Filtering -----
    gt_filt = {go for go in gt_raw if go in unique_go}
    pred_filt = {go for go in pred_raw if go in unique_go}

    # Skip degenerate cases
    if len(gt_filt) == 0:
        continue

    # ----- Exact Match Logic: GT ⊆ Pred -----
    if gt_filt.issubset(pred_filt):
        exact_matches.append({
            'raw_id': raw_id,
            'uniprot_id': pid,
            'gt': gt_filt,
            'pred': pred_filt
        })

print(f"Found {len(exact_matches)} Exact Match Instances out of {len(y_true)} evaluated.")


