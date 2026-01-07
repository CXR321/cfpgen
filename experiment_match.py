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

for entry in test_data:
    pid = entry['uniprot_id']
    gt_go_ids = [index_to_go[i] for i in entry['go_f_mapped']]
    
    # Logic: Must be Unseen Combo AND composed of Known Atoms
    if not is_strict_unseen_combo(gt_go_ids):
        continue
    if not all(go_id in valid_train_gos for go_id in gt_go_ids):
        continue

    strict_unseen_targets[pid] = set(gt_go_ids)

print(f"Found {len(strict_unseen_targets)} Strictly Unseen Targets.")

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
for raw_id, group in tqdm(instance_groups, desc="Propagating"):
    pid = group['uniprot_id'].iloc[0]
    
    if pid in strict_unseen_targets:
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

# Global Metrics
f1_mic = f1_score(y_true, y_pred, average='micro', zero_division=0)
f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
rec_mic = recall_score(y_true, y_pred, average='micro', zero_division=0)
prec_mic = precision_score(y_true, y_pred, average='micro', zero_division=0)

print("\n" + "="*60)
print("METRICS REPORT (Official Logic: Propagation + Intersection)")
print("="*60)
print(f"Micro F1:        {f1_mic:.4f}")
print(f"Macro F1:        {f1_mac:.4f}")
print(f"Micro Recall:    {rec_mic:.4f}")
print(f"Micro Precision: {prec_mic:.4f}")

# ================= 6. Identify Exact Matches =================
# An "Exact Match" in this context is defined as a sample where
# the set of propagated GO terms (filtered by intersection) matches exactly.
# i.e., F1 Score for that specific sample == 1.0

print("\nSearching for EXACT MATCHES (Sample F1 = 1.0)...")

exact_matches = []

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
