import os
import sys
import pandas as pd
import pickle
import numpy as np
from collections import defaultdict

sys.path.insert(0, 'src/byprot/utils')
from ontology import Ontology

# Configuration
TEST_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
TRAIN_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
GO_MAPPING_PATH = 'go_mapping.pkl'
GO_OBO_PATH = 'data/go.obo' 

BASE_TSV = './generation-results-cfpgen_650m_unseen/cfpgen_650m_go_nondup_preds_mf.tsv'
BASE_PDB_DIR = './generation-results-cfpgen_650m_unseen/esmfold_pdb'

MODEL_TSV = './generation-results-dplm2-goonly-unseen-all/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut_nondup_preds_mf.tsv'
MODEL_PDB_DIR = './generation-results-dplm2-goonly-unseen-all/esmfold_pdb'

# Load Ontology
ontology = Ontology(GO_OBO_PATH)
with open(GO_MAPPING_PATH, 'rb') as f:
    go_mapping = pickle.load(f)
index_to_go = {v: k for k, v in go_mapping.items()}

# Load Train
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
    if not go_list: return False
    if tuple(sorted(go_list)) in train_combos_set: return False
    sets_to_intersect = [go_to_train_indices[go_id] for go_id in go_list if go_id in go_to_train_indices]
    if len(sets_to_intersect) != len(go_list): return True 
    if len(set.intersection(*sets_to_intersect)) > 0: return False 
    return True

# Load Test Targets
with open(TEST_PATH, 'rb') as f:
    test_data = pickle.load(f)

strict_unseen_targets = {}
seen_gt_combos = set()
for entry in test_data:
    pid = entry['uniprot_id']
    gt_go_ids = [index_to_go[i] for i in entry['go_f_mapped']]
    if not is_strict_unseen_combo(gt_go_ids):
        continue
    if not all(go_id in valid_train_gos for go_id in gt_go_ids):
        continue
    gt_key = frozenset(gt_go_ids)
    if gt_key not in seen_gt_combos:
        strict_unseen_targets[pid] = set(gt_go_ids)
        seen_gt_combos.add(gt_key)

print(f"Loaded {len(strict_unseen_targets)} test targets.")

def extract_uniprot_id(raw_id):
    if 'SEQUENCE_ID=' in raw_id:
        temp = raw_id.replace('SEQUENCE_ID=', '')
        return temp.split('_L=')[0]
    return raw_id

def get_struct_metrics_mapping(pdb_dir):
    mapping = {}
    if not os.path.exists(pdb_dir):
        return mapping
    for f in os.listdir(pdb_dir):
        if f.endswith('.pdb') and 'plddt_' in f:
            raw_id = f.split('_plddt_')[0]
            try:
                plddt = float(f.split('_plddt_')[1].split('_')[0])
                ptm = float(f.split('_ptm_')[1].split('.pdb')[0])
                mapping[raw_id] = {'plddt': plddt, 'ptm': ptm}
            except:
                pass
    return mapping

def analyze_predictions(tsv_path, pdb_dir, name):
    print(f"\n--- Analyzing {name} ---")
    metrics_map = get_struct_metrics_mapping(pdb_dir)
    
    df = pd.read_csv(tsv_path, sep='\t', header=None, names=['raw_id', 'go_id', 'score'])
    df['uniprot_id'] = df['raw_id'].apply(extract_uniprot_id)
    instance_groups = df.groupby('raw_id')
    
    success_plddts = []
    fail_plddts = []
    success_ptms = []
    fail_ptms = []
    
    fail_cases = []
    
    for raw_id, group in instance_groups:
        pid = group['uniprot_id'].iloc[0]
        if pid not in strict_unseen_targets:
            continue
            
        gt_set_raw = strict_unseen_targets[pid]
        pred_set_raw = set(group['go_id'])
        
        gt_prop = set()
        for go in gt_set_raw:
            # gt_prop.update(ontology.get_ancestors(go))
            gt_prop.update(set([go]))
            
        pred_prop = pred_set_raw
        
        intersect = gt_prop & pred_prop
        
        if raw_id in metrics_map:
            plddt = metrics_map[raw_id]['plddt']
            ptm = metrics_map[raw_id]['ptm']
            if len(intersect) > 0:
                success_plddts.append(plddt)
                success_ptms.append(ptm)
            else:
                fail_plddts.append(plddt)
                fail_ptms.append(ptm)
                fail_cases.append({
                    'raw_id': raw_id,
                    'pid': pid,
                    'plddt': plddt,
                    'ptm': ptm,
                    'gt_prop': list(gt_prop),
                    'pred_prop': list(pred_prop)
                })
                
    print(f"Total evaluated instances: {len(success_plddts) + len(fail_plddts)}")
    print(f"Success cases (at least 1 correct GO): {len(success_plddts)}")
    print(f"Fail cases (0 correct GOs): {len(fail_plddts)}")
    
    if success_plddts:
        success_arr = np.array(success_plddts)
        success_pct_above_70 = np.mean(success_arr > 70) * 100
        print(f"Success Avg pLDDT: {np.mean(success_plddts):.2f} ± {np.std(success_plddts):.2f}")
        print(f"Success pLDDT > 70: {success_pct_above_70:.2f}%")
        if success_ptms:
            success_ptm_arr = np.array(success_ptms)
            success_ptm_pct_above_0_5 = np.mean(success_ptm_arr > 0.5) * 100
            print(f"Success Avg pTM:   {np.mean(success_ptms):.3f} ± {np.std(success_ptms):.3f}")
            print(f"Success pTM > 0.5: {success_ptm_pct_above_0_5:.2f}%")

    if fail_plddts:
        fail_arr = np.array(fail_plddts)
        fail_pct_above_70 = np.mean(fail_arr > 70) * 100
        print(f"Fail Avg pLDDT: {np.mean(fail_plddts):.2f} ± {np.std(fail_plddts):.2f}")
        print(f"Fail pLDDT > 70: {fail_pct_above_70:.2f}%")
        if fail_ptms:
            fail_ptm_arr = np.array(fail_ptms)
            fail_ptm_pct_above_0_5 = np.mean(fail_ptm_arr > 0.5) * 100
            print(f"Fail Avg pTM:   {np.mean(fail_ptms):.3f} ± {np.std(fail_ptms):.3f}")
            print(f"Fail pTM > 0.5: {fail_ptm_pct_above_0_5:.2f}%")
        
    return fail_cases

print("1. Baseline Analysis")
base_fail_cases = analyze_predictions(BASE_TSV, BASE_PDB_DIR, "Baseline (cfpgen_650m)")

print("\n2. Model Analysis")
model_fail_cases = analyze_predictions(MODEL_TSV, MODEL_PDB_DIR, "Model (dplm2_goonly)")

print("\n3. Exploring Model Fail Cases (High pLDDT but failed)")
# Sort fail cases by plddt descending to see high confidence but wrong prediction
model_fail_cases.sort(key=lambda x: x['plddt'], reverse=True)
for i, case in enumerate(model_fail_cases[:]):
    print(f"\nFail Case {i+1}:")
    print(f"Raw ID: {case['raw_id']}")
    print(f"UniProt ID: {case['pid']}")
    print(f"pLDDT: {case['plddt']:.2f}")
    print(f"GT GO (Propagated): {len(case['gt_prop'])} terms. Sample: {case['gt_prop']}")
    print(f"Pred GO: {len(case['pred_prop'])} terms. Sample: {case['pred_prop']}")
