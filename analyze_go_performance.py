
import sys
import os
import importlib.util

# Import Ontology directly from the file to avoid triggering deepspeed via byprot package
ontology_path = os.path.join(os.path.dirname(__file__), "src/byprot/utils/ontology.py")
spec = importlib.util.spec_from_file_location("ontology", ontology_path)
ontology_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ontology_module)
Ontology = ontology_module.Ontology

import numpy as np
import pandas as pd
import pickle
import re
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def main():
    data_root = 'data-bin/uniprotKB/cfpgen_general_dataset/test.pkl'
    test_predictions = '../cfpgen/generation-results-dplm2-goonly-alldata-dm-ca-me-scale-0.2-weight-headclloss-2.0_sn-pn-11wstep/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut_preds_mf.tsv'
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
    obo_path = os.path.join(base_dir, 'data', 'go.obo')
    ontology = Ontology(obo_path, with_rels=True)
    
    test_data = load_pkl_file(data_root)

    predictions = {}
    with open(test_predictions) as f:
        for line in f:
            it = line.strip().split('\t')
            if len(it) < 3: continue
            # Simplified ID extraction (assuming IDs are like P12345)
            match = re.search(r'([\w\d]+)', it[0])
            if match:
                prot_id = match.group(1)
                # Filter specific cases based on eval_go logic
                if 'SEQUENCE_ID=' in it[0]:
                    prot_id = re.match(r'SEQUENCE_ID=([\w\d\.]+)_L', it[0]).groups()[0]
                elif 'SEQUENCE_' in it[0]:
                    prot_id = re.match(r'SEQUENCE_([\w\d\.]+)_L=', it[0]).groups()[0]
                elif '_seq30_' in it[0]:
                    prot_id = re.match(r'go_prompt_longest_motif_seq30_([\w\d\.]+)', it[0]).groups()[0]
                
                go_id = it[1]
                score = float(it[2])
                if prot_id not in predictions:
                    predictions[prot_id] = {}
                predictions[prot_id][go_id] = score

    preds = {k:list(v.keys()) for k,v in predictions.items()}
    gts = {ele['uniprot_id']:set(ele['go_numbers']['F']) for ele in test_data}

    common_uids = [uid for uid in preds.keys() if uid in gts]
    gt_list = [gts[uid] for uid in common_uids]
    pred_list = [preds[uid] for uid in common_uids]    

    # Expand ancestors
    for i, this_gt_go in enumerate(gt_list):
        new_this_go = []
        for go in this_gt_go:
            new_this_go.extend(ontology.get_ancestors(go))
        gt_list[i] = set(new_this_go)

    # Unique GO terms in BOTH
    unique_go_gt = set()
    for go_set in gt_list:
        unique_go_gt.update(go_set)
    unique_go_pred = set()
    for go_set in pred_list:
        unique_go_pred.update(go_set)
    unique_go = unique_go_gt & unique_go_pred

    # Filter by common set
    filtered_pred_list = []
    filtered_gt_list = []
    for i in range(len(common_uids)):
        filtered_pred_list.append(set([ele for ele in pred_list[i] if ele in unique_go]))
        filtered_gt_list.append(set([ele for ele in gt_list[i] if ele in unique_go]))

    mlb = MultiLabelBinarizer()
    all_go_terms = filtered_gt_list + filtered_pred_list
    mlb.fit(all_go_terms)

    y_true_binary = mlb.transform(filtered_gt_list)
    y_pred_binary = mlb.transform(filtered_pred_list)

    # Calculate per-term stats
    term_stats = []
    for i, term in enumerate(mlb.classes_):
        y_true_term = y_true_binary[:, i]
        y_pred_term = y_pred_binary[:, i]
        
        freq = np.sum(y_true_term)
        if freq > 0:
            p = precision_score(y_true_term, y_pred_term, zero_division=0)
            r = recall_score(y_true_term, y_pred_term, zero_division=0)
            f1 = f1_score(y_true_term, y_pred_term, zero_division=0)
            
            term_name = ontology.get_term(term).get('name', 'Unknown')
            term_stats.append({
                'term': term,
                'name': term_name,
                'f1': f1,
                'precision': p,
                'recall': r,
                'frequency': freq
            })

    # Sort by F1 desc and frequency desc
    term_stats.sort(key=lambda x: (x['f1'], x['frequency']), reverse=True)

    print(f"{'GO ID':<15} {'F1':<8} {'Freq':<8} {'Name'}")
    print("-" * 60)
    for stat in term_stats:
        if stat['f1'] >= 0.8 and stat['frequency'] >= 10:
            print(f"{stat['term']:<15} {stat['f1']:<8.4f} {stat['frequency']:<8} {stat['name']}")

if __name__ == '__main__':
    main()
