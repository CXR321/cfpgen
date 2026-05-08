
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
from sklearn.metrics import f1_score
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
    use_name = False
    with open(test_predictions) as f:
        for line in f:
            it = line.strip().split('\t')
            if len(it) < 3: continue
            if 'prompt_first_seq30' in it[0]:
                prot_id = re.match(r'prompt_first_seq30_([\w\d]+)', it[0]).groups()[0]
            elif 'name=' in it[0]:
                prot_id = re.match(r'name=([\w\d\.]+)', it[0]).groups()[0]
                use_name = True
            elif 'recovery' in it[0]:
                prot_id = re.match(r'([\w\d]+)', it[0]).groups()[0]
            elif 'SEQUENCE_ID=' in it[0]:
                prot_id = re.match(r'SEQUENCE_ID=([\w\d\.]+)_L', it[0]).groups()[0]
            elif 'SEQUENCE_' in it[0]:
                prot_id = re.match(r'SEQUENCE_([\w\d\.]+)_L=', it[0]).groups()[0]
            elif '_seq30_' in it[0]:
                prot_id = re.match(r'go_prompt_longest_motif_seq30_([\w\d\.]+)', it[0]).groups()[0]
            else :
                prot_id = it[0]
            go_id = it[1]
            score = float(it[2])
            if prot_id not in predictions:
                predictions[prot_id] = {}
            predictions[prot_id][go_id] = score

    preds = {k:list(v.keys()) for k,v in predictions.items()}

    if use_name:
        gts = {ele['name']:set(ele['go_numbers']['F']) for ele in test_data}
        id_to_seq = {ele['name']:ele['sequence'] for ele in test_data}
        id_to_gos = {ele['name']:ele['go_numbers']['F'] for ele in test_data}
    else:
        gts = {ele['uniprot_id']:set(ele['go_numbers']['F']) for ele in test_data}
        id_to_seq = {ele['uniprot_id']:ele['sequence'] for ele in test_data}
        id_to_gos = {ele['uniprot_id']:ele['go_numbers']['F'] for ele in test_data}

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

    # Load generated sequences
    gen_fasta = 'generation-results-dplm2-goonly-alldata-dm-ca-me-scale-0.2-weight-headclloss-2.0_sn-pn-11wstep/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut.fasta'
    gen_ids, gen_seqs = [], []
    if os.path.exists(gen_fasta):
        with open(gen_fasta, 'r') as f:
            curr_id = ''
            curr_seq = []
            for line in f:
                if line.startswith('>'):
                    if curr_id:
                        gen_ids.append(curr_id)
                        gen_seqs.append(''.join(curr_seq))
                    curr_id = line[1:].strip().split()[0]
                    # Extract ID from something like SEQUENCE_ID=P11943_L=200
                    match = re.search(r'SEQUENCE_ID=([\w\d\.]+)', curr_id)
                    if match:
                        curr_id = match.group(1).split('_')[0]
                    curr_seq = []
                else:
                    curr_seq.append(line.strip())
            if curr_id:
                gen_ids.append(curr_id)
                gen_seqs.append(''.join(curr_seq))
    
    id_to_gen_seq = dict(zip(gen_ids, gen_seqs))

    # DEBUG
    print(f"Sample gen_ids: {gen_ids[:5]}")
    print(f"Sample common_uids: {common_uids[:5]}")

    perfect_matches = []
    for i in range(len(common_uids)):
        if np.array_equal(y_true_binary[i], y_pred_binary[i]):
            if np.sum(y_true_binary[i]) > 0:
                uid = common_uids[i]
                perfect_matches.append({
                    'id': uid,
                    'gt_gos': id_to_gos[uid],
                    'pred_gos': pred_list[i],
                    'gen_seq': id_to_gen_seq.get(uid, "NOT FOUND")
                })

    print(f"Found {len(perfect_matches)} perfect matches out of {len(common_uids)} common samples.")
    
    # Print top 5
    for i, match in enumerate(perfect_matches[:10]):
        print(f"\nMatch {i+1}:")
        print(f"ID: {match['id']}")
        print(f"GO Label Combination (Target): {match['gt_gos']}")
        print(f"Generated Protein Sequence: {match['gen_seq']}")

if __name__ == '__main__':
    main()
