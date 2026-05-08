import pandas as pd
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from src.byprot.utils.ontology import Ontology

# ================= Configuration =================
# 1. Paths
# Ground Truth Pickle (New Test Set)
TEST_GT_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/generated_candidates_motif_emb.pkl'
GO_MAPPING_PATH = 'go_mapping.pkl'
GO_OBO_PATH = 'data/go.obo'

# Predictions TSV
PATH_OURS = 'generation-results-dplm2-goonly-new-unseen/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut_preds_mf.tsv'
# PATH_OURS = 'generation-results-dplm2-goonly-new-unseen-filter/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut_preds_mf.tsv'
PATH_BASELINE = 'generation-results-cfpgen_650m-new_unseen/cfpgen_650m_go_preds_mf.tsv'

# 2. Threshold
CONFIDENCE_THRESHOLD = 0.0

# ================= Helper Functions =================

def load_ontology(obo_path):
    print(f"Loading Ontology from {obo_path}...")
    return Ontology(obo_path)

def load_grouped_ground_truth(pkl_path, mapping_path):
    print(f"Loading Ground Truth from {pkl_path}...")
    
    with open(mapping_path, 'rb') as f:
        go_mapping = pickle.load(f)
    index_to_go = {v: k for k, v in go_mapping.items()}

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Dictionary: frozenset(GO_ids) -> list of uniprot_ids
    grouped_data = defaultdict(list)
    pid_to_gt = {}

    for entry in data:
        pid = entry['uniprot_id']
        # Convert indices to GO strings
        go_ids = frozenset([index_to_go[i] for i in entry['go_f_mapped']])
        
        grouped_data[go_ids].append(pid)
        pid_to_gt[pid] = go_ids
    
    print(f"Loaded {len(pid_to_gt)} proteins.")
    print(f"Found {len(grouped_data)} unique GO label combinations.")
    return grouped_data, pid_to_gt

def load_predictions(tsv_path):
    print(f"Loading Predictions from {tsv_path}...")
    try:
        # Try standard loading
        df = pd.read_csv(tsv_path, sep='\t', header=None, names=['raw_id', 'go_id', 'score'])
    except:
        # Fallback for variable whitespace
        df = pd.read_csv(tsv_path, sep=r'\s+', header=None, names=['raw_id', 'go_id', 'score'])
    
    df = df[df['score'] >= CONFIDENCE_THRESHOLD]

    def extract_id(raw):
        if 'SEQUENCE_ID=' in raw:
            return raw.replace('SEQUENCE_ID=', '').split('_L=')[0]
        return raw

    df['uniprot_id'] = df['raw_id'].apply(extract_id)
    
    # Dict: uniprot_id -> set of GO IDs
    pred_dict = df.groupby('uniprot_id')['go_id'].apply(set).to_dict()
    return pred_dict

def evaluate_subset(pids, gt_combo_set, pred_dict, ontology):
    """
    Computes metrics for a specific group of proteins (that share the same GT).
    """
    all_gt_prop = []
    all_pred_prop = []
    
    # 1. Propagation
    # Since all PIDs in this group have the SAME raw GT, we propagate it once to save time
    gt_prop_base = set()
    # for go in gt_combo_set:
    #     gt_prop_base.update(ontology.get_ancestors(go))
    gt_prop_base = gt_combo_set
    
    # Collect data for all PIDs in this group
    for pid in pids:
        # GT is constant for this group
        all_gt_prop.append(gt_prop_base)
        
        # Pred varies per PID
        pred_raw = pred_dict.get(pid, set())
        pred_prop = set()
        # for go in pred_raw:
        #     pred_prop.update(ontology.get_ancestors(go))
        pred_prop = pred_raw
        all_pred_prop.append(pred_prop)

    # 2. Intersection (Define valid evaluation space)
    unique_gt = set().union(*all_gt_prop)
    unique_pred = set().union(*all_pred_prop)
    unique_space = unique_gt & unique_pred
    
    final_gt = []
    final_pred = []
    subset_matches = 0
    
    for g, p in zip(all_gt_prop, all_pred_prop):
        # Filter by intersection space
        g_filt = {go for go in g}
        p_filt = {go for go in p}

        g_expand = set()
        for go in g:
            g_expand.update(ontology.get_ancestors(go))
        
        # final_gt.append(list(g_filt))
        final_gt.append(list(g_expand))
        final_pred.append(list(p_filt))

        # print(f"true: {g_filt}")
        # print(f"pred: {p_filt}")
        
        # --- Exact Match Logic (Subset: GT <= Pred) ---
        # Note: We check if the FILTERED GT is a subset of FILTERED Pred.
        # This ensures we don't penalize for missing GO terms that are not in the intersection space 
        # (though usually intersection includes all GTs if ontology is consistent).

        if len(g_filt.intersection(p_filt)) > 0:
            subset_matches += len(g_filt.intersection(p_filt))

        # if len(g_filt) > 0 and g_filt.issubset(p_filt):
        #     subset_matches += 1
        elif len(g_filt) == 0:
            # If GT is empty (rare), and Pred is empty, it's a match? 
            # Or if GT is empty, any Pred is a superset.
            # subset_matches += 1
            exit()

    # 3. Compute Metrics
    mlb = MultiLabelBinarizer()
    mlb.fit(final_gt + final_pred)
    y_true = mlb.transform(final_gt)
    y_pred = mlb.transform(final_pred)
    
    f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    subset_acc = subset_matches / (len(pids) * 2) if len(pids) > 0 else 0.0
    
    return f1, subset_acc

# ================= Main Execution =================

def main():
    # 1. Load Resources
    ontology = load_ontology(GO_OBO_PATH)
    grouped_data, _ = load_grouped_ground_truth(TEST_GT_PATH, GO_MAPPING_PATH)
    
    # 2. Load Predictions
    preds_ours = load_predictions(PATH_OURS)
    preds_base = load_predictions(PATH_BASELINE)
    
    # 3. Iterate over each GO Combination Group
    rows = []
    
    print("\nStarting evaluation per GO combination...")
    for gt_combo, pids in tqdm(grouped_data.items(), desc="Evaluating Groups"):
        
        # Calculate Ours
        f1_ours, em_ours = evaluate_subset(pids, gt_combo, preds_ours, ontology)
        
        # Calculate Baseline
        f1_base, em_base = evaluate_subset(pids, gt_combo, preds_base, ontology)
        
        # Format GO Combo for display (comma separated)
        combo_str = ",".join(sorted(list(gt_combo)))
        
        rows.append({
            'GO_Combo': combo_str,
            'Count': len(pids),
            'Ours_F1': f1_ours,
            'Base_F1': f1_base,
            'Ours_SubsetMatch': em_ours,
            'Base_SubsetMatch': em_base
        })
    
    # 4. Create DataFrame and Sort
    df = pd.DataFrame(rows)
    # Sort by Count (descending) so most frequent combos appear first
    # df = df.sort_values(by='Count', ascending=False)
    
    # 5. Output
    print("\n" + "="*80)
    print("COMPARISON RESULTS (Top 10 groups by size)")
    print("Exact Match Definition: GT is subset of Pred")
    print("="*80)
    print(df.head(10).to_string(index=False))
    
    # # Calculate weighted averages for a summary line
    total_samples = df['Count'].sum()
    avg_f1_ours = (df['Ours_F1'] * df['Count']).sum() / total_samples
    avg_f1_base = (df['Base_F1'] * df['Count']).sum() / total_samples
    avg_em_ours = (df['Ours_SubsetMatch'] * df['Count']).sum() / total_samples
    avg_em_base = (df['Base_SubsetMatch'] * df['Count']).sum() / total_samples

    print("-" * 80)
    print(f"WEIGHTED AVG | Count: {total_samples} | Ours F1: {avg_f1_ours:.4f} | Base F1: {avg_f1_base:.4f} | Ours Match: {avg_em_ours:.4f} | Base Match: {avg_em_base:.4f}")
    print("="*80)

    # Save full table
    out_file = "grouped_comparison_results.csv"
    df.to_csv(out_file, index=False)
    print(f"\nFull breakdown saved to {out_file}")

# # ================= FILTERING LOGIC =================
#     # 只保留 Ours_SubsetMatch > 0 的行
#     # Ours_SubsetMatch 是正确率 (matches / count)，大于 0 意味着至少对了 1 个
#     print(f"\nOriginal unique GO combos: {len(df)}")
    
#     df_filtered = df[df['Ours_SubsetMatch'] >= 0].copy()
    
#     # 按 Count 降序排列 (可选)
#     # df_filtered = df_filtered.sort_values(by='Count', ascending=False)
    
#     print(f"Filtered GO combos (at least one match in Ours): {len(df_filtered)}")
#     print("="*80)
#     print(df_filtered.head(10).to_string(index=False))
    
#     # 5. Save Results
#     out_file = "successful_go_combos.csv"
#     df_filtered.to_csv(out_file, index=False)
#     print(f"\nSuccessfully saved {len(df_filtered)} combos to {out_file}")

if __name__ == "__main__":
    main()