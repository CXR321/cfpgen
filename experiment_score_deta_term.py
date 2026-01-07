import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
from src.byprot.utils.ontology import Ontology

# ================= Configuration =================
# Set ICML/Academic Plotting Style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'
# Consistent Palette: Baseline (Blue), Ours (Red)
COLORS = {"Baseline": "#4c72b0", "Ours": "#c44e52"}

TRAIN_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
TEST_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
GO_MAPPING_PATH = 'go_mapping.pkl'
GO_OBO_PATH = 'data/go.obo' 

# Models
MODELS = {
    'Ours': './generation-results-dplm2-goonly-alldata-dm-ca-me-scale-0.2-weight-headclloss-2.0_sn-pn-11wstep/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut_preds_mf.tsv',
    'Baseline': './generation-results-cfpgen_650m/cfpgen_650m_go_preds_mf.tsv', 
    'Reference': '/AIRvePFS/dair/chenxr-data/repo/cfpgen/data-bin/uniprotKB/cfpgen_general_dataset/test_preds_mf.tsv'
}

CONFIDENCE_THRESHOLD = 0.0
MIN_SUPPORT = 10 

# ================= Helper Functions =================
def load_predictions(tsv_path):
    print(f"Loading {tsv_path}...")
    try:
        df = pd.read_csv(tsv_path, sep='\t', header=None, names=['raw_id', 'go_id', 'score'])
    except:
        df = pd.read_csv(tsv_path, sep=r'\s+', header=None, names=['raw_id', 'go_id', 'score'])
    
    def get_clean_id(raw):
        if 'SEQUENCE_ID=' in raw:
            return raw.replace('SEQUENCE_ID=', '').split('_L=')[0]
        return raw 
    df['clean_id'] = df['raw_id'].apply(get_clean_id)
    df = df[df['score'] >= CONFIDENCE_THRESHOLD]
    
    # Aggregation: pid -> set of GOs
    pid_to_preds = defaultdict(set)
    for raw_id, group in df.groupby('raw_id'):
        pid = group['clean_id'].iloc[0]
        pid_to_preds[pid].update(group['go_id'])
    return pid_to_preds

def propagate_and_index(pid_to_gos, ontology):
    """ Propagate ancestors and build inverted index """
    go_to_pids = defaultdict(set)
    for pid, gos in tqdm(pid_to_gos.items(), desc="Propagating"):
        # Note: If ancestor propagation is needed, uncomment the lines below
        # propagated = set()
        # for go in gos:
        #     propagated.update(ontology.get_ancestors(go))
        # for go in propagated:
        #     go_to_pids[go].add(pid)
        for go in gos:
            go_to_pids[go].add(pid)
    return go_to_pids

# ================= Main Process =================
def main():
    ontology = Ontology(GO_OBO_PATH)
    
    # 1. Load Data
    print("Loading Data...")
    with open(GO_MAPPING_PATH, 'rb') as f:
        index_to_go = {v: k for k, v in pickle.load(f).items()}
    
    with open(TEST_PATH, 'rb') as f:
        test_data = pickle.load(f)

    # 2. Build Ground Truth Index
    print("Building Ground Truth Index...")
    gt_map = {}
    for entry in test_data:
        pid = entry['uniprot_id']
        gos = set([index_to_go[i] for i in entry['go_f_mapped']])
        gt_map[pid] = gos
    
    term_gt_index = propagate_and_index(gt_map, ontology)

    # 3. Load Model Predictions
    model_term_indexes = {}
    for name, path in MODELS.items():
        print(f"Processing {name}...")
        preds_map = load_predictions(path)
        model_term_indexes[name] = propagate_and_index(preds_map, ontology)

    # 4. Calculate Term Metrics
    valid_terms = [t for t, pids in term_gt_index.items() if len(pids) >= MIN_SUPPORT]
    print(f"Analyzing {len(valid_terms)} GO terms (Support >= {MIN_SUPPORT})...")

    term_metrics = []
    for go in valid_terms:
        gt_pids = term_gt_index[go]
        n_gt = len(gt_pids)
        
        metrics_row = {'go_id': go, 'support': n_gt, 'name': go} # 'name' logic can be enhanced if ontology has names
        
        for model_name in MODELS.keys():
            pred_pids = model_term_indexes[model_name].get(go, set())
            tp_pids = gt_pids & pred_pids
            
            recall = len(tp_pids) / n_gt if n_gt > 0 else 0.0
            metrics_row[f'{model_name}_recall'] = recall
            
        term_metrics.append(metrics_row)

    df = pd.DataFrame(term_metrics)

    # 5. Calculate Delta Recall
    df['d_rec_ours'] = df['Ours_recall'] - df['Reference_recall']
    df['d_rec_base'] = df['Baseline_recall'] - df['Reference_recall']

    # ================= 6. Custom Filters for Top 5 Best & Worst =================
    print("\nApplying custom filters for Top/Bottom selection...")
    
    # --- Top 5 Best ---
    df_top_candidates = df[df['support'] >= 30].copy()
    top5 = df_top_candidates.sort_values(by=['d_rec_ours', 'support'], ascending=[False, False]).head(5)
    
    # --- Top 5 Worst ---
    df_bottom_candidates = df[df['support'] >= 35].copy()
    bottom5 = df_bottom_candidates.sort_values(by=['d_rec_ours', 'support'], ascending=[True, False]).head(5)
    
    # Concatenate: Top on top, Bottom reversed at bottom
    selected_df = pd.concat([top5, bottom5[::-1]])
    
    print("\nSelected GO Terms:")
    print(selected_df[['go_id', 'd_rec_ours', 'd_rec_base', 'support']])

    # ================= 7. Plotting (ICML Style) =================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    y_pos = np.arange(len(selected_df))
    height = 0.35
    
    d_ours = selected_df['d_rec_ours']
    d_base = selected_df['d_rec_base']
    
    # Plot Baseline (Blue)
    rects1 = ax.barh(y_pos - height/2, d_base, height, label='Baseline vs Ref', 
                     color=COLORS["Baseline"], alpha=0.9, edgecolor='black', linewidth=0.5)
    
    # Plot Ours (Red)
    rects2 = ax.barh(y_pos + height/2, d_ours, height, label='Ours vs Ref', 
                     color=COLORS["Ours"], alpha=0.9, edgecolor='black', linewidth=0.5)
    
    # Labels and Titles
    ax.set_xlabel('$\Delta$ Recall (Model - Reference)', fontsize=14, weight='bold')
    ax.set_title('Top 5 Best & Worst GO Terms by Recall Gain', fontsize=16, weight='bold', pad=20)
    ax.set_yticks(y_pos)
    
    # Format Y-axis labels
    def format_label(row):
        return f"{row['go_id']}"
    ax.set_yticklabels(selected_df.apply(format_label, axis=1), fontsize=11)
    
    # Vertical zero line
    ax.axvline(0, color='black', linewidth=1.2)
    
    # Horizontal separator line
    sep_idx = 4.5
    ax.axhline(sep_idx, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Section Annotations (Best vs Worst)
    # Get X-axis limits to position text nicely
    x_min, x_max = ax.get_xlim()
    text_x_pos = x_min + (x_max - x_min) * 0.02 # slight offset from left edge
    
    # "Best" Label
    ax.text(text_x_pos, sep_idx + 0.2, 'Top 5 WORST Gains', 
            fontsize=12, color='firebrick', fontweight='bold', va='bottom')
    
    # "Worst" Label
    ax.text(text_x_pos, sep_idx - 0.2, 'Top 5 BEST Drops', 
            fontsize=12, color='green', fontweight='bold', va='top')

    # Legend
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=12)
    
    plt.tight_layout()
    plt.savefig('analysis_top_bottom_go_terms_recall.png', dpi=300, bbox_inches='tight')
    print("Plot saved to analysis_top_bottom_go_terms_recall.png")
    plt.show()
    
    selected_df.to_csv('analysis_top_bottom_go_terms_recall.csv', index=False)

if __name__ == '__main__':
    main()