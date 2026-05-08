import os
import re
import pickle
import numpy as np
import pandas as pd
import click as ck
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from src.byprot.utils.ontology import Ontology

@ck.command()
@ck.option('--data-root', '-dr', default='./data-bin/uniprotKB/cfpgen_general_dataset/test.pkl', help='Data folder')
@ck.option('--test-predictions', '-tp', default='./generation-results-dplm2-goonly-alldata-dm-ca-me-scale-0.2-weight-headclloss-2.0_sn-pn-11wstep/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut_preds_mf.tsv', help='Test data set name (predictions TSV)')
@ck.option('--pdb-dir', '-pd', default='./generation-results-dplm2-goonly-alldata-dm-ca-me-scale-0.2-weight-headclloss-2.0_sn-pn-11wstep/esmfold_pdb', help='Directory containing esmfold pdb files')
@ck.option('--plddt-threshold', '-pt', default=70.0, help='Threshold for good plddt')
@ck.option('--f1-threshold', '-ft', default=0.5, help='Threshold for good F1 score')
def main(data_root, test_predictions, pdb_dir, plddt_threshold, f1_threshold):
    # 1. Load Ontology
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
    obo_path = os.path.join(base_dir, 'data', 'go.obo')
    ontology = Ontology(obo_path, with_rels=True)

    # 2. Load GT Data
    with open(data_root, 'rb') as f:
        test_data = pickle.load(f)

    # 3. Load Predictions
    predictions = {}
    use_name = False
    with open(test_predictions) as f:
        for line in f:
            it = line.strip().split('\t')
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
            else:
                prot_id = it[0]
            
            go_id = it[1]
            score = float(it[2])
            if prot_id not in predictions:
                predictions[prot_id] = {}
            predictions[prot_id][go_id] = score

    preds = {k: list(v.keys()) for k, v in predictions.items()}

    if use_name:
        gts = {ele['name']: set(ele['go_numbers']['F']) for ele in test_data}
    else:
        gts = {ele['uniprot_id']: set(ele['go_numbers']['F']) for ele in test_data}

    common_uids = [uid for uid in preds.keys() if uid in gts]
    gt_list = [gts[uid] for uid in common_uids]
    pred_list = [preds[uid] for uid in common_uids]

    # Propagate GO ancestors
    for i, this_gt_go in enumerate(gt_list):
        new_this_go = []
        for go in this_gt_go:
            new_this_go.extend(ontology.get_ancestors(go))
        gt_list[i] = set(new_this_go)

    unique_go_gt = set()
    for go_set in gt_list:
        unique_go_gt.update(go_set)

    unique_go_pred = set()
    for go_set in pred_list:
        unique_go_pred.update(go_set)

    unique_go = unique_go_gt & unique_go_pred

    for i, this_gt_go in enumerate(pred_list):
        pred_list[i] = set([ele for ele in this_gt_go if ele in unique_go])
    for i, this_gt_go in enumerate(gt_list):
        gt_list[i] = set([ele for ele in this_gt_go if ele in unique_go])

    # Calculate per-sample F1
    mlb = MultiLabelBinarizer()
    all_go_terms = gt_list + pred_list
    mlb.fit(all_go_terms)

    y_true_binary = mlb.transform(gt_list)
    y_pred_binary = mlb.transform(pred_list)

    f1_per_sample = []
    for i in range(len(y_true_binary)):
        if np.sum(y_true_binary[i]) == 0 and np.sum(y_pred_binary[i]) == 0:
            f1_per_sample.append(1.0)
        else:
            # print(y_true_binary[i])
            # print(y_pred_binary[i])
            # exit()
            f1 = f1_score(y_true_binary[i].reshape(1, -1), 
                          y_pred_binary[i].reshape(1, -1), 
                          average='micro', zero_division=0)
            f1_per_sample.append(f1)

    f1_dict = {uid: f1 for uid, f1 in zip(common_uids, f1_per_sample)}

    # 4. Parse PDB filenames for pLDDT
    plddt_dict = {}
    ptm_dict = {}
    
    pdb_files = os.listdir(pdb_dir)
    for fname in pdb_files:
        if not fname.endswith('.pdb'):
            continue
        # SEQUENCE_ID=A0A023W421_L=213_plddt_57.27533721923828_ptm_0.607.pdb
        match = re.search(r'SEQUENCE_ID=([\w\d]+)_.*plddt_([\d\.]+)_ptm_([\d\.]+)', fname)
        if match:
            uid = match.group(1)
            plddt = float(match.group(2))
            ptm = float(match.group(3)[:-1])
            plddt_dict[uid] = plddt
            ptm_dict[uid] = ptm

    # 5. Combine and Analyze
    analysis_data = []
    for uid in common_uids:
        if uid in plddt_dict:
            analysis_data.append({
                'uniprot_id': uid,
                'f1_score': f1_dict[uid],
                'plddt': plddt_dict[uid],
                'ptm': ptm_dict[uid]
            })

    df = pd.DataFrame(analysis_data)
    
    print(f"Total overlapping samples for analysis: {len(df)}")
    if len(df) == 0:
        print("No overlapping data found. Please check IDs and file paths.")
        return

    # Basic stats
    print("\n--- Basic Statistics ---")
    print(df[['f1_score', 'plddt', 'ptm']].describe())
    
    # Correlation
    print("\n--- Correlation Matrix ---")
    print(df[['f1_score', 'plddt', 'ptm']].corr())
    
    # Failure categorization
    df['good_plddt'] = df['plddt'] >= plddt_threshold
    df['good_f1'] = df['f1_score'] >= f1_threshold

    print(f"\n--- Failure Analysis (pLDDT >= {plddt_threshold}, F1 >= {f1_threshold}) ---")
    
    both_good = len(df[(df['good_plddt']) & (df['good_f1'])])
    plddt_bad_f1_good = len(df[(~df['good_plddt']) & (df['good_f1'])])
    plddt_good_f1_bad = len(df[(df['good_plddt']) & (~df['good_f1'])])
    both_bad = len(df[(~df['good_plddt']) & (~df['good_f1'])])
    
    print(f"Both Good (Success): {both_good} ({both_good/len(df)*100:.2f}%)")
    print(f"Only Structure Failed (Bad pLDDT, Good F1): {plddt_bad_f1_good} ({plddt_bad_f1_good/len(df)*100:.2f}%)")
    print(f"Only Function Failed (Good pLDDT, Bad F1): {plddt_good_f1_bad} ({plddt_good_f1_bad/len(df)*100:.2f}%)")
    print(f"Both Failed: {both_bad} ({both_bad/len(df)*100:.2f}%)")
    
    print("\nConclusion: Which is the more common failure mode?")
    if plddt_bad_f1_good > plddt_good_f1_bad:
        print("-> The model more easily generates proteins with POOR STRUCTURE (low pLDDT) but correct function.")
    elif plddt_good_f1_bad > plddt_bad_f1_good:
        print("-> The model more easily generates proteins with POOR FUNCTION (low F1) but good structure.")
    else:
        print("-> Both failure modes are equally likely.")

    # Save to CSV for further manual checking
    output_csv = "failure_analysis_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nDetailed results saved to {output_csv}")
    
    # Generate scatter plot
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x='plddt', y='f1_score', alpha=0.5)
        plt.axvline(x=plddt_threshold, color='r', linestyle='--', label=f'pLDDT threshold ({plddt_threshold})')
        plt.axhline(y=f1_threshold, color='g', linestyle='--', label=f'F1 threshold ({f1_threshold})')
        plt.title('pLDDT vs GO F1 Score')
        plt.xlabel('pLDDT')
        plt.ylabel('GO F1 Score')
        plt.legend()
        plt.savefig('plddt_vs_f1.png')
        print("Scatter plot saved to plddt_vs_f1.png")

        import matplotlib.pyplot as plt
        import seaborn as sns

        # 创建 JointGrid 密度图
        g = sns.jointplot(
            data=df, 
            x='plddt', 
            y='f1_score', 
            kind='kde',        # 核密度估计
            fill=True,         # 填充颜色
            cmap='Blues',      # 蓝色渐变，颜色越深密度越大
            thresh=0.01,       # 过滤掉极低密度的外围点
            height=8, 
            ratio=4
        )
        
        # 在主图上添加阈值虚线
        g.ax_joint.axvline(x=plddt_threshold, color='red', linestyle='--', linewidth=1.5, label=f'pLDDT={plddt_threshold}')
        g.ax_joint.axhline(y=f1_threshold, color='green', linestyle='--', linewidth=1.5, label=f'F1={f1_threshold}')
        
        # 设置标签和标题 (JointGrid 的标题需要稍微调整位置)
        g.set_axis_labels('pLDDT', 'GO F1 Score')
        g.ax_joint.legend(loc='lower left')
        g.fig.suptitle('Density Distribution: pLDDT vs GO F1 Score', y=1.02, fontsize=14)
        
        plt.savefig('plddt_vs_f1_density.png', dpi=300, bbox_inches='tight')
        print("Density plot saved to plddt_vs_f1_density.png")
    except Exception as e:
        print(f"Could not generate plot: {e}")

if __name__ == '__main__':
    main()
