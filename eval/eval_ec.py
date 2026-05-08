import os, sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer
import ast
from collections import Counter
import random
import pickle
import re


def save_pkl_file(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Updated data saved to {file_path}")


def load_pkl_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def safe_split_and_check(x, threshold=0.01):
    if pd.isna(x) or '/' not in x:
        return None
    try:
        parts = x.split('/')
        if len(parts) != 2:
            return None
        score = float(parts[1])
        if score > threshold:
            return parts[0]
    except (ValueError, IndexError):
        return None
    return None


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python <csv_file_name> <gt_pkl>")
        sys.exit(1)

    csv_file_name = sys.argv[1]
    gt_file = sys.argv[2]

    gts = load_pkl_file(gt_file)

    df = pd.read_fwf(csv_file_name)
    df = df[df.columns[-1]].str.split(",", expand=True)

    num_cols = df.shape[1]
    columns = ['Sequence'] + [f'Prediction{i+1}' for i in range(num_cols - 1)]
    df.columns = columns

    if 'L=' in df['Sequence'][0]:
        if "_ID=" in df['Sequence'][0]:
            df['ID'] = df['Sequence'].apply(lambda x: x.split("_ID=")[1].split("_L=")[0])
        else:
            df['ID'] = df['Sequence'].apply(lambda x: x.split("SEQUENCE_")[1].split("_L=")[0])
    elif 'prompt' in df['Sequence'][0]:
        df['ID'] = df['Sequence'].apply(lambda x: x.split(" ")[0].split('_')[-1])
    else:
        df['ID'] = df['Sequence'].apply(lambda x: x.split(" ")[0])

    predictions = [col for col in df.columns if col.startswith('Prediction')]
    for k, col in enumerate(predictions):
        df[f'{col}_EC'] = df[col].apply(
            lambda x: safe_split_and_check(x))

    df['All_EC_Predictions'] = df[[f'{col}_EC' for col in predictions]].apply(
        lambda row: ','.join(
            [ec.split('EC:')[1] for ec in row if pd.notna(ec) and 'EC:' in ec]
        ),
        axis=1
    )

    gt_dict = {ele['uniprot_id']: str(ele['EC_number']) for ele in gts}
    df['GT'] = df['ID'].map(gt_dict)
    df['GT'] = df['GT'].apply(lambda x: x[1:-1])

    df['GT_Set'] = df['GT'].apply(lambda x: set([ele[1:-1] for ele in x.split(',')]) if pd.notna(x) else set())
    df['Pred_Set'] = df['All_EC_Predictions'].apply(lambda x: set(x.split(',')) if pd.notna(x) else set())

    y_true = df['GT_Set'].tolist()
    y_pred = df['Pred_Set'].tolist()

    mlb = MultiLabelBinarizer()
    y_true_cleaned = [set(s.strip("'").strip('"') for s in labels) for labels in y_true]
    y_pred_cleaned = [set(s.strip("'").strip('"') for s in labels) for labels in y_pred]

    unique_ec_gt = set()
    for go_set in y_true_cleaned:
        unique_ec_gt.update(go_set)

    unique_ec_pred = set()
    for go_set in y_pred_cleaned:
        unique_ec_pred.update(go_set)

    y_true_binary = mlb.fit_transform(y_true_cleaned)
    y_pred_binary = mlb.transform(y_pred_cleaned)

    precision_mac = precision_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
    recall_mac = recall_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
    f1_mac = f1_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)

    precision_mic = precision_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
    recall_mic = recall_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
    f1_mic = f1_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)

    # AUC-ROC
    auc_roc_macro = roc_auc_score(y_true_binary, y_pred_binary, average='macro')
    auc_roc_micro = roc_auc_score(y_true_binary, y_pred_binary, average='micro')

    # AUC-PR (AUPR)
    aupr_macro = average_precision_score(y_true_binary, y_pred_binary, average='macro')
    aupr_micro = average_precision_score(y_true_binary, y_pred_binary, average='micro')

    output_log = csv_file_name.replace('_maxsep.csv', '_eval.log')

    print(f'F1 Score (Micro): {f1_mic:.3f}')
    print(f'F1 Score (Macro): {f1_mac:.3f}')
    print(f'AUPR (Macro): {aupr_macro:.3f}')
    print(f'AUC-ROC (Macro): {auc_roc_macro:.3f}\n')

    print(f'AUPR (Micro): {aupr_micro:.3f}')
    print(f'AUC-ROC (Micro): {auc_roc_micro:.3f}\n')

    print(f'Precision (Macro): {precision_mac:.3f}')
    print(f'Recall (Macro): {recall_mac:.3f}')
    print(f'Precision (Micro): {precision_mic:.3f}')
    print(f'Recall (Micro): {recall_mic:.3f}')

    with open(output_log, 'w') as log_file:
        log_file.write(f'Precision (Macro): {precision_mac:.4f}\n')
        log_file.write(f'Recall (Macro): {recall_mac:.4f}\n')
        log_file.write(f'F1 Score (Macro): {f1_mac:.4f}\n')
        log_file.write(f'AUC-ROC (Macro): {auc_roc_macro:.4f}\n')
        log_file.write(f'AUPR (Macro): {aupr_macro:.4f}\n\n')

        log_file.write(f'Precision (Micro): {precision_mic:.4f}\n')
        log_file.write(f'Recall (Micro): {recall_mic:.4f}\n')
        log_file.write(f'F1 Score (Micro): {f1_mic:.4f}\n')
        log_file.write(f'AUC-ROC (Micro): {auc_roc_micro:.4f}\n')
        log_file.write(f'AUPR (Micro): {aupr_micro:.4f}\n\n')


