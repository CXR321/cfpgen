import sys, os, time
import random
import tempfile
import shutil
import uuid
from tqdm import tqdm
import pandas as pd
import datetime
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import subprocess
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import re


def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def parse_fasta_function_labels(fasta_file):
    func_labels_list = []

    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()

            # 检查是否是FASTA文件的描述行
            if line.startswith(">"):
                # 移除 '>' 并将标签按 '|' 分割
                func_labels = line[1:].split('|')
                # 将分割后的标签添加到列表
                func_labels_list.append(func_labels)

    return func_labels_list

def run_interproscan(fasta_file, output_file):
    """
    运行 InterProScan 分析FASTA文件，并将结果保存到指定的输出文件中。
    返回 IPR domains 和 GO numbers 两个列表。
    """
    # 指定 InterProScan 的工作目录
    interproscan_dir = "/home/yinj0b/repository/my_interproscan/interproscan-5.69-101.0"

    # 设置 Java 环境变量
    os.environ["JAVA_HOME"] = "/home/yinj0b/thirdparty/java-11/jdk-11.0.24+8"
    os.environ["JRE_HOME"] = os.environ["JAVA_HOME"] + "/jre"
    os.environ["CLASSPATH"] = ".:" + os.environ["JAVA_HOME"] + "/lib:" + os.environ["JRE_HOME"] + "/lib"
    os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]

    # 构造 InterProScan 命令
    command = [
        "./interproscan.sh", "-i", fasta_file, "-dp", "-f", "tsv", "-goterms",
        "-o", output_file, "-cpu", '32'
    ]

    # 进入指定的工作目录并运行 InterProScan
    current_dir = os.getcwd()

    try:
        # 切换到 InterProScan 的工作目录
        os.chdir(interproscan_dir)
        # 运行 InterProScan
        with open(os.devnull, 'w') as devnull:
            subprocess.run(command, stdout=devnull, stderr=devnull, check=True)
    finally:
        # 切换回原来的工作目录
        os.chdir(current_dir)

    ipr_domains = set()
    go_numbers = set()

    # 解析 InterProScan 的输出文件
    with open(output_file, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) >= 14:
                ipr = fields[11]  # IPR number
                go_terms = fields[13]  # GO numbers
                if ipr:
                    ipr_domains.add(ipr)
                if go_terms:
                    go_numbers.update(go_terms.split('|'))

    return list(ipr_domains), list(go_numbers)

def save_single_fasta(entry, output_path):
    """
    将单个FASTA条目保存为独立的FASTA文件。
    """
    with open(output_path, 'w') as f:
        f.write(f"{entry['header']}\n{entry['sequence']}\n")

def process_fasta_part(part_entries, part_index, num_parts, fasta_file):
    """
    处理每个部分的FASTA条目并保存预测结果。
    """
    unique_id = uuid.uuid4().hex  # 生成唯一标识符
    temp_dir = os.path.join(os.path.dirname(fasta_file), f"{unique_id}_part{part_index}")
    os.makedirs(temp_dir, exist_ok=True)  # 创建临时目录

    ipr_domains_list = []
    go_numbers_list = []

    for i, entry in tqdm(enumerate(part_entries), desc=f"Processing part {part_index}/{num_parts}"):
        # 为每个条目创建临时FASTA文件
        temp_fasta_file = os.path.join(temp_dir, f"temp_fasta_{i}.fasta")
        save_single_fasta(entry, temp_fasta_file)

        # 为 InterProScan 生成的输出文件名
        interpro_output_file = os.path.join(temp_dir, f"interpro_output_{i}.tsv")

        # 运行 InterProScan，并获取 IPR 和 GO numbers
        ipr_domains, go_numbers = run_interproscan(temp_fasta_file, interpro_output_file)

        # 将结果添加到列表
        ipr_domains_list.append(ipr_domains)
        go_numbers_list.append(go_numbers)

        # 删除临时文件
        os.remove(temp_fasta_file)
        os.remove(interpro_output_file)

    # 保存每个部分的预测结果
    part_pkl_path = os.path.splitext(fasta_file)[0] + f'_part{part_index}.pkl'
    with open(part_pkl_path, 'wb') as pkl_file:
        pickle.dump((ipr_domains_list, go_numbers_list), pkl_file)

    shutil.rmtree(temp_dir)

    return part_pkl_path

def split_and_process_fasta(fasta_file, num_parts, ds=10, sub_set=None):
    """
    将FASTA文件分成指定数量的份并保存每份的预测结果。
    """
    with open(fasta_file, 'r') as f:
        fasta_entries = []
        entry = {'header': '', 'sequence': ''}

        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if entry['header'] and entry['sequence']:
                    fasta_entries.append(entry)
                    entry = {'header': '', 'sequence': ''}  # 重置 entry

                entry['header'] = line  # 保存新的 header 行
            else:
                entry['sequence'] += line  # 拼接序列部分

        # 添加最后一个 entry
        if entry['header'] and entry['sequence']:
            fasta_entries.append(entry)

    fasta_entries = fasta_entries[::ds]
    if sub_set is not None:
        fasta_entries = [
            ele for ele in fasta_entries
            if (match := re.search(r'SEQUENCE_ID=([^\s_]+)', ele['header'])) and match.group(1) in sub_set
        ]  # cfpgen
    total_entries = len(fasta_entries)
    entries_per_part = total_entries // num_parts
    part_pkl_files = []

    print(f"Processing {total_entries} sequences.")

    for part_index in range(num_parts):
        part_pkl_path = os.path.splitext(fasta_file)[0] + f'_part{part_index}.pkl'
        if os.path.exists(part_pkl_path):
            print(f"Part {part_index} already processed, skipping...")
            part_pkl_files.append(part_pkl_path)
            continue

        start_index = part_index * entries_per_part
        end_index = (start_index + entries_per_part) if part_index < (num_parts - 1) else total_entries
        part_entries = fasta_entries[start_index:end_index]
        part_pkl_path = process_fasta_part(part_entries, part_index, num_parts, fasta_file)
        part_pkl_files.append(part_pkl_path)

    return part_pkl_files

def process_fasta(fasta_file, num_parts=1, sub_set=None):
    """
    逐条提取FASTA文件中的每个条目，将其分成多份并进行处理。
    支持断点续传功能。
    """
    part_pkl_files = split_and_process_fasta(fasta_file, num_parts, ds=1, sub_set=sub_set)

    # 合并所有的.pkl文件
    combined_predictions = []
    for part_pkl in part_pkl_files:
        with open(part_pkl, 'rb') as pkl_file:
            ipr_domains_list, _ = pickle.load(pkl_file)
            combined_predictions.extend(ipr_domains_list)
        # os.remove(part_pkl)

    return combined_predictions

def get_esm3_seq_names(fasta_file, sub_set):
    with open(fasta_file, 'r') as f:
        fasta_entries = []
        entry = {'header': '', 'sequence': ''}

        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if entry['header'] and entry['sequence']:
                    fasta_entries.append(entry)
                    entry = {'header': '', 'sequence': ''}  # 重置 entry

                entry['header'] = line  # 保存新的 header 行
            else:
                entry['sequence'] += line  # 拼接序列部分

        # 添加最后一个 entry
        if entry['header'] and entry['sequence']:
            fasta_entries.append(entry)

    if sub_set is not None:
        fasta_entries = [ele for ele in fasta_entries if ele['header'].split('>')[1] in sub_set]
    return [ele['header'].split('>')[1] for ele in fasta_entries]

def extract_seq_id(header):
    match = re.search(r'SEQUENCE_ID=([^\s_]+)', header)
    return match.group(1) if match else None

def get_seq_names(fasta_file, sub_set):
    """
    从 FASTA 文件中提取 SEQUENCE_ID。

    参数：
        fasta_file (str): FASTA 文件路径。
        sub_set (set): 包含目标 SEQUENCE_ID 的集合。

    返回：
        list: 匹配 sub_set 中 ID 的 FASTA header 列表。
    """
    with open(fasta_file, 'r') as f:
        fasta_entries = []
        entry = {'header': '', 'sequence': ''}

        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if entry['header'] and entry['sequence']:
                    fasta_entries.append(entry)
                    entry = {'header': '', 'sequence': ''}  # 重置 entry

                entry['header'] = line  # 保存新的 header 行
            else:
                entry['sequence'] += line  # 拼接序列部分

        # 添加最后一个 entry
        if entry['header'] and entry['sequence']:
            fasta_entries.append(entry)

    if sub_set is not None:
        fasta_entries = [
            ele for ele in fasta_entries
            if (seq_id := extract_seq_id(ele['header'])) and seq_id in sub_set
        ]
    return [extract_seq_id(ele['header']) for ele in fasta_entries]

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python eval_ipr.py <fasta_file> <gt_pkl>")
        sys.exit(1)

    fasta_file = sys.argv[1]
    gt_pkl_file = sys.argv[2]

    with open(gt_pkl_file, 'rb') as f:
        gt_data = pickle.load(f)[::10]   # to save computational cost
    select_samples = [ele['uniprot_id'] for ele in gt_data]

    '''
    step 1: generate ipscan predictions
    '''
    output_log = fasta_file.replace('.fasta', '_interproscan_eval.log')
    predictions = process_fasta(fasta_file, sub_set=select_samples)

    # save predictions
    predictions_pkl = os.path.splitext(output_log)[0] + '.pkl'
    with open(predictions_pkl, 'wb') as pkl_file:
        pickle.dump(predictions, pkl_file)

    '''
    step 2: eval ipr score
    '''
    predictions_pkl = os.path.splitext(output_log)[0] + '.pkl'
    predictions = load_pkl_file(predictions_pkl)
    gts = [ele['ipr_numbers'] for ele in gt_data]

    y_true = [[ele.strip("'").strip('"') for ele in gt] for gt in gts]
    y_pred = [[ele.strip("'").strip('"') for ele in pred if ele != '-'] for pred in predictions]

    mlb = MultiLabelBinarizer()
    y_true_cleaned = [set(s.strip("'").strip('"') for s in labels) for labels in y_true]
    y_pred_cleaned = [set(s.strip("'").strip('"') for s in labels) for labels in y_pred]

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

    out_file = predictions_pkl.replace('.pkl', '_ipr-eval.txt')

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

    with open(out_file, 'w') as log_file:

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

    print(f"Evaluation results saved to {out_file}")


