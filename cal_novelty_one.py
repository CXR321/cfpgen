import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio import Align

# ================= 配置区域 =================
TEST_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
GO_MAPPING_PATH = 'go_mapping.pkl'

# 目标信息
TARGET_UNIPROT_ID = 'Q4X1A4'
TARGET_SEQ = 'MKAVHFGAGNIGRGFIGKLLADNGIEVTFADVNQPVIDALNARHSYEVNVVGENAQTDVVKNVAGINSMQEPEKVVEAIATADLVTTAVGPNILPIIAPLIAKGIVRRHETNDRPLNIIACENMVRGTTQLKGAVFDHLPEEHKAWVEEHVGFVDSAVDRIVPPSASEDILAVTVETFSEWIVDKTQFKGTLPNIPGMELTDNLMAFVERKLFTLNTGHAITAYLGQLAGHKTIRDAILDPQIRATVKGAMEESGSVLIKRYGFDREKHAAYIEKIIARFENPYLSDEVERVAGETIRKLGPNERLTKPLAGILEYDLPHDKLVEAYNSL'

# ================= 函数定义 =================

def calculate_sequence_identity(seq1, seq2):
    """
    使用 Biopython PairwiseAligner 计算全局序列一致性。
    Identity = Matches / Alignment Length
    """
    if not seq1 or not seq2: return 0.0
    if seq1 == seq2: return 1.0

    # 创建全局比对器
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'

    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = 0.0
    aligner.extend_gap_score = 0.0    

    # 获取最佳比对
    try:
        alignment = aligner.align(seq1, seq2)[0]
    except Exception:
        return 0.0
    


    matches = alignment.score
    total_len = alignment.shape[1]
    
    return matches / total_len if total_len > 0 else 0.0

def main():
    # 1. 加载数据
    print(f"Loading Test Data from {TEST_PATH}...")
    with open(TEST_PATH, 'rb') as f:
        test_data = pickle.load(f)

    # 加载映射 (为了打印好看的GO ID，虽然比对只需要Index)
    with open(GO_MAPPING_PATH, 'rb') as f:
        index_to_go = {v: k for k, v in pickle.load(f).items()}

    # 2. 在测试集中寻找目标蛋白的 GO 条件
    target_entry = None
    for entry in test_data:
        if entry['uniprot_id'] == TARGET_UNIPROT_ID:
            target_entry = entry
            break
    
    if target_entry is None:
        print(f"Error: Target {TARGET_UNIPROT_ID} not found in test dataset.")
        return

    # 获取目标的 GO Index 集合 (使用 set 以忽略顺序)
    target_go_indices = set(target_entry['go_f_mapped'])
    target_go_ids = [index_to_go[i] for i in target_go_indices]
    
    print("\n" + "="*60)
    print(f"TARGET: {TARGET_UNIPROT_ID}")
    print(f"GO Condition: {target_go_ids}")
    print("="*60)

    # 3. 遍历测试集，筛选同类并计算一致性
    max_identity = 0.0
    best_match_id = None
    best_match_seq = None
    
    same_condition_count = 0
    
    print("Scanning test set for proteins with the SAME GO condition...")
    
    for entry in tqdm(test_data):
        pid = entry['uniprot_id']
        
        # 跳过自己
        # if pid == TARGET_UNIPROT_ID:
        #     continue

        # 获取当前蛋白的 GO
        current_go_indices = set(entry['go_f_mapped'])
        
        # [核心筛选]：只计算拥有 完全相同 GO 集合 的蛋白
        # 如果你想要宽林一点（比如包含关系），可以改用 issubset
        # if True | (current_go_indices == target_go_indices):
        if (current_go_indices == target_go_indices):
            same_condition_count += 1
            
            # 获取序列 (兼容 key 写法)
            seq = entry.get('sequence', entry.get('aa_seq', ''))
            
            # 计算一致性
            identity = calculate_sequence_identity(TARGET_SEQ, seq)
            print(f"seq: {pid} sim: {identity}")
            
            if identity > max_identity:
                max_identity = identity
                best_match_id = pid
                best_match_seq = seq
                
                # 如果发现 100% 一致，可以直接 break (除非你想找所有)
                # if identity >= 1.0: break

    # 4. 输出结果
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total proteins with EXACT same GO condition: {same_condition_count}")
    
    if best_match_id:
        print(f"Max Sequence Identity: {max_identity:.4%}")
        print(f"Best Match ID:         {best_match_id}")
        
        # 简单展示一下长度对比
        print(f"Target Len:            {len(TARGET_SEQ)}")
        print(f"Match Len:             {len(best_match_seq)}")
        
        if max_identity > 0.9:
            print("\n[Warning] High sequence identity detected! Potential near-duplicate.")
        elif max_identity < 0.3:
            print("\n[Info] Low sequence identity. This is a challenging generalization case (remote homology).")
    else:
        print("No other proteins found with this exact GO combination.")

if __name__ == '__main__':
    main()