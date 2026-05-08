import os
import re
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, DSSP

# ======================
# 配置
# ======================
PDB_DIR = "./pdbs"   # 存放 pdb 的文件夹
MODEL_ID = 0         # 默认用第一个 model

# DSSP 二级结构映射
SS_MAP = {
    "H": "H",  # alpha helix
    "G": "H",  # 3-10 helix
    "I": "H",  # pi helix
    "E": "E",  # beta strand
    "B": "E",  # beta bridge
}

def classify_ss(ss):
    if ss in SS_MAP:
        return SS_MAP[ss]
    else:
        return "C"  # coil / loop

def process_pdb(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[MODEL_ID]

    dssp = DSSP(model, pdb_path, dssp='/AIRvePFS/dair/chenxr-data/repo/dssp/mkdssp')

    counts = {"H": 0, "E": 0, "C": 0}
    total = 0

    for key in dssp.keys():
        ss = dssp[key][2]  # DSSP secondary structure code
        ss_class = classify_ss(ss)
        counts[ss_class] += 1
        total += 1

    return {
        "pdb": os.path.basename(pdb_path),
        "n_res": total,
        "H_frac": counts["H"] / total if total > 0 else 0.0,
        "E_frac": counts["E"] / total if total > 0 else 0.0,
        "C_frac": counts["C"] / total if total > 0 else 0.0,
        "H_cnt": counts["H"],
        "E_cnt": counts["E"],
        "C_cnt": counts["C"],
    }

def main_ss():
    records = []

    for fname in os.listdir(PDB_DIR):
        if fname.endswith(".pdb"):
            pdb_path = os.path.join(PDB_DIR, fname)
            try:
                rec = process_pdb(pdb_path)
                records.append(rec)
            except Exception as e:
                print(f"[WARN] Failed on {fname}: {e}")

    df = pd.DataFrame(records)
    df.to_csv("secondary_structure_stats.csv", index=False)

    print("Per-PDB statistics saved to secondary_structure_stats.csv\n")
    print("Folder-level mean statistics:")
    print(df[["H_frac", "E_frac", "C_frac"]].mean())

def extract_plddt_ptm(filename):
    """从文件名中提取pLDDT和pTM值"""
    # 匹配模式：plddt_XX.XXXXX_ptm_XX.XXX.pdb
    plddt_match = re.search(r'plddt_([\d.]+)_', filename)
    ptm_match = re.search(r'ptm_([\d.]+)\.pdb$', filename)
    
    plddt = None
    ptm = None
    
    if plddt_match:
        plddt = float(plddt_match.group(1))
    if ptm_match:
        ptm = float(ptm_match.group(1))
    
    return plddt, ptm

def calculate_stats(folder_path):
    plddt_values = []
    ptm_values = []
    paired_values = []
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdb') and 'ptm' in filename:
            plddt, ptm = extract_plddt_ptm(filename)
            if plddt is not None:
                plddt_values.append(plddt)
            if ptm is not None:
                ptm_values.append(ptm)
            if plddt is not None and ptm is not None:
                paired_values.append((plddt, ptm))
    
    if not plddt_values and not ptm_values:
        print("未找到包含pLDDT或pTM值的PDB文件")
        return None
    
    stats = {}
    
    if plddt_values:
        plddt_array = np.array(plddt_values)
        
        def calculate_plddt_percentage_above(threshold):
            return np.sum(plddt_array >= threshold) / len(plddt_array) * 100
        
        # pLDDT统计信息
        plddt_stats = {
            'count': len(plddt_values),
            'mean': np.mean(plddt_array),
            'median': np.median(plddt_array),
            'min': np.min(plddt_array),
            'max': np.max(plddt_array),
            'std': np.std(plddt_array),
            '25th_percentile': np.percentile(plddt_array, 25),
            '75th_percentile': np.percentile(plddt_array, 75),
            'plddt_70_above': calculate_plddt_percentage_above(70),
            'plddt_90_above': calculate_plddt_percentage_above(90),
            'plddt_80_above': calculate_plddt_percentage_above(80),
            'plddt_50_above': calculate_plddt_percentage_above(50),
        }
        stats['plddt'] = plddt_stats
    
    if ptm_values:
        ptm_array = np.array(ptm_values)
        
        def calculate_ptm_percentage_above(threshold):
            return np.sum(ptm_array >= threshold) / len(ptm_array) * 100
        
        # pTM统计信息
        ptm_stats = {
            'count': len(ptm_values),
            'mean': np.mean(ptm_array),
            'median': np.median(ptm_array),
            'min': np.min(ptm_array),
            'max': np.max(ptm_array),
            'std': np.std(ptm_array),
            '25th_percentile': np.percentile(ptm_array, 25),
            '75th_percentile': np.percentile(ptm_array, 75),
            'ptm_0.5_above': calculate_ptm_percentage_above(0.5),  # pTM > 0.5
            'ptm_0.7_above': calculate_ptm_percentage_above(0.7),  # pTM > 0.7
            'ptm_0.8_above': calculate_ptm_percentage_above(0.8),  # pTM > 0.8
            'ptm_0.9_above': calculate_ptm_percentage_above(0.9),  # pTM > 0.9
        }
        stats['ptm'] = ptm_stats

    if paired_values:
        paired_array = np.array(paired_values)
        # paired_array[:, 0] 是 plddt, paired_array[:, 1] 是 ptm
        
        # 统计 pLDDT > 70 且 pTM > 0.5
        # 注意：代码中使用了 > (大于)。如果想要包含边界值(>=)，请修改符号。
        mask_success = (paired_array[:, 0] > 70) & (paired_array[:, 1] > 0.5)
        success_count = np.sum(mask_success)
        success_rate = success_count / len(paired_array) * 100
        
        stats['combined'] = {
            'count': len(paired_values),
            'plddt_70_ptm_0.5_rate': success_rate
        }

    return stats

def print_stats(stats, folder_path):
    print("文件夹路径:", folder_path)
    print("=" * 50)
    
    if 'plddt' in stats:
        print("pLDDT统计信息:")
        print("-" * 40)
        plddt = stats['plddt']
        print(f"文件数量: {plddt['count']}")
        print(f"平均值: {plddt['mean']:.2f}")
        print(f"中位数: {plddt['median']:.2f}")
        print(f"最小值: {plddt['min']:.2f}")
        print(f"最大值: {plddt['max']:.2f}")
        print(f"标准差: {plddt['std']:.2f}")
        print(f"25百分位数: {plddt['25th_percentile']:.2f}")
        print(f"75百分位数: {plddt['75th_percentile']:.2f}")
        print(f"pLDDT ≥ 50的百分比: {plddt['plddt_50_above']:.2f}%")
        print(f"pLDDT ≥ 70的百分比: {plddt['plddt_70_above']:.2f}%")
        print(f"pLDDT ≥ 80的百分比: {plddt['plddt_80_above']:.2f}%")
        print(f"pLDDT ≥ 90的百分比: {plddt['plddt_90_above']:.2f}%")
        print()
    
    if 'ptm' in stats:
        print("pTM统计信息:")
        print("-" * 40)
        ptm = stats['ptm']
        print(f"文件数量: {ptm['count']}")
        print(f"平均值: {ptm['mean']:.3f}")
        print(f"中位数: {ptm['median']:.3f}")
        print(f"最小值: {ptm['min']:.3f}")
        print(f"最大值: {ptm['max']:.3f}")
        print(f"标准差: {ptm['std']:.3f}")
        print(f"25百分位数: {ptm['25th_percentile']:.3f}")
        print(f"75百分位数: {ptm['75th_percentile']:.3f}")
        print(f"pTM ≥ 0.5的百分比: {ptm['ptm_0.5_above']:.2f}%")
        print(f"pTM ≥ 0.7的百分比: {ptm['ptm_0.7_above']:.2f}%")
        print(f"pTM ≥ 0.8的百分比: {ptm['ptm_0.8_above']:.2f}%")
        print(f"pTM ≥ 0.9的百分比: {ptm['ptm_0.9_above']:.2f}%")
        print()

    if 'combined' in stats:
            print("联合筛选统计 (Intersection Metrics):")
            print("-" * 40)
            combined = stats['combined']
            print(f"参与联合统计的文件数: {combined['count']}")
            print(f"pLDDT > 70 且 pTM > 0.5 的比例: {combined['plddt_70_ptm_0.5_rate']:.2f}%")
            print()
    
    print("=" * 50)

if __name__ == '__main__':
    # 使用示例
    folder_path = 'generation-results-dplm2-goonly-alldata-dm-ca-weight-headclloss-2.0_sn-pn-8wstep'
    folder_path = 'generation-results-dplm2-goonly-alldata-dm-ca-me-scale-0.2-weight-headclloss-2.0_sn-pn-11wstep'
    # folder_path = 'generation-results-dplm2-goonly-alldata-dm-ca-me_sn-pn-4wstep'
    # folder_path = 'generation-chroma'
    # folder_path = 'generation-results-cfpgen_650m'
    # folder_path = 'generation-results-dplm2-goonly-alldata-dm-ca-mf_cf-4wstep'
    # folder_path = 'generation-results-dplm2-goonly-struct'

    
    folder_path = './' + folder_path + '/esmfold_pdb'


    folder_path = '/AIRvePFS/dair/chenxr-data/repo/Denovo-Pinal/generation-pinal-random/esmfold_pdb'
    # folder_path = '/AIRvePFS/dair/chenxr-data/repo/Denovo-Pinal/generation-pinal-len/esmfold_pdb'

    # folder_path = '/AIRvePFS/dair/chenxr-data/repo/ProDVa/evaluations/esmfold_pdb'

    stats = calculate_stats(folder_path)

    if stats:
        print_stats(stats, folder_path)

    # 如果需要计算二级结构统计，取消注释下面的行
    # PDB_DIR = folder_path
    # main_ss()