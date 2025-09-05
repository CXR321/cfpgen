import os
import re
import numpy as np

def extract_plddt(filename):
    """从文件名中提取pLDDT值"""
    match = re.search(r'plddt_([\d.]+)\.pdb$', filename)
    if match:
        return float(match.group(1))
    return None

def calculate_stats(folder_path):
    plddt_values = []
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdb'):
            plddt = extract_plddt(filename)
            if plddt is not None:
                plddt_values.append(plddt)
    
    if not plddt_values:
        print("未找到包含pLDDT值的PDB文件")
        return

    plddt_values = np.array(plddt_values)

    def calculate_percentage_above(threshold):
        return np.sum(plddt_values >= threshold) / len(plddt_values) * 100

    # 计算统计信息
    stats = {
        'count': len(plddt_values),
        'mean': np.mean(plddt_values),
        'median': np.median(plddt_values),
        'min': np.min(plddt_values),
        'max': np.max(plddt_values),
        'std': np.std(plddt_values),
        '25th_percentile': np.percentile(plddt_values, 25),
        '75th_percentile': np.percentile(plddt_values, 75),

        'plddt_70_above': calculate_percentage_above(70),
        'plddt_90_above': calculate_percentage_above(90),
    }
    
    return stats

# 使用示例
# folder_path = './generation-results-dplm2/esmfold_pdb/'  # 替换为你的文件夹路径
# folder_path = './generation-results-cfpgen_650m/esmfold_pdb/'
folder_path = './generation-results-dplm2-diff-cross/esmfold_pdb/'
stats = calculate_stats(folder_path)

if stats:
    print("文件夹路径:", folder_path)
    print("pLDDT统计信息:")
    print(f"文件数量: {stats['count']}")
    print(f"平均值: {stats['mean']:.2f}")
    print(f"中位数: {stats['median']:.2f}")
    print(f"最小值: {stats['min']:.2f}")
    print(f"最大值: {stats['max']:.2f}")
    print(f"标准差: {stats['std']:.2f}")
    print(f"25百分位数: {stats['25th_percentile']:.2f}")
    print(f"75百分位数: {stats['75th_percentile']:.2f}")
    print(f"pLDDT大于70的百分比: {stats['plddt_70_above']:.2f}%")
    print(f"pLDDT大于90的百分比: {stats['plddt_90_above']:.2f}%")