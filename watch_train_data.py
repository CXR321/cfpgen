import pickle
import matplotlib.pyplot as plt
import numpy as np
from sympy import intersection
import random

# exit()

def plot_coverage_distribution(data):
    """
    绘制motif覆盖率的分布直方图
    """
    # 收集所有蛋白质的motif覆盖率
    motif_coverages = []
    go_motif_coverages = []
    fully_covered_proteins = []  # 记录哪些蛋白质的GO标签被完全覆盖
    partially_covered_count = []  # 记录哪些蛋白质的GO标签被部分覆盖

    partially_covered_proteins = []  # 记录哪些蛋白质的GO标签被部分覆盖
    no_coverage_proteins = []  # 记录哪些蛋白质的GO标签没有被覆盖
    
    for protein in data:
        seq_length = len(protein['sequence'])
        go_f_terms = protein['go_numbers'].get('F', [])
        
        # 计算motif_segment覆盖率
        motif_segments = protein.get('motif', [])
        if motif_segments:
            motif_lengths = [motif['end'] - motif['start'] + 1 for motif in motif_segments]
            avg_motif_coverage = sum(motif_lengths) / len(motif_lengths) / seq_length
            motif_coverages.append(avg_motif_coverage)
        
        # 计算有GO注释的pfam_motif覆盖率
        pfam_motifs = protein.get('pfam_motif', [])
        go_motif_lengths = []
        pfam_go_terms = set()  # 记录当前蛋白质的所有GO注释
        
        for pfam in pfam_motifs:
            if pfam.get('strong_go_id') and len(pfam['strong_go_id']) >= 1:
                motif_length = pfam['end'] - pfam['start'] + 1
                go_motif_lengths.append(motif_length)
                pfam_go_terms.update(pfam['strong_go_id'])
        
        if go_motif_lengths:
            avg_go_coverage = sum(go_motif_lengths) / len(go_motif_lengths) / seq_length
            go_motif_coverages.append(avg_go_coverage)

        # 检查GO标签是否被完全覆盖
        is_fully_covered = False
        if go_f_terms and pfam_go_terms:
            is_fully_covered = set(go_f_terms).issubset(pfam_go_terms)
            if is_fully_covered:
                fully_covered_proteins.append(1)
            else:
                fully_covered_proteins.append(0)
        else:
            fully_covered_proteins.append(0)

        if set(go_f_terms).intersection(pfam_go_terms):
            partially_covered_count.append(1)
            if not is_fully_covered:
                partially_covered_proteins.append(protein)
        else:
            partially_covered_count.append(0)
            no_coverage_proteins.append(protein)


    print(f"fully covered proteins: {sum(fully_covered_proteins)} {sum(fully_covered_proteins) / len(fully_covered_proteins) * 100:.2f}")
    print(f"partially covered proteins: {sum(partially_covered_count)} {sum(partially_covered_count) / len(partially_covered_count) * 100:.2f}")

    print(f"partially covered protein example: {partially_covered_proteins[100]}")
    print(f"no coverage protein example: {no_coverage_proteins[100]}")

    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 绘制motif覆盖率分布
    if motif_coverages:
        bins = np.linspace(0, 1, 11)  # 0-1之间分成10个区间
        counts, bin_edges = np.histogram(motif_coverages, bins=bins)
        percentages = counts / len(motif_coverages) * 100
        
        ax1.bar(bin_edges[:-1], percentages, width=0.1, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Motif Coverage')
        ax1.set_ylabel('Percentage of Proteins (%)')
        ax1.set_title('Distribution of Motif Segment Coverage')
        ax1.grid(True, alpha=0.3)
        
        # 在柱子上添加百分比标签
        for i, (count, percentage) in enumerate(zip(counts, percentages)):
            if count > 0:
                ax1.text(bin_edges[i] + 0.05, percentage + 1, f'{percentage:.1f}%', 
                        ha='center', va='bottom', fontsize=9)
    
    # 绘制GO motif覆盖率分布
    if go_motif_coverages:
        bins = np.linspace(0, 1, 11)
        counts, bin_edges = np.histogram(go_motif_coverages, bins=bins)
        percentages = counts / len(go_motif_coverages) * 100
        
        ax2.bar(bin_edges[:-1], percentages, width=0.1, alpha=0.7, 
                edgecolor='black', color='orange')
        ax2.set_xlabel('GO Motif Coverage')
        ax2.set_ylabel('Percentage of Proteins (%)')
        ax2.set_title('Distribution of GO-annotated PFAM Motif Coverage')
        ax2.grid(True, alpha=0.3)
        
        # 在柱子上添加百分比标签
        for i, (count, percentage) in enumerate(zip(counts, percentages)):
            if count > 0:
                ax2.text(bin_edges[i] + 0.05, percentage + 1, f'{percentage:.1f}%', 
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()

    plt.savefig('motif_coverage_distribution.png')
    
    # 返回统计数据
    return {
        'motif_coverages': motif_coverages,
        'go_motif_coverages': go_motif_coverages,
        'total_proteins': len(data),
        'proteins_with_motif': len(motif_coverages),
        'proteins_with_go_motif': len(go_motif_coverages)
    }


if __name__ == '__main__':

    path = "data-bin/uniprotKB/cfpgen_general_dataset/train_data_motif_emb_<200.pkl"
    path = "data-bin/uniprotKB/cfpgen_general_dataset/train.pkl"
    path = "data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added.pkl"
    path = "data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif.pkl"

    path = "data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl"

    with open(path, "rb") as f:
        train_data = pickle.load(f)


    stats = plot_coverage_distribution(train_data)  # 替换为您的完整数据集

    print(f"总蛋白质数量: {stats['total_proteins']}")
    print(f"包含motif的蛋白质数量: {stats['proteins_with_motif']}")
    print(f"包含GO注释motif的蛋白质数量: {stats['proteins_with_go_motif']}")

    # print(train_data[0])
    # exit()
    # function_name = "GO:0003723"
    function_name = "GO:0000034"
    function_name = "GO:0046789"
    function_name = "GO:0039660"

    match_pdb = []

    random.seed(42)

    # for pdb in train_data:
    #     if function_name in pdb["go_numbers"]['F']:
    #         match_pdb.append(pdb)

    for pdb in train_data:
        if pdb.get("pfam_emb", None) is not None:
            match_pdb.append(pdb)

    random_samples = random.sample(match_pdb, min(10, len(match_pdb)))
    for i, sample in enumerate(random_samples):
        print(f"样本 {i+1}: {sample}")
    print("============================")
    # print(train_data[0])