import pandas as pd
import numpy as np
from Bio.Align import PairwiseAligner
from Bio.Align.substitution_matrices import load
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

from load_all_train_data import load_all_pfam_emb_data
from watch_train_data import extract_motif_info

from tqdm import tqdm
from Bio.PDB import PDBParser, Superimposer, Selection
from Bio.PDB.PDBIO import PDBIO
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
import requests
import os
from io import StringIO


def download_pdb(pdb_id, save_dir="./pdb_files"):
    """下载PDB文件"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    pdb_path = os.path.join(save_dir, f"{pdb_id.lower()}.pdb")
    
    if os.path.exists(pdb_path):
        return pdb_path
    
    try:
        if os.path.exists(f"./data-bin/missed_pdb/AF-{pdb_id}-F1-model_v4.pdb"):
            return f"./data-bin/missed_pdb/AF-{pdb_id}-F1-model_v4.pdb"

        url = f"https://alphafold.ebi.ac.uk/files/AF-{pdb_id}-F1-model_v4.pdb"
        response = requests.get(url)
        if response.status_code == 200:
            with open(pdb_path, 'w') as f:
                f.write(response.text)
            print(f"下载成功: {pdb_id}")
            return pdb_path
        else:
            print(f"无法下载 {pdb_id}")
            return None
    except Exception as e:
        print(f"下载错误 {pdb_id}: {e}")
        return None

def extract_structure_fragment(pdb_file, start_pos, end_pos, chain_id='A'):
    """从PDB文件中提取指定位置的结构片段"""
    parser = PDBParser(QUIET=True)
    
    try:
        structure = parser.get_structure('protein', pdb_file)
        model = structure[0]  # 第一个模型
        
        # 提取指定链和位置的原子坐标
        atoms = []
        residues_coords = []
        
        for residue in model[chain_id]:
            if residue.id[1] >= start_pos and residue.id[1] <= end_pos:
                # 提取Cα原子或其他主链原子
                if 'CA' in residue:
                    atoms.append(residue['CA'])
                    residues_coords.append(residue['CA'].get_coord())
                # 如果需要所有主链原子
                # for atom in residue:
                #     if atom.get_name() in ['N', 'CA', 'C', 'O']:
                #         atoms.append(atom)
                #         residues_coords.append(atom.get_coord())
        
        return np.array(residues_coords), atoms
    except Exception as e:
        print(f"解析PDB文件错误: {e}")
        return None, None

def calculate_rmsd(coords1, coords2):
    """计算两个结构之间的RMSD"""
    if coords1 is None or coords2 is None:
        return float('inf')
    
    # 确保两个结构有相同数量的原子
    min_len = min(len(coords1), len(coords2))
    if min_len == 0:
        return float('inf')
    
    coords1 = coords1[:min_len]
    coords2 = coords2[:min_len]
    
    # 计算RMSD
    diff = coords1 - coords2
    rmsd = np.sqrt(np.sum(diff**2) / min_len)
    return rmsd

def calculate_tm_score(coords1, coords2):
    """计算TM-score (Template Modeling Score)"""
    if coords1 is None or coords2 is None:
        return 0
    
    min_len = min(len(coords1), len(coords2))
    if min_len == 0:
        return 0
    
    coords1 = coords1[:min_len]
    coords2 = coords2[:min_len]
    
    # 简化的TM-score计算
    d0 = 1.24 * (min_len - 15) ** (1/3) - 1.8  # 经验参数
    distances = np.sqrt(np.sum((coords1 - coords2) ** 2, axis=1))
    tm_score = np.sum(1 / (1 + (distances / d0) ** 2)) / min_len
    
    return tm_score

def calculate_gdt_ts(coords1, coords2, thresholds=[1.0, 2.0, 4.0, 8.0]):
    """计算GDT_TS (Global Distance Test Total Score)"""
    if coords1 is None or coords2 is None:
        return 0
    
    min_len = min(len(coords1), len(coords2))
    if min_len == 0:
        return 0
    
    coords1 = coords1[:min_len]
    coords2 = coords2[:min_len]
    
    distances = np.sqrt(np.sum((coords1 - coords2) ** 2, axis=1))
    
    gdt_scores = []
    for threshold in thresholds:
        fraction = np.sum(distances <= threshold) / min_len
        gdt_scores.append(fraction)
    
    gdt_ts = np.mean(gdt_scores)
    return gdt_ts

def align_structures(coords1, coords2):
    """使用Kabsch算法进行结构对齐"""
    if coords1 is None or coords2 is None:
        return coords1, coords2, float('inf')
    
    min_len = min(len(coords1), len(coords2))
    if min_len < 3:  # 需要至少3个点进行对齐
        return coords1, coords2, float('inf')
    
    coords1 = coords1[:min_len]
    coords2 = coords2[:min_len]
    
    # 中心化坐标
    center1 = np.mean(coords1, axis=0)
    center2 = np.mean(coords2, axis=0)
    coords1_centered = coords1 - center1
    coords2_centered = coords2 - center2
    
    # 计算协方差矩阵
    H = coords1_centered.T @ coords2_centered
    
    # SVD分解
    U, S, Vt = np.linalg.svd(H)
    
    # 计算旋转矩阵
    rotation = Vt.T @ U.T
    
    # 确保是右手坐标系
    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = Vt.T @ U.T
    
    # 应用旋转和平移
    coords2_aligned = (coords2_centered @ rotation) + center1
    
    return coords1, coords2_aligned, calculate_rmsd(coords1, coords2_aligned)

def calculate_physical_similarity(pdb_file1, start1, end1, pdb_file2, start2, end2):
    """计算两个结构片段的物理相似度"""
    # 提取结构片段
    coords1, atoms1 = extract_structure_fragment(pdb_file1, start1, end1)
    coords2, atoms2 = extract_structure_fragment(pdb_file2, start2, end2)
    
    if coords1 is None or coords2 is None:
        return {
            'rmsd': float('inf'),
            'tm_score': 0,
            'gdt_ts': 0,
            'aligned_rmsd': float('inf'),
            'success': False
        }
    
    # 计算未对齐的RMSD
    rmsd_unaligned = calculate_rmsd(coords1, coords2)
    
    # 结构对齐后计算RMSD
    coords1_aligned, coords2_aligned, rmsd_aligned = align_structures(coords1, coords2)
    
    # 计算其他相似度指标
    tm_score = calculate_tm_score(coords1_aligned, coords2_aligned)
    gdt_ts = calculate_gdt_ts(coords1_aligned, coords2_aligned)
    
    return {
        'rmsd_unaligned': rmsd_unaligned,
        'rmsd_aligned': rmsd_aligned,
        'tm_score': tm_score,
        'gdt_ts': gdt_ts,
        'coords1_len': len(coords1),
        'coords2_len': len(coords2),
        'success': True
    }

# 主分析函数
def analyze_physical_structure_similarity(data):
    """分析所有蛋白质对的物理结构相似度"""
    results = []
    
    # 首先下载所有需要的PDB文件
    pdb_files = {}
    for item in data:
        pdb_id = item['protein_name']  # 假设protein_name是PDB ID
        pdb_path = download_pdb(pdb_id)
        if pdb_path:
            pdb_files[pdb_id] = pdb_path
    
    # 计算所有蛋白质对的结构相似度
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            protein1 = data[i]
            protein2 = data[j]
            
            pdb1 = pdb_files.get(protein1['protein_name'])
            pdb2 = pdb_files.get(protein2['protein_name'])
            
            if pdb1 and pdb2 and (abs(protein1['end_position'] - protein1['start_position'] - protein2['end_position'] + protein2['start_position']) < 30):

                # print(f"计算 {protein1['protein_name']} vs {protein2['protein_name']}...")
                
                similarity = calculate_physical_similarity(
                    pdb1, protein1['start_position'], protein1['end_position'],
                    pdb2, protein2['start_position'], protein2['end_position']
                )
                
                result = {
                    'protein1': protein1['protein_name'],
                    'protein2': protein2['protein_name'],
                    'start1': protein1['start_position'],
                    'end1': protein1['end_position'],
                    'start2': protein2['start_position'],
                    'end2': protein2['end_position'],
                    **similarity
                }
                results.append(result)
            else:
                # print(f"跳过 {protein1['protein_name']} vs {protein2['protein_name']} (缺少PDB文件)")
                pass
    
    return results

# 可视化结果
def visualize_physical_similarity(results, aa_similarity_matrix, struct_similarity_matrix, protein_names):
    """可视化物理结构相似度结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 提取数据
    physical_metrics = ['rmsd_aligned', 'tm_score', 'gdt_ts']
    physical_labels = ['RMSD (Å)', 'TM-Score', 'GDT_TS']
    
    # 创建物理相似度矩阵
    n_proteins = len(protein_names)
    physical_matrices = {}
    
    for metric in physical_metrics:
        matrix = np.ones((n_proteins, n_proteins))
        for result in results:
            i = protein_names.index(result['protein1'])
            j = protein_names.index(result['protein2'])
            value = result[metric]
            matrix[i][j] = value
            matrix[j][i] = value
        physical_matrices[metric] = matrix
    
    # 绘制热图
    # for idx, (metric, label) in enumerate(zip(physical_metrics, physical_labels)):
    #     sns.heatmap(physical_matrices[metric], annot=True, cmap='viridis_r' if metric == 'rmsd_aligned' else 'viridis',
    #                fmt='.3f', ax=axes[0, idx], cbar_kws={'label': label},
    #                xticklabels=protein_names, yticklabels=protein_names)
    #     axes[0, idx].set_title(f'physical similarity: {label}')
    #     axes[0, idx].tick_params(axis='x', rotation=45)
    #     axes[0, idx].tick_params(axis='y', rotation=0)
    
    # 相关性分析
    aa_values = []
    struct_values = []
    physical_values = []
    
    for result in results:
        i = protein_names.index(result['protein1'])
        j = protein_names.index(result['protein2'])
        
        aa_values.append(aa_similarity_matrix[i][j])
        struct_values.append(struct_similarity_matrix[i][j])
        physical_values.append(result['tm_score'])  # 使用TM-score作为物理相似度代表
    
    # AA序列 vs 物理结构
    axes[1, 0].scatter(aa_values, physical_values, alpha=0.7, s=100)
    axes[1, 0].set_xlabel('aa_similarity')
    axes[1, 0].set_ylabel('TM-Score (physical similarity)')
    corr_aa_phys = np.corrcoef(aa_values, physical_values)[0, 1]
    axes[1, 0].set_title(f'AA sequence vs physical similarity\ncorrelation coefficient: {corr_aa_phys:.3f}')
    axes[1, 0].grid(True, alpha=0.3)

    # aa_seq vs rmsd_aligned
    axes[0, 1].scatter(aa_values, [result['rmsd_aligned'] for result in results], alpha=0.7, s=100)
    axes[0, 1].set_xlabel('aa_similarity')
    axes[0, 1].set_ylabel('RMSD (Å)')
    corr_aa_rmsd = np.corrcoef(aa_values, [result['rmsd_aligned'] for result in results])[0, 1]
    axes[0, 1].set_title(f'AA sequence vs RMSD\ncorrelation coefficient: {corr_aa_rmsd:.3f}')
    axes[0, 1].grid(True, alpha=0.3)

    
    # 结构序列 vs 物理结构
    axes[1, 1].scatter(struct_values, physical_values, alpha=0.7, s=100)
    axes[1, 1].set_xlabel('struct_similarity')
    axes[1, 1].set_ylabel('TM-Score (physical similarity)')
    corr_struct_phys = np.corrcoef(struct_values, physical_values)[0, 1]
    axes[1, 1].set_title(f'structural sequence vs physical similarity\ncorrelation coefficient: {corr_struct_phys:.3f}')
    axes[1, 1].grid(True, alpha=0.3)

    # struct_seq vs rmsd_aligned
    axes[1, 2].scatter(struct_values, [result['rmsd_aligned'] for result in results], alpha=0.7, s=100)
    axes[1, 2].set_xlabel('struct_similarity')
    axes[1, 2].set_ylabel('RMSD (Å)')
    corr_struct_rmsd = np.corrcoef(struct_values, [result['rmsd_aligned'] for result in results])[0, 1]
    axes[1, 2].set_title(f'structural sequence vs RMSD\ncorrelation coefficient: {corr_struct_rmsd:.3f}')
    axes[1, 2].grid(True, alpha=0.3)


    
    # # 三种相似度比较
    # x_pos = np.arange(len(results))
    # width = 0.25
    
    # axes[1, 2].bar(x_pos - width, aa_values, width, label='AA sequence', alpha=0.7)
    # axes[1, 2].bar(x_pos, struct_values, width, label='structural sequence', alpha=0.7)
    # axes[1, 2].bar(x_pos + width, physical_values, width, label='physical similarity', alpha=0.7)
    
    # axes[1, 2].set_xlabel('protein pairs')
    # axes[1, 2].set_ylabel('similarity')
    # axes[1, 2].set_title('three types of similarity comparison')
    # axes[1, 2].legend()
    # axes[1, 2].set_xticks(x_pos)
    # axes[1, 2].set_xticklabels([f"{r['protein1']}-{r['protein2']}" for r in results], rotation=45)
    
    plt.tight_layout()
    plt.show()
    plt.savefig('physical_similarity.png')
    
    return physical_matrices

if __name__ == '__main__':

    train_data = load_all_pfam_emb_data("train")
    # for i in range(len(train_data)):
    #     if train_data[i].get("motif_position_s", None) is not None:
    #         print(train_data[i])
    #         exit()
    # print(train_data[1])

    motif_dict = extract_motif_info(train_data, max_motif_length=300)

    sorted_top_motifs = sorted(motif_dict.items(), key=lambda x: len(x[1]), reverse=True)
    # 你的数据
    for i, (motif_desc, motifs) in enumerate(sorted_top_motifs, 1):
        data = motifs[:60]
        break

    def calculate_aa_similarity(seq1, seq2):
        """计算氨基酸序列相似度"""
        aligner = PairwiseAligner()
        aligner.substitution_matrix = load("BLOSUM62")
        alignments = aligner.align(seq1, seq2)
        best_alignment = alignments[0]
        score = best_alignment.score
        # 归一化到0-1范围
        max_score = max(aligner.align(seq1, seq1).score, aligner.align(seq2, seq2).score)
        return score / max_score if max_score > 0 else 0

    def calculate_struct_similarity(struct1, struct2):
        """计算结构序列相似度 - 使用编辑距离"""
        # 将结构序列转换为字符串进行比较
        str1 = ' '.join(struct1)
        str2 = ' '.join(struct2)
        
        # 使用简单的编辑距离
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein_distance(str1, str2)
        max_len = max(len(str1), len(str2))
        similarity = 1 - (distance / max_len) if max_len > 0 else 0
        return similarity

    def create_similarity_matrix(data, similarity_func, seq_type='aa'):
        """创建相似度矩阵"""
        n = len(data)
        matrix = np.zeros((n, n))
        
        for i in tqdm(range(n)):
            for j in range(i, n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    if seq_type == 'aa':
                        seq1 = data[i]['aa_sequence']
                        seq2 = data[j]['aa_sequence']
                    else:
                        seq1 = data[i]['struct_sequence']
                        seq2 = data[j]['struct_sequence']
                    
                    similarity = similarity_func(seq1, seq2)
                    matrix[i][j] = similarity
                    matrix[j][i] = similarity
        
        return matrix

    # 计算相似度矩阵
    print("计算氨基酸序列相似度...")
    aa_similarity_matrix = create_similarity_matrix(data, calculate_aa_similarity, 'aa')

    print("计算结构序列相似度...")
    struct_similarity_matrix = create_similarity_matrix(data, calculate_struct_similarity, 'struct')

    # 获取蛋白质名称
    protein_names = [item['protein_name'] for item in data]

    # 创建DataFrame以便更好地显示
    aa_sim_df = pd.DataFrame(aa_similarity_matrix, 
                            index=protein_names, 
                            columns=protein_names)

    struct_sim_df = pd.DataFrame(struct_similarity_matrix, 
                            index=protein_names, 
                            columns=protein_names)

    # print("氨基酸序列相似度矩阵:")
    # print(aa_sim_df.round(3))
    print("氨基酸序列相似度均值:")
    print(aa_sim_df.mean().round(3))
    # print("\n结构序列相似度矩阵:")
    # print(struct_sim_df.round(3))
    print("结构序列相似度均值:")
    print(struct_sim_df.mean().round(3))

    # 计算两种相似度的相关性
    aa_values = []
    struct_values = []

    for i in range(len(data)):
        for j in range(i+1, len(data)):
            aa_values.append(aa_similarity_matrix[i][j])
            struct_values.append(struct_similarity_matrix[i][j])

    correlation = np.corrcoef(aa_values, struct_values)[0, 1]
    print(f"\n氨基酸序列相似度与结构序列相似度的相关系数: {correlation:.3f}")

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 氨基酸序列相似度热图
    sns.heatmap(aa_sim_df, annot=True, cmap='YlOrRd', fmt='.3f', 
                ax=axes[0, 0], cbar_kws={'label': 'sim'})
    axes[0, 0].set_title('aa_similarity_matrix')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].tick_params(axis='y', rotation=0)

    # 结构序列相似度热图
    sns.heatmap(struct_sim_df, annot=True, cmap='YlOrRd', fmt='.3f', 
                ax=axes[0, 1], cbar_kws={'label': 'sim'})
    axes[0, 1].set_title('struct_similarity_matrix')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].tick_params(axis='y', rotation=0)

    # 散点图显示关系
    axes[1, 0].scatter(aa_values, struct_values, alpha=0.7, s=100)
    axes[1, 0].set_xlabel('aa_sim')
    axes[1, 0].set_ylabel('struct_sim')
    axes[1, 0].set_title(f'aa_sim vs struct_sim\ncorrelation: {correlation:.3f}')
    axes[1, 0].grid(True, alpha=0.3)

    # 添加趋势线
    if len(aa_values) > 1:
        z = np.polyfit(aa_values, struct_values, 1)
        p = np.poly1d(z)
        axes[1, 0].plot(aa_values, p(aa_values), "r--", alpha=0.8)

    # 相似度差异分析
    similarity_differences = [aa - struct for aa, struct in zip(aa_values, struct_values)]
    axes[1, 1].bar(range(len(similarity_differences)), similarity_differences)
    axes[1, 1].set_xlabel('protein pair')
    axes[1, 1].set_ylabel('sim_diff (aa_sim - struct_sim)')
    axes[1, 1].set_title('aa_sim vs struct_sim sim_diff')
    axes[1, 1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.savefig('aa_struct_sim.png')

    print("开始分析物理结构相似度...")
    physical_results = analyze_physical_structure_similarity(data)
    visualize_physical_similarity(physical_results, aa_similarity_matrix, struct_similarity_matrix, protein_names)
    
    # # 显示结果
    # print("\n=== 物理结构相似度分析结果 ===")
    # for result in physical_results:
    #     print(f"\n{result['protein1']} vs {result['protein2']}:")
    #     print(f"  对齐后RMSD: {result['rmsd_aligned']:.3f} Å")
    #     print(f"  TM-Score: {result['tm_score']:.3f}")
    #     print(f"  GDT_TS: {result['gdt_ts']:.3f}")
    #     print(f"  片段长度: {result['coords1_len']} vs {result['coords2_len']}")

    # 详细分析每对蛋白质
    # print("\n详细蛋白质对分析:")
    # pair_count = 0
    # for i in range(len(data)):
    #     for j in range(i+1, len(data)):
    #         pair_count += 1
    #         print(f"\n蛋白质对 {pair_count}: {data[i]['protein_name']} vs {data[j]['protein_name']}")
    #         print(f"  氨基酸序列相似度: {aa_similarity_matrix[i][j]:.3f}")
    #         print(f"  结构序列相似度: {struct_similarity_matrix[i][j]:.3f}")
    #         print(f"  相似度差异: {aa_similarity_matrix[i][j] - struct_similarity_matrix[i][j]:.3f}")
            
    #         # 显示序列信息
    #         print(f"  AA序列1: {data[i]['aa_sequence']}")
    #         print(f"  AA序列2: {data[j]['aa_sequence']}")
    #         print(f"  结构序列1: {' '.join(data[i]['struct_sequence'])}")
    #         print(f"  结构序列2: {' '.join(data[j]['struct_sequence'])}")