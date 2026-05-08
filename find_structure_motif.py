# import Bio.PDB
# from Bio.PDB import PDBParser, Superimposer
# import numpy as np

# def get_residues(structure, chain_id=None):
#     """
#     获取结构中所有的残基列表。
#     如果指定了 chain_id，只获取该链的残基。
#     """
#     residues = []
#     for model in structure:
#         for chain in model:
#             if chain_id and chain.id != chain_id:
#                 continue
#             for residue in chain:
#                 # 过滤掉水分子和异质原子，只保留标准氨基酸
#                 if Bio.PDB.is_aa(residue, standard=True):
#                     residues.append(residue)
#     return residues

# def get_ca_atoms(residues):
#     """
#     从残基列表中提取 CA (Alpha Carbon) 原子。
#     """
#     atoms = []
#     for r in residues:
#         if 'CA' in r:
#             atoms.append(r['CA'])
#     return atoms

# def find_best_structural_match(pdb1_path, pdb2_path, motif_chain, motif_start, motif_end, output_pml="result.pml"):
#     """
#     在 pdb2 中寻找与 pdb1 指定片段结构最相似的区域。
#     """
#     parser = PDBParser(QUIET=True)
    
#     # 1. 解析结构
#     struct1 = parser.get_structure("PDB1", pdb1_path)
#     struct2 = parser.get_structure("PDB2", pdb2_path)
    
#     # 2. 提取 Query Motif (来自 PDB1)
#     # 注意：这里假设 pdb 文件中的残基 ID 是连续整数，如果不是，需根据实际情况调整筛选逻辑
#     all_res1 = get_residues(struct1, motif_chain)
#     motif_res = [r for r in all_res1 if motif_start <= r.id[1] <= motif_end]
    
#     motif_atoms = get_ca_atoms(motif_res)
#     motif_len = len(motif_atoms)
    
#     if motif_len == 0:
#         print("错误：在 PDB1 中未找到指定的 Motif 残基，请检查链 ID 和残基编号。")
#         return

#     print(f"Motif 长度: {motif_len} 个残基 (PDB1 Chain {motif_chain}: {motif_start}-{motif_end})")

#     # 3. 准备 Target (PDB2) 进行滑动窗口搜索
#     target_res = get_residues(struct2) # 搜索 PDB2 的所有链
#     target_atoms_full = get_ca_atoms(target_res)
    
#     best_rmsd = float('inf')
#     best_window_start_index = -1
#     best_rotation = None
#     best_translation = None
    
#     sup = Superimposer()
    
#     # 4. 滑动窗口遍历
#     # 窗口大小必须与 motif 的原子数一致
#     print("正在 PDB2 中进行结构搜索...")
    
#     for i in range(len(target_atoms_full) - motif_len + 1):
#         window_atoms = target_atoms_full[i : i + motif_len]
        
#         # 确保窗口内的原子都在同一个链上（跨链的结构比对通常无意义）
#         chain_ids = set(atom.get_parent().get_parent().id for atom in window_atoms)
#         if len(chain_ids) > 1:
#             continue
            
#         # 设置坐标
#         # fixed: motif (我们要找的样子), moving: window (候选区域)
#         sup.set_atoms(motif_atoms, window_atoms)
        
#         current_rmsd = sup.rms
        
#         if current_rmsd < best_rmsd:
#             best_rmsd = current_rmsd
#             best_window_start_index = i
#             best_rotation = sup.rotran[0]
#             best_translation = sup.rotran[1]

#     # 5. 输出结果
#     if best_window_start_index != -1:
#         best_res_start = target_atoms_full[best_window_start_index].get_parent()
#         best_res_end = target_atoms_full[best_window_start_index + motif_len - 1].get_parent()
        
#         target_chain_id = best_res_start.get_parent().id
        
#         print("-" * 30)
#         print(f"找到最佳匹配区域！")
#         print(f"最小 RMSD: {best_rmsd:.4f} Å")
#         print(f"PDB2 位置: Chain {target_chain_id}, Residue {best_res_start.id[1]} - {best_res_end.id[1]}")
#         print("-" * 30)
        
#         # 6. 生成 PyMOL 脚本
#         # 逻辑：加载两个 pdb，将 pdb2 移动到匹配 pdb1 的位置（或者反过来），然后高亮
        
#         # 这里我们生成一个脚本，在 PyMOL 中选中这两个片段
#         with open(output_pml, "w") as f:
#             f.write(f"load {pdb1_path}, prot1\n")
#             f.write(f"load {pdb2_path}, prot2\n")
#             f.write("bg_color white\n")
#             f.write("hide everything\n")
#             f.write("show cartoon\n")
            
#             # 定义选区
#             # PDB1 的 Motif
#             f.write(f"select motif_original, prot1 and chain {motif_chain} and resi {motif_start}-{motif_end}\n")
#             f.write("color gray80, prot1\n")
#             f.write("color blue, motif_original\n")
            
#             # PDB2 的 匹配区域
#             f.write(f"select motif_match, prot2 and chain {target_chain_id} and resi {best_res_start.id[1]}-{best_res_end.id[1]}\n")
#             f.write("color gray80, prot2\n")
#             f.write("color red, motif_match\n")
            
#             # 对齐
#             # 使用 PyMOL 的 align 命令，强制将找到的两个片段对齐
#             f.write(f"align motif_match, motif_original\n")
            
#             # 视觉优化
#             f.write("zoom motif_original\n")
#             f.write("set cartoon_fancy_helices, 1\n")
#             f.write("deselect\n")
            
#         print(f"可视化脚本已生成: {output_pml}")
#         print("请在 PyMOL 中打开该文件 (File -> Run Script...)")
        
#     else:
#         print("未找到合适的匹配区域（可能是目标蛋白比 Motif 短）。")

# # ================= 配置区域 =================
# # 在这里修改你的文件路径和参数
# # 示例：假设 pdb1 是短肽或者包含关键结构的蛋白，pdb2 是要搜索的目标蛋白

# # 输入文件路径 (可以使用绝对路径或相对路径)
# file1 = "AF-Q48KZ8-F1-model_v6.pdb"  # 你的第一个 PDB 文件
# file2 = "SEQUENCE_ID=Q48KZ8_L=223_plddt_94.37220764160156_ptm_0.941.pdb" # 你的第二个 PDB 文件
# # file2 = './pdbs/AF-A9HEL1-F1-model_v6.pdb'

# # cfp
# # file2 = 'SEQUENCE_ID=Q48KZ8_L=223_plddt_36.94288635253906_ptm_0.181.pdb'

# # 定义第一个 PDB 中你感兴趣的片段
# # {'go_term': 'methenyltetrahydrofolate cyclohydrolase activity'
# query_chain = "A"    # 链 ID
# query_start = 34     # 起始残基编号
# query_end   = 56     # 结束残基编号

# # {'go_term': 'methylenetetrahydrofolate dehydrogenase (NADP+) activity'
# query_chain = "A"    # 链 ID
# query_start = 124     # 起始残基编号 124
# # query_start = 1

# query_end   = 280     # 结束残基编号 280
# # query_end   = 100


# if __name__ == "__main__":
#     find_best_structural_match(file1, file2, query_chain, query_start, query_end)

import Bio.PDB
from Bio.PDB import PDBParser, Superimposer
from Bio.SeqUtils import seq1
from Bio import Align
import numpy as np

# ================= 工具函数 =================

def get_chain_sequence_and_mapping(structure, chain_id, start_res_id=None, end_res_id=None):
    """
    提取指定链的：
    1. 氨基酸序列字符串 (Sequence String)
    2. 序列中每个字符对应的残基对象列表 (Mapping List)
    """
    res_list = []
    seq_str = ""
    
    # 获取第一个模型
    model = list(structure)[0]
    
    # 检查链是否存在
    if chain_id not in model:
        print(f"Error: Chain {chain_id} not found in structure.")
        return "", []

    chain = model[chain_id]
    
    for residue in chain:
        # 只处理标准氨基酸
        if not Bio.PDB.is_aa(residue, standard=True):
            continue
            
        # 检查编号范围 (如果有指定)
        rid = residue.id[1]
        if start_res_id is not None and rid < start_res_id:
            continue
        if end_res_id is not None and rid > end_res_id:
            continue
            
        # 获取单字母代码
        try:
            # residue.resname 是三字母 (e.g., ALA)，转为 A
            one_letter = seq1(residue.resname)
            seq_str += one_letter
            res_list.append(residue)
        except Exception:
            continue
            
    return seq_str, res_list

def perform_sequence_guided_superposition(pdb1_path, pdb2_path, q_chain, q_start, q_end, output_pml="seq_align_result.pml"):
    """
    流程：
    1. 提取 PDB1 Motif 序列 和 PDB2 全长序列
    2. 进行序列比对
    3. 根据比对结果提取对应的 CA 原子对
    4. 计算 RMSD 并生成 PyMOL
    """
    parser = PDBParser(QUIET=True)
    struct1 = parser.get_structure("Query", pdb1_path)
    struct2 = parser.get_structure("Target", pdb2_path)
    
    # 1. 获取序列和映射
    print(f"正在提取 Query Motif (Chain {q_chain}: {q_start}-{q_end})...")
    seq1_str, res1_objs = get_chain_sequence_and_mapping(struct1, q_chain, q_start, q_end)
    
    # 假设 Target 我们搜索全长，通常 Target 只有一条链或者我们只关心 Chain A
    # 这里为了通用，假设 Target 的主要链也是 A，或者你可以遍历 Target 所有链
    target_chain_id = "A" # <--- 如果你的 Target 链名不同，请修改这里
    print(f"正在提取 Target (Chain {target_chain_id})...")
    seq2_str, res2_objs = get_chain_sequence_and_mapping(struct2, target_chain_id)
    
    if not seq1_str or not seq2_str:
        print("错误：无法提取序列，请检查 PDB 文件或残基编号。")
        return

    print(f"Query 序列长度: {len(seq1_str)}")
    print(f"Target 序列长度: {len(seq2_str)}")

    # 2. 序列比对 (Local Alignment)
    # 使用 PairwiseAligner (Biopython > 1.78)
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'  # 局部比对：在长序列中找短 Motif
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
    
    alignments = aligner.align(seq1_str, seq2_str)
    
    if not alignments:
        print("未找到序列比对结果。")
        return
        
    # 取得分最高的比对结果
    best_aln = alignments[0]
    print("-" * 30)
    print("序列比对完成。最佳比对得分:", best_aln.score)
    print(best_aln) # 打印比对图示
    
    # 3. 提取用于结构叠加的原子对
    # best_aln.indices 会返回两个数组，分别是 seq1 和 seq2 中对齐的索引
    # 注意：alignments 对象的处理方式稍微复杂，我们需要遍历对齐路径
    
    aligned_atoms_1 = [] # Query (Fixed)
    aligned_atoms_2 = [] # Target (Moving)
    
    # 获取对齐的坐标 (indices of aligned residues)
    # aligned 属性返回两个元组，表示两个序列中对齐的片段
    # 我们需要将其展开为一一对应的残基索引
    path = best_aln.path # path 是 (index1, index2) 的列表
    
    # 遍历路径，只收集 match 的部分 (去除 gap)
    # 这里的逻辑是：如果 seq1[i] 和 seq2[j] 对齐，且都不是 gap
    
    # 获取比对后的两个字符串 pattern
    # format() 返回类似 ("A-GC", "ATGC")
    # aligned_seq1, aligned_seq2 = best_aln[0], best_aln[1] 
    
    # 这种方式解析具体 residue index 比较繁琐，我们用更直接的 indices 迭代法：
    # 利用 best_aln.aligned 得到对齐片段的 ranges
    # aligned 结构: ( [(start1, end1), ...], [(start2, end2), ...] )
    
    ranges1, ranges2 = best_aln.aligned
    
    # 收集 Target 上匹配区域的起止 ID，用于 PyMOL 显示
    target_match_start_resid = None
    target_match_end_resid = None
    
    for (start1, end1), (start2, end2) in zip(ranges1, ranges2):
        # 这一段是连续对齐的 (虽然可能中间有 mismatch，但在逻辑上是无 gap 的)
        length = end1 - start1 
        # (理论上 end1-start1 应该等于 end2-start2)
        
        for k in range(length):
            idx1 = start1 + k
            idx2 = start2 + k
            
            r1 = res1_objs[idx1]
            r2 = res2_objs[idx2]
            
            # 记录 Target 范围
            if target_match_start_resid is None: target_match_start_resid = r2.id[1]
            target_match_end_resid = r2.id[1]
            
            # 只有当两个残基都有 CA 原子时才加入计算
            if 'CA' in r1 and 'CA' in r2:
                aligned_atoms_1.append(r1['CA'])
                aligned_atoms_2.append(r2['CA'])

    if len(aligned_atoms_1) < 3:
        print("错误：有效对齐的原子数太少 (<3)，无法进行结构叠加。")
        return

    print(f"结构叠加原子对数量: {len(aligned_atoms_1)}")

    # 4. 计算 RMSD
    sup = Superimposer()
    sup.set_atoms(aligned_atoms_1, aligned_atoms_2)
    
    print(f"基于序列对齐的 RMSD: {sup.rms:.4f} Å")
    print("-" * 30)
    
    # 5. 生成 PyMOL 脚本
    with open(output_pml, "w") as f:
        f.write(f"load {pdb1_path}, query_prot\n")
        f.write(f"load {pdb2_path}, target_prot\n")
        f.write("bg_color white\n")
        f.write("hide everything\n")
        f.write("show cartoon\n")
        
        # 定义 Query Motif 选区
        f.write(f"select query_motif, query_prot and chain {q_chain} and resi {q_start}-{q_end}\n")
        f.write("color blue, query_motif\n")
        
        # 定义 Target Match 选区 (根据序列比对找到的范围)
        # 注意：这里使用的是序列比对覆盖的这一段范围
        if target_match_start_resid and target_match_end_resid:
            f.write(f"select target_match, target_prot and chain {target_chain_id} and resi {target_match_start_resid}-{target_match_end_resid}\n")
            f.write("color red, target_match\n")
            print(f"Target 匹配区域: Chain {target_chain_id} Residues {target_match_start_resid}-{target_match_end_resid}")
        
        # 进行对齐 (Align the matched region to the query motif)
        # 这里的 align 命令会让 PyMOL 基于选区做拟合，视觉上会非常接近我们算的 RMSD
        f.write(f"align target_match, query_motif\n")
        
        f.write("zoom query_motif\n")
        f.write("deselect\n")
        
    print(f"脚本已生成: {output_pml}")


# ================= 配置区域 =================

# 输入文件路径
file1 = "AF-Q48KZ8-F1-model_v6.pdb"  # Query (包含 Motif)
file2 = "SEQUENCE_ID=Q48KZ8_L=223_plddt_94.37220764160156_ptm_0.941.pdb" # Target (被搜索)

# 定义 Query Motif (来自 PDB1)
query_chain = "A"    
query_start = 124    
query_end   = 280    

if __name__ == "__main__":
    perform_sequence_guided_superposition(file1, file2, query_chain, query_start, query_end)