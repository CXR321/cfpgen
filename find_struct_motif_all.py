# import os
# import requests
# import Bio.PDB
# from Bio.PDB import PDBParser, MMCIFParser, Superimposer
# import numpy as np

# # ================= 配置区域 =================

# # 1. 你的生成蛋白 (被搜索的目标) - 假设这个依然是 PDB 格式
# MY_GENERATED_PDB = "SEQUENCE_ID=Q48KZ8_L=223_plddt_94.37220764160156_ptm_0.941.pdb"
# MY_CHAIN_ID = "A" 

# # 2. 真实蛋白列表 (Uniprot ID, Motif Start, Motif End)
# TARGETS_DATA = [
#     ('Q6F9E6', 131, 291),
#     ('A2RVV7', 184, 348),
#     ('Q7WAC0', 124, 279),
#     ('Q7VZC8', 124, 279),
#     ('Q47ZE0', 129, 277),
#     ('Q1J118', 126, 280),
#     ('Q24ZZ6', 123, 279),
#     ('Q0RNE2', 120, 274),
#     ('Q2JCX4', 120, 274),
#     ('Q39Z32', 122, 284),
#     ('Q74GN1', 122, 284),
#     ('Q5V3D9', 122, 285),
#     ('Q31HQ4', 124, 279),
#     ('Q74J95', 123, 280),
#     ('Q1DA76', 122, 287),
#     ('A1SED2', 125, 281),
#     ('A1R228', 124, 279),
#     ('A1B5A7', 124, 281),
#     ('A4XTZ3', 124, 280),
#     ('B0KJG3', 130, 286),
#     ('Q88LI7', 124, 285),
#     ('Q87YR0', 124, 280),
#     ('Q4ZVN1', 124, 280),
#     ('Q2KCC6', 125, 288),
#     ('Q989A7', 123, 288),
#     ('Q92T88', 123, 288),
#     ('A5V4U1', 123, 282),
#     ('Q0SFQ8', 123, 277),
#     ('Q167Y9', 125, 285),
#     ('P24186', 124, 278)
# ]

# # ================= 工具函数 =================

# def download_alphafill_cif(uniprot_id, output_dir="pdbs"):
#     """
#     从 AlphaFill 下载 CIF 文件
#     URL示例: https://alphafill.eu/v1/aff/Q167Y9
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     # 注意：AlphaFill 下载的是 cif 文件
#     filename = f"{uniprot_id}.cif"
#     filepath = os.path.join(output_dir, filename)
    
#     if os.path.exists(filepath): 
#         return filepath
    
#     url = f"https://alphafill.eu/v1/aff/{uniprot_id}"
#     print(f"正在下载: {url} ...")
    
#     try:
#         r = requests.get(url)
#         if r.status_code == 200:
#             with open(filepath, "wb") as f: 
#                 f.write(r.content)
#             return filepath
#         else:
#             print(f"下载失败 {uniprot_id}: HTTP {r.status_code}")
#     except Exception as e:
#         print(f"下载异常 {uniprot_id}: {e}")
        
#     return None

# def get_ca_atoms_from_chain(structure, chain_id=None, start=None, end=None):
#     """
#     提取 CA 原子列表。
#     """
#     atoms = []
    
#     # 兼容处理：获取第一个 Model (不管是 CIF 还是 PDB)
#     # structure 类似字典，key 是 model ID (可能是 0, 1, 或其他)
#     models = list(structure.get_models())
#     if not models:
#         return []
#     first_model = models[0]

#     # 获取链
#     if chain_id and chain_id in first_model:
#         chain = first_model[chain_id]
#     else:
#         # 容错：如果找不到指定链，取第一条
#         chain = list(first_model.get_chains())[0]

#     for res in chain:
#         if not Bio.PDB.is_aa(res, standard=True):
#             continue
        
#         # 如果有范围限制，检查 ID
#         rid = res.id[1]
#         if start is not None and rid < start: continue
#         if end is not None and rid > end: continue
            
#         if 'CA' in res:
#             atoms.append(res['CA'])
            
#     return atoms

# def sliding_window_search(target_atoms_full, query_motif_atoms):
#     """
#     滑动窗口核心算法。
#     """
#     motif_len = len(query_motif_atoms)
#     target_len = len(target_atoms_full)
    
#     if motif_len > target_len:
#         return None, None, 999.0
    
#     best_rmsd = float('inf')
#     best_idx = -1
    
#     sup = Superimposer()
    
#     # 遍历每一个可能的起始点
#     for i in range(target_len - motif_len + 1):
#         # 截取窗口
#         window_atoms = target_atoms_full[i : i + motif_len]
        
#         # 计算 RMSD
#         try:
#             sup.set_atoms(query_motif_atoms, window_atoms)
#             current_rmsd = sup.rms
            
#             if current_rmsd < best_rmsd:
#                 best_rmsd = current_rmsd
#                 best_idx = i
#         except Exception:
#             # 极少数情况可能原子不对齐，跳过
#             continue
            
#     if best_idx != -1:
#         return best_idx, best_idx + motif_len - 1, best_rmsd
#     else:
#         return None, None, 999.0

# # ================= 主程序 =================

# def main():
#     # 1. 准备 Target (生成的蛋白)
#     # 假设你的生成蛋白依然是 PDB 格式
#     pdb_parser = PDBParser(QUIET=True)
    
#     if not os.path.exists(MY_GENERATED_PDB):
#         print(f"错误：找不到文件 {MY_GENERATED_PDB}")
#         return

#     print(f"加载生成蛋白: {MY_GENERATED_PDB}")
#     gen_struct = pdb_parser.get_structure("GEN", MY_GENERATED_PDB)
#     gen_atoms_full = get_ca_atoms_from_chain(gen_struct, MY_CHAIN_ID)
#     print(f"生成蛋白长度 (CA原子数): {len(gen_atoms_full)}")
    
#     # 2. 准备 CIF 解析器 (用于 AlphaFill 下载的文件)
#     cif_parser = MMCIFParser(QUIET=True)

#     # 3. 准备 PyMOL 脚本
#     pml_file = "structural_window_search.pml"
#     f_pml = open(pml_file, "w")
#     f_pml.write(f"load {MY_GENERATED_PDB}, gen_prot\n")
#     f_pml.write("bg_color white\n")
#     f_pml.write("hide everything\n")
#     f_pml.write("show cartoon, gen_prot\n")
#     f_pml.write("color white, gen_prot\n")
#     f_pml.write("set cartoon_transparency, 0.7, gen_prot\n")
    
#     print("-" * 50)
#     print("开始滑动窗口结构比对 (AlphaFill CIF 源)...")
    
#     count = 0
#     for tid, t_start, t_end in TARGETS_DATA:
#         # 下载 CIF
#         cif_path = download_alphafill_cif(tid)
#         if not cif_path: continue
        
#         # 解析真实蛋白 (注意这里用 cif_parser)
#         try:
#             t_struct = cif_parser.get_structure(tid, cif_path)
#         except Exception as e:
#             print(f"[{tid}] CIF 解析失败: {e}")
#             continue
        
#         # 提取 Query Motif
#         # AlphaFill/AlphaFold 通常 Chain ID 是 'A'
#         motif_atoms = get_ca_atoms_from_chain(t_struct, start=t_start, end=t_end)
        
#         if len(motif_atoms) < 10:
#             print(f"[{tid}] 跳过: Motif 太短或提取失败 (原子数: {len(motif_atoms)})")
#             continue
            
#         # === 核心：滑动窗口比对 ===
#         best_start_idx, best_end_idx, min_rmsd = sliding_window_search(gen_atoms_full, motif_atoms)
        
#         if best_start_idx is not None:
#             res_start_obj = gen_atoms_full[best_start_idx].get_parent()
#             res_end_obj   = gen_atoms_full[best_end_idx].get_parent()
            
#             r_start_id = res_start_obj.id[1]
#             r_end_id   = res_end_obj.id[1]
            
#             print(f"[{tid}] 最佳匹配 -> RMSD: {min_rmsd:.3f} | 生成蛋白区域: {r_start_id}-{r_end_id}")
            
#             # === 写入 PyMOL ===
#             obj_name = f"real_{tid}"
#             # PyMOL 可以直接 load .cif 文件
#             f_pml.write(f"load {cif_path}, {obj_name}\n")
#             f_pml.write(f"hide everything, {obj_name}\n")
            
#             motif_sel = f"motif_{tid}"
#             f_pml.write(f"select {motif_sel}, {obj_name} and resi {t_start}-{t_end}\n")
#             f_pml.write(f"color red, {motif_sel}\n")
#             f_pml.write(f"show cartoon, {motif_sel}\n")
            
#             gen_match_sel = f"gen_match_{tid}"
#             f_pml.write(f"select {gen_match_sel}, gen_prot and resi {r_start_id}-{r_end_id}\n")
#             f_pml.write(f"color blue, {gen_match_sel}\n")
#             f_pml.write(f"show cartoon, {gen_match_sel}\n")
            
#             f_pml.write(f"align {motif_sel}, {gen_match_sel}\n")
            
#             count += 1
#         else:
#             print(f"[{tid}] 未找到匹配")

#     # 4. 结束
#     f_pml.write("zoom gen_prot\n")
#     f_pml.write("deselect\n")
#     f_pml.close()
    
#     print("-" * 50)
#     print(f"处理完成，共比对 {count} 个结构。")
#     print(f"可视化脚本已生成: {pml_file}")

# if __name__ == "__main__":
#     main()

import os
import random
import requests
import ast
import Bio.PDB
from Bio.PDB import PDBParser, MMCIFParser, Superimposer
import numpy as np

# ================= 配置区域 =================

# 1. 你的生成蛋白 (被搜索的目标)
# MY_GENERATED_PDB = "SEQUENCE_ID=Q5V3D9_L=327_plddt_82.2352294921875_ptm_0.871.pdb"
MY_GENERATED_PDB = 'SEQUENCE_ID=Q9M1K9_L=297_plddt_94.46477508544922_ptm_0.857.pdb'
SAVE_NAME = MY_GENERATED_PDB.replace('SEQUENCE_ID=', '').split('_L=')[0]

MY_CHAIN_ID = "A" 

# 2. 目标数据文件路径
# 请将你的数据保存到这个文件中，每一行一条记录
TARGET_LIST_FILE = "watch_train_data.out" 

# ================= 数据加载函数 =================

def load_targets_from_file(filepath):
    """
    从文本文件加载目标蛋白数据。
    文件格式示例:
    protein: ('Q1AXT3', [{'go_term': '...', 'start': 32, ...}, ...])
    """
    targets = []
    # target_go_term = 'methylenetetrahydrofolate dehydrogenase (NADP+) activity'
    # target_go_term = 'methenyltetrahydrofolate cyclohydrolase activity'

    target_go_term = 'oxidoreductase activity'  # [A0A2H3D905] RMSD: 0.281 | Hit: 56-73
    target_go_term = 'identical protein binding' # [P19669] RMSD: 2.325 | Hit: 106-123

    target_go_terms = ['oxidoreductase activity', 'identical protein binding']
    
    print(f"正在读取目标列表: {filepath} ...")
    
    if not os.path.exists(filepath):
        print(f"错误: 文件 {filepath} 不存在！请先创建该文件。")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # 去掉开头的 "protein: " 标签以便解析
            if line.startswith("protein: "):
                content = line.replace("protein: ", "", 1)
            else:
                content = line
            
            try:
                # 使用 ast.literal_eval 安全地将字符串转为 Python Tuple
                # data 结构: ('UniprotID', [ {motif_info}, {motif_info} ])
                entry = ast.literal_eval(content)
                
                uniprot_id = entry[0]
                motifs = entry[1]
                
                found = False
                for m in motifs:
                    # 寻找指定的 GO Term
                    if m.get('go_term') in target_go_terms:
                        start = m.get('start')
                        end = m.get('end')
                        
                        # 简单的合法性检查
                        if start and end and end > start:
                            targets.append((uniprot_id, start, end, m.get('go_term')))
                            found = True
                            # 如果只需要找到第一个匹配的 motif 就停止，可以使用 break
                            # break 
                
                # if not found:
                #     print(f"[{uniprot_id}] 未找到指定 GO Term 的 Motif")

            except Exception as e:
                print(f"解析行出错: {line[:30]}... 错误: {e}")
                continue
                
    print(f"成功加载 {len(targets)} 个有效目标片段。")
    return targets

# ================= 工具函数 (AlphaFill/PDB处理) =================

def download_alphafill_cif(uniprot_id, output_dir="pdbs"):
    """从 AlphaFill 下载 CIF 文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = f"{uniprot_id}.cif"
    filepath = os.path.join(output_dir, filename)
    
    if os.path.exists(filepath): return filepath
    
    url = f"https://alphafill.eu/v1/aff/{uniprot_id}"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            with open(filepath, "wb") as f: f.write(r.content)
            return filepath
    except Exception as e:
        print(f"下载异常 {uniprot_id}: {e}")
    return None

def get_ca_atoms_from_chain(structure, chain_id=None, start=None, end=None):
    """提取 CA 原子列表，兼容 CIF/PDB"""
    atoms = []
    models = list(structure.get_models())
    if not models: return []
    first_model = models[0] # 取第一个模型

    # 尝试获取链，如果失败则取第一条链
    if chain_id and chain_id in first_model:
        chain = first_model[chain_id]
    else:
        chain = list(first_model.get_chains())[0]

    for res in chain:
        # 只处理标准氨基酸
        if not Bio.PDB.is_aa(res, standard=True): continue
        
        rid = res.id[1]
        # 筛选残基范围
        if start is not None and rid < start: continue
        if end is not None and rid > end: continue
            
        if 'CA' in res:
            atoms.append(res['CA'])
            
    return atoms

def sliding_window_search(target_atoms_full, query_motif_atoms):
    """滑动窗口 RMSD 计算"""
    motif_len = len(query_motif_atoms)
    target_len = len(target_atoms_full)
    
    if motif_len > target_len: return None, None, 999.0
    
    best_rmsd = float('inf')
    best_idx = -1
    sup = Superimposer()
    
    for i in range(target_len - motif_len + 1):
        window_atoms = target_atoms_full[i : i + motif_len]
        try:
            sup.set_atoms(query_motif_atoms, window_atoms)
            if sup.rms < best_rmsd:
                best_rmsd = sup.rms
                best_idx = i
        except: continue
            
    if best_idx != -1:
        return best_idx, best_idx + motif_len - 1, best_rmsd
    return None, None, 999.0

# ================= 主程序 =================

def main():
    # 1. 加载目标列表
    targets_data = load_targets_from_file(TARGET_LIST_FILE)

    targets_data = [data if data[0] == 'A0A2H3D905' or data[0] == 'P0ABP8' else None for data in targets_data]
    targets_data = [data for data in targets_data if data is not None]
    # print(targets_data)

    # targets_data = random.sample(targets_data, 20) # 随机选取 10 个目标
    if not targets_data:
        return

    # 2. 准备 Target (生成的蛋白 - PDB格式)
    pdb_parser = PDBParser(QUIET=True)
    if not os.path.exists(MY_GENERATED_PDB):
        print(f"错误：找不到文件 {MY_GENERATED_PDB}")
        return

    print(f"加载生成蛋白: {MY_GENERATED_PDB}")
    gen_struct = pdb_parser.get_structure("GEN", MY_GENERATED_PDB)
    gen_atoms_full = get_ca_atoms_from_chain(gen_struct, MY_CHAIN_ID)
    print(f"生成蛋白长度: {len(gen_atoms_full)} residues")
    
    # 3. 准备结果文件
    cif_parser = MMCIFParser(QUIET=True)
    pml_file = f"structural_search_result_{SAVE_NAME}.pml"
    
    with open(pml_file, "w") as f_pml:
        # 初始化 PyMOL 环境
        f_pml.write(f"load {SAVE_NAME}.pdb, gen_prot\n")
        f_pml.write("bg_color white\n")
        f_pml.write("hide everything\n")
        f_pml.write("show cartoon, gen_prot\n")
        f_pml.write("color white, gen_prot\n")
        f_pml.write("set cartoon_transparency, 0.6, gen_prot\n")
        
        print("-" * 50)
        print("开始结构比对...")
        
        count = 0
        for tid, t_start, t_end, go_term in targets_data:
            # 下载与解析
            cif_path = download_alphafill_cif(tid)
            if not cif_path: continue
            
            try:
                t_struct = cif_parser.get_structure(tid, cif_path)
            except:
                print(f"[{tid}] CIF 解析失败")
                continue
            
            # 提取真实 Motif (AlphaFold 预测结构通常链名为 A)
            motif_atoms = get_ca_atoms_from_chain(t_struct, chain_id='A', start=t_start, end=t_end)
            
            if len(motif_atoms) < 5:
                print(f"[{tid}] Motif 提取失败或过短")
                continue
                
            # 滑动窗口比对
            start_idx, end_idx, rmsd = sliding_window_search(gen_atoms_full, motif_atoms)
            
            if start_idx is not None:
                # 获取生成蛋白上匹配区域的真实残基编号
                r_start = gen_atoms_full[start_idx].get_parent().id[1]
                r_end   = gen_atoms_full[end_idx].get_parent().id[1]
                
                print(f"[{tid}] [{go_term}] RMSD: {rmsd:.3f} | Hit: {r_start}-{r_end}")
                
                # 只有当 RMSD 足够好时才写入 PyMOL (例如 < 2.0 或 3.0，这里全写)
                # 生成 PyMOL 命令
                obj_name = f"real_{tid}"
                f_pml.write(f"load {cif_path}, {obj_name}\n")
                f_pml.write(f"hide everything, {obj_name}\n")
                
                # 选中真实 Motif 并标红
                motif_sel = f"motif_{tid}"
                f_pml.write(f"select {motif_sel}, {obj_name} and resi {t_start}-{t_end}\n")
                f_pml.write(f"color red, {motif_sel}\n")
                f_pml.write(f"show cartoon, {motif_sel}\n")
                
                # 选中生成蛋白的匹配区域并标蓝
                gen_sel = f"gen_match_{tid}"
                f_pml.write(f"select {gen_sel}, gen_prot and resi {r_start}-{r_end}\n")
                f_pml.write(f"color blue, {gen_sel}\n")
                f_pml.write(f"show cartoon, {gen_sel}\n")
                
                # 叠合
                f_pml.write(f"align {motif_sel}, {gen_sel}\n")
                
                count += 1
            else:
                print(f"[{tid}] 未找到结构匹配")

        # 收尾
        f_pml.write("zoom gen_prot\n")
        f_pml.write("deselect\n")
        print("-" * 50)
        print(f"完成! 共生成 {count} 个比对。脚本: {pml_file}")

if __name__ == "__main__":
    main()