import os
import numpy as np
import warnings
import copy
from Bio import BiopythonWarning
from Bio.PDB import MMCIFParser, PDBParser, Superimposer, MMCIFIO
from Bio.SeqUtils import seq1

# 忽略警告
warnings.simplefilter('ignore', BiopythonWarning)

# --- 配置 ---
FILE_A = "A6USK4_11_425_3ojl.1.B.cif"  # Target
FILE_B = "SEQUENCE_ID=Q4X1A4_L=330_886_plddt_94.8526382446289_ptm_0.955.pdb"  # Mobile
OUTPUT_FILE = "structure_aligned.cif"

# 扫描窗口大小 (类似指纹匹配，越长越准但越慢，建议 15-20)
SEED_WINDOW_SIZE = 30
# 最终收录的距离阈值
FINAL_DISTANCE_CUTOFF = 1

def get_structure(filepath, struct_id="struct"):
    if filepath.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure(struct_id, filepath)

def get_calpha_atoms(structure):
    """只提取 CA 原子"""
    atoms = []
    model = next(iter(structure))
    chain = next(iter(model))
    for residue in chain:
        if residue.id[0] == ' ' and 'CA' in residue:
            atoms.append(residue['CA'])
    return atoms

def scan_best_seed(atoms_fixed, atoms_moving, window_size=20):
    """
    核心步骤：滑动窗口扫描。
    在 Fixed 结构上滑，也在 Moving 结构上滑，寻找局部形状最像的片段。
    """
    len_f = len(atoms_fixed)
    len_m = len(atoms_moving)
    
    print(f"Scanning for structural seed (Window={window_size})...")
    print(f"Target size: {len_f}, Mobile size: {len_m}")
    
    best_rmsd = float('inf')
    best_seed_pair = None # (fixed_sublist, moving_sublist)
    best_indices = None   # (start_f, start_m)

    si = Superimposer()
    
    # 为了速度，Target步长设为 5，Mobile步长设为 1
    # 这是一种启发式搜索，能大幅提升速度
    step_f = 5 
    step_m = 1
    
    count = 0
    total_steps = ((len_f - window_size) // step_f) * ((len_m - window_size) // step_m)
    
    for i in range(0, len_f - window_size, step_f):
        # 打印进度
        if i % 50 == 0:
            print(f"  Scanning Target residue {i}/{len_f}...")
            
        coords_f = atoms_fixed[i : i + window_size]
        
        for j in range(0, len_m - window_size, step_m):
            coords_m = atoms_moving[j : j + window_size]
            
            # 快速计算 RMSD
            try:
                si.set_atoms(coords_f, coords_m)
                if si.rms < best_rmsd:
                    best_rmsd = si.rms
                    best_seed_pair = (coords_f, coords_m)
                    best_indices = (i, j)
            except Exception:
                continue
                
    print(f"\nBest Seed Found: Target[{best_indices[0]}] vs Mobile[{best_indices[1]}]")
    print(f"Seed RMSD: {best_rmsd:.3f}")
    return best_seed_pair, best_indices

def structure_extension(atoms_fixed, atoms_moving, seed_pair):
    """
    基于种子初始叠加，然后贪婪地寻找所有空间邻近点。
    """
    # 1. 初始叠加
    seed_fixed, seed_moving = seed_pair
    si = Superimposer()
    si.set_atoms(seed_fixed, seed_moving)
    
    # 2. 将旋转应用到所有 Mobile 原子（临时副本）
    temp_moving = copy.deepcopy(atoms_moving)
    si.apply(temp_moving)
    
    # 3. 寻找最近邻 (Nearest Neighbor Search)
    # 对于每个 Target 原子，找到最近的 Mobile 原子
    # 如果距离 < 阈值，则加入对齐列表
    
    print("Extending alignment based on spatial proximity...")
    
    # 为了简单，使用暴力距离矩阵（几百个原子很快）
    coords_f = np.array([a.get_coord() for a in atoms_fixed])
    coords_m = np.array([a.get_coord() for a in temp_moving])
    
    # 计算距离矩阵 (N_fixed x N_moving)
    # 利用广播机制: (N, 1, 3) - (1, M, 3)
    diff = coords_f[:, np.newaxis, :] - coords_m[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # 贪婪匹配：
    # 这是一个简化版的 Dynamic Time Warping (DTW) 或 简单的最近邻
    # 这里我们使用简单的规则：如果 dist[i, j] < cutoff 且是局部最小，就选它
    
    aligned_pairs = [] # (index_f, index_m)
    used_m = set()
    
    # 遍历 Target
    for i in range(len(atoms_fixed)):
        # 找到与 Target[i] 最近的 Mobile 原子
        best_j = -1
        min_d = float('inf')
        
        # 仅搜索当前对齐线附近的 j，防止错位太远（可选优化）
        # 这里全搜
        for j in range(len(atoms_moving)):
            if j in used_m: continue # 保证一对一
            
            d = dists[i, j]
            if d < min_d:
                min_d = d
                best_j = j
        
        if min_d < FINAL_DISTANCE_CUTOFF:
            aligned_pairs.append((i, best_j))
            used_m.add(best_j)
            
    # 4. 基于扩展后的点，重新计算最终的 RMSD
    final_fixed = [atoms_fixed[i] for i, j in aligned_pairs]
    final_moving = [atoms_moving[j] for i, j in aligned_pairs]
    
    si_final = Superimposer()
    si_final.set_atoms(final_fixed, final_moving)
    
    return aligned_pairs, si_final

def print_alignment_details(pairs, atoms_a, atoms_b, si):
    print("\n" + "="*80)
    print(f"{'Idx':<5} | {'Target (A)':<25} | {'Mobile (B)':<25} | {'Dist':<6}")
    print("-" * 80)
    
    # 临时应用变换用于打印
    temp_atoms_b = [copy.deepcopy(atoms_b[j]) for i, j in pairs]
    si.apply(temp_atoms_b)
    
    count = 0
    for idx, (i, j) in enumerate(pairs):
        atom_a = atoms_a[i]
        atom_b_orig = atoms_b[j]
        atom_b_trans = temp_atoms_b[idx]
        
        res_a = atom_a.get_parent()
        res_b = atom_b_orig.get_parent()
        res_str_a = f"{res_a.get_resname()} {res_a.id[1]}"
        res_str_b = f"{res_b.get_resname()} {res_b.id[1]}"
        
        dist = np.linalg.norm(atom_a.get_coord() - atom_b_trans.get_coord())
        
        if len(pairs) < 50 or count < 10 or count > len(pairs) - 10:
             print(f"{count:<5} | {res_str_a:<25} | {res_str_b:<25} | {dist:.3f}")
        elif count == 10:
             print("... (hidden) ...")
        count += 1
    print("="*80 + "\n")

def main():
    print(f"Loading Structures...")
    s_a = get_structure(FILE_A, "Fixed")
    atoms_a = get_calpha_atoms(s_a)
    
    s_b = get_structure(FILE_B, "Moving")
    atoms_b = get_calpha_atoms(s_b)

    # 1. 扫描寻找最佳种子 (Structure Scanning)
    seed_pair, _ = scan_best_seed(atoms_a, atoms_b, window_size=SEED_WINDOW_SIZE)
    
    if not seed_pair:
        print("Failed to find any matching seed.")
        return

    # 2. 空间扩展 (Extension)
    final_pairs, si = structure_extension(atoms_a, atoms_b, seed_pair)
    
    print("-" * 30)
    print(f"Structure Alignment Result:")
    print(f"Aligned Length: {len(final_pairs)} residues (based on distance < {FINAL_DISTANCE_CUTOFF}A)")
    print(f"RMSD: {si.rms:.3f} Angstrom")
    print("-" * 30)

    if len(final_pairs) == 0:
        print("No structural alignment found.")
        return

    print_alignment_details(final_pairs, atoms_a, atoms_b, si)

    # 3. 保存
    all_atoms_b = list(s_b.get_atoms())
    si.apply(all_atoms_b)
    
    io = MMCIFIO()
    io.set_structure(s_b)
    io.save(OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")
    
    # 4. PML
    with open("view_struct_align.pml", "w") as f:
        f.write(f"load {os.path.abspath(FILE_A)}, target\n")
        f.write(f"load {os.path.abspath(OUTPUT_FILE)}, mobile\n")
        f.write("color gray80, target\n")
        f.write("color cyan, mobile\n")
        # 突出显示对齐区域
        f.write("select aln_target, target and (")
        f.write(" or ".join([f"resi {atoms_a[i].get_parent().id[1]}" for i, j in final_pairs]))
        f.write(")\n")
        f.write("color magenta, aln_target\n")
        f.write("show sticks, aln_target\n")
        f.write("zoom aln_target\n")

if __name__ == "__main__":
    main()