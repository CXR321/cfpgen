import os
import ast
import requests
import warnings
import copy
from Bio import BiopythonWarning
from Bio.PDB import MMCIFParser, MMCIFIO, PDBParser, Superimposer
from Bio.SeqUtils import seq1

# 忽略 PDB 解析的一般警告
warnings.simplefilter('ignore', BiopythonWarning)

# --- 全局配置 ---
TARGET_ID = "Q4X1A4"  # 目标参考蛋白
TARGET_PDB_PATH = "SEQUENCE_ID=Q4X1A4_L=330_886_plddt_94.8526382446289_ptm_0.955.pdb"
# TARGET_PDB_PATH = "./pdbs/Q6IE46.cif"
# TARGET_PDB_PATH = "./pdbs/A4VUK5.cif"
INPUT_FILE = "watch_train_data.out" 
OUTPUT_DIR = "alignment_output_motif" # 输出目录
PDB_DIR = "pdbs"

# 筛选条件
MAX_MOTIF_LENGTH = 150
ALLOWED_GO_TERMS = {
    # 'oxidoreductase activity', 
    # 'identical protein binding',
    # 'hydrolase activity',
    # 'carbohydrate derivative binding',
    # 'mannitol-1-phosphate 5-dehydrogenase activity',
    'NAD binding',
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. 下载模块 ---
def download_alphafill_cif(uniprot_id, output_dir=PDB_DIR):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    filename = f"{uniprot_id}.cif"
    filepath = os.path.join(output_dir, filename)
    if os.path.exists(filepath): return filepath
    
    url = f"https://alphafill.eu/v1/aff/{uniprot_id}"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            with open(filepath, "wb") as f: f.write(r.content)
            return filepath
    except Exception:
        pass
    return None

# --- 2. 解析与筛选 ---
def parse_watch_file(filepath):
    tasks = []
    if not os.path.exists(filepath): return tasks

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line.startswith("protein:"): continue
            content_str = line.replace("protein: ", "").strip()
            try:
                pdb_id, motif_list = ast.literal_eval(content_str)
                for i, m in enumerate(motif_list):
                    if m.get('go_term') not in ALLOWED_GO_TERMS: continue
                    
                    start, end = m['start'], m['end']
                    if (end - start + 1) >= MAX_MOTIF_LENGTH:
                        continue

                    tasks.append({
                        'id': pdb_id,
                        'motif_seq': m['motif_segment'],
                        'start': start,
                        'end': end,
                        'go_term': m['go_term'],
                        'motif_idx': i
                    })
            except:
                pass
    return tasks

# --- 3. 结构处理工具 ---

def get_first_chain(structure):
    """获取结构中的第一条链"""
    for model in structure:
        for chain in model:
            return chain
    return None

def get_all_ca_atoms(structure):
    """
    获取整条链所有的 CA 原子，返回列表。
    用于在 Target 上进行滑窗遍历。
    """
    atoms = []
    chain = get_first_chain(structure)
    if not chain: return []
    
    # 获取所有残基，过滤掉水分子等 hetero 原子
    for residue in chain:
        if residue.id[0] == ' ': # 标准氨基酸
            if 'CA' in residue:
                atoms.append(residue['CA'])
    return atoms

def get_atoms_by_residue_range(structure, start_res, end_res):
    """
    提取指定残基范围内的 CA 原子 (1-based index, PDB numbering)。
    用于提取 Motif 的原子。
    """
    atoms = []
    chain = get_first_chain(structure)
    if not chain: return []
    
    for residue in chain:
        # residue.id[1] 是 PDB 中的序列号
        if start_res <= residue.id[1] <= end_res:
            if 'CA' in residue:
                atoms.append(residue['CA'])
    return atoms

def scan_best_rmsd_window(target_ca_list, motif_ca_list):
    """
    核心逻辑：在 Target 的 CA 列表中滑动窗口，
    寻找与 Motif CA 列表 RMSD 最小的片段。
    
    Args:
        target_ca_list: Target 的所有 CA 原子列表
        motif_ca_list: Motif 的 CA 原子列表
        
    Returns:
        best_fixed_atoms: 对应的 Target 原子片段 (用于最后做 apply)
        min_rmsd: 最小的 RMSD 值
    """
    motif_len = len(motif_ca_list)
    target_len = len(target_ca_list)
    
    if motif_len == 0 or target_len < motif_len:
        return None, float('inf')

    min_rmsd = float('inf')
    best_fixed_atoms = None
    
    si = Superimposer()
    
    # 开始滑窗
    # i 是窗口在 target_ca_list 中的起始索引
    for i in range(target_len - motif_len + 1):
        # 取出当前的窗口片段
        window_fixed = target_ca_list[i : i + motif_len]
        
        try:
            # 计算 RMSD (注意：set_atoms 不会修改原子坐标，只计算矩阵和 rms)
            si.set_atoms(window_fixed, motif_ca_list)
            
            if si.rms < min_rmsd:
                min_rmsd = si.rms
                best_fixed_atoms = window_fixed
        except Exception:
            continue
            
    return best_fixed_atoms, min_rmsd

# --- 4. 主逻辑 ---

def main():
    print(f"--- Step 1: Loading Tasks ---")
    tasks = parse_watch_file(INPUT_FILE)
    print(f"Found {len(tasks)} valid motifs (Length < {MAX_MOTIF_LENGTH}).")
    if not tasks: return

    print(f"\n--- Step 2: Loading Target {TARGET_ID} ---")
    target_path = TARGET_PDB_PATH
    if not target_path: return

    # 加载 Target 结构
    parser_pdb = PDBParser(QUIET=True)
    target_struct = parser_pdb.get_structure(TARGET_ID, target_path)

    # parser_pdb = MMCIFParser(QUIET=True)
    # target_struct = parser_pdb.get_structure(TARGET_ID, target_path)
    
    # 提取 Target 的所有 CA 原子用于滑窗
    target_all_ca = get_all_ca_atoms(target_struct)
    print(f"Target Loaded. Total CA atoms: {len(target_all_ca)}")

    # PyMOL 初始化
    pymol_script = [
        "reinitialize", 
        "bg_color white",
        f"load {os.path.abspath(target_path)}, target",
        "color gray90, target", 
        "set transparency, 0.6, target"
    ]

    print(f"\n--- Step 3: Aligning Motifs by Structural Scanning ---")
    
    cif_parser = MMCIFParser(QUIET=True)

    for task in tasks:
        uid = task['id']
        in_start, in_end = task['start'], task['end']
        label = task['go_term']
        
        # 下载/加载 Motif 所在蛋白结构
        input_path = download_alphafill_cif(uid)
        if not input_path: continue

        try:
            # 每次加载新对象以防坐标污染
            moving_struct = cif_parser.get_structure(f"{uid}_tmp", input_path)
        except Exception:
            continue

        # A. 获取 Motif 的原子 (Moving Atoms)
        moving_atoms = get_atoms_by_residue_range(moving_struct, in_start, in_end)
        
        if len(moving_atoms) < 3:
            print(f"Skipping {uid}: Motif atoms too few ({len(moving_atoms)}).")
            continue

        # B. 核心：通过滑窗寻找最佳匹配位置 (Scanning)
        # 不需要序列比对，直接用几何形状去套
        best_fixed_atoms, min_rmsd = scan_best_rmsd_window(target_all_ca, moving_atoms)

        if best_fixed_atoms is None:
            print(f"Skipping {uid}: Window scan failed.")
            continue
        
        # 设定一个 RMSD 阈值，如果最小 RMSD 都很大，可能说明这个 Motif 在 Target 上根本没有对应的结构
        # 这里的 5.0 只是一个示例，你可以根据需要调整
        if min_rmsd > 5.0:
            print(f"Skipping {uid}: RMSD too high ({min_rmsd:.3f}) - No structural match found.")
            continue

        # C. 执行最终对齐
        # 再次初始化 Superimposer，用最佳窗口的原子进行 Apply
        super_imposer = Superimposer()
        try:
            super_imposer.set_atoms(best_fixed_atoms, moving_atoms)
            super_imposer.apply(moving_struct.get_atoms()) # 将旋转应用到整个 moving 结构
            
            # 记录匹配到的 Target 位置信息 (用于日志)
            match_start_res = best_fixed_atoms[0].get_parent().id[1]
            match_end_res = best_fixed_atoms[-1].get_parent().id[1]
            
            print(f"Aligned {uid} | RMSD: {min_rmsd:.3f} | Match Target Res: {match_start_res}-{match_end_res} | Label: {label}")
        except Exception as e:
            print(f"Alignment error {uid}: {e}")
            continue

        # D. 保存 (使用 MMCIFIO)
        out_filename = f"{uid}_m{task['motif_idx']}_{label.split()[0]}.cif"
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        
        io = MMCIFIO()
        io.set_structure(moving_struct)
        try:
            io.save(out_path)
        except:
            continue

        # E. PyMOL 命令 (逻辑保持不变)
        obj_name = f"{uid}_m{task['motif_idx']}"
        pymol_script.append(f"load {os.path.abspath(out_path)}, {obj_name}")
        
        if 'oxidoreductase' in label:
            stick_color = "cyan"
        else:
            stick_color = "magenta"
            
        pymol_script.append(f"color gray80, {obj_name}")
        pymol_script.append(f"hide cartoon, {obj_name}") 
        
        sel_name = f"sel_{obj_name}"
        pymol_script.append(f"select {sel_name}, {obj_name} and resi {in_start}-{in_end}")
        
        pymol_script.append(f"show sticks, {sel_name}")
        pymol_script.append(f"color {stick_color}, {sel_name}")
        pymol_script.append(f"util.cnc {sel_name}") 
        pymol_script.append(f"group {label.replace(' ', '_')}, {obj_name}")

    # 保存脚本
    pml_file = "view_motif_align.pml"
    with open(pml_file, "w") as f:
        f.write("\n".join(pymol_script))
        f.write("\nzoom target\n")
    
    print(f"\nDone. Run 'pymol {pml_file}'")

if __name__ == "__main__":
    main()