import os
import ast
import requests
import warnings
from Bio import BiopythonWarning
from Bio.PDB import MMCIFParser, MMCIFIO, PDBParser, Superimposer
from Bio.Align import PairwiseAligner
from Bio.SeqUtils import seq1

# 忽略 PDB 解析的一般警告
warnings.simplefilter('ignore', BiopythonWarning)

# --- 全局配置 ---
TARGET_ID = "Q9M1K9"  # 目标参考蛋白
INPUT_FILE = "watch_train_data.out" 
OUTPUT_DIR = "alignment_output_motif" # 输出目录
PDB_DIR = "pdbs"

# 筛选条件
MAX_MOTIF_LENGTH = 100
ALLOWED_GO_TERMS = {
    'oxidoreductase activity', 
    'identical protein binding'
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
                    # 筛选 GO Term
                    if m.get('go_term') not in ALLOWED_GO_TERMS: continue
                    
                    # 筛选 长度 < 100
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

def get_structure_sequence(structure):
    """提取第一条链的序列"""
    chain = get_first_chain(structure)
    if not chain: return ""
    # 过滤掉非标准残基，避免 SeqUtils 报错
    seq = "".join([seq1(r.get_resname()) for r in chain if r.id[0] == ' '])
    return seq

def get_atoms_by_residue_range(structure, start_res, end_res):
    """提取指定残基范围内的 CA 原子 (1-based index)"""
    atoms = []
    chain = get_first_chain(structure)
    if not chain: return []
    
    for residue in chain:
        # residue.id[1] 是 PDB 中的序列号
        if start_res <= residue.id[1] <= end_res:
            if 'CA' in residue:
                atoms.append(residue['CA'])
    return atoms

def map_motif_to_target(target_seq, motif_seq):
    """
    使用局部比对找到 Motif 在 Target 上的坐标范围。
    注意：这只是为了找到 'Anchor'（锚点），对齐本身是基于结构的。
    """
    aligner = PairwiseAligner()
    aligner.mode = 'local'
    aligner.open_gap_score = -10 # 严厉的 gap 惩罚，保证 motif 尽量完整匹配
    aligner.extend_gap_score = -1
    
    alignments = aligner.align(target_seq, motif_seq)
    if not alignments: return None, None
    
    best_aln = alignments[0]
    target_indices = best_aln.aligned[0] # Target 上的匹配片段 [(start, end)]
    
    # 转换为 1-based 坐标
    start_res = target_indices[0][0] + 1
    end_res = target_indices[-1][1]
    
    return start_res, end_res

# --- 4. 主逻辑 ---

def main():
    print(f"--- Step 1: Loading Tasks ---")
    tasks = parse_watch_file(INPUT_FILE)
    print(f"Found {len(tasks)} valid motifs (Length < {MAX_MOTIF_LENGTH}).")
    if not tasks: return

    print(f"\n--- Step 2: Loading Target {TARGET_ID} ---")
    # target_path = download_alphafill_cif(TARGET_ID)
    target_path = 'SEQUENCE_ID=Q9M1K9_L=297_plddt_94.46477508544922_ptm_0.857.pdb'
    if not target_path: return

    cif_parser = MMCIFParser(QUIET=True)
    # target_struct = cif_parser.get_structure(TARGET_ID, target_path)
    parser_pdb = PDBParser(QUIET=True)
    target_struct = parser_pdb.get_structure(TARGET_ID, target_path)
    target_seq = get_structure_sequence(target_struct)
    
    # PyMOL 初始化
    pymol_script = [
        "reinitialize", 
        "bg_color white",
        f"load {os.path.abspath(target_path)}, target",
        "color gray90, target", # Target 设为很浅的灰色
        "set transparency, 0.6, target" # 设置透明度，突出 Motif
    ]

    print(f"\n--- Step 3: Aligning Motifs ---")
    
    for task in tasks:
        uid = task['id']
        motif_seq = task['motif_seq']
        in_start, in_end = task['start'], task['end']
        label = task['go_term']
        
        # 下载
        input_path = download_alphafill_cif(uid)
        if not input_path: continue

        # 每次加载独立的结构对象，因为 Superimposer 会修改坐标
        try:
            moving_struct = cif_parser.get_structure(f"{uid}_tmp", input_path)
        except Exception:
            continue

        # A. 定位：找到 Target 上的对应区域
        t_start, t_end = map_motif_to_target(target_seq, motif_seq)
        if t_start is None:
            continue

        # B. 获取原子：仅获取 Motif 区域的原子
        # Target 的原子 (Fixed)
        fixed_atoms = get_atoms_by_residue_range(target_struct, t_start, t_end)
        # Input 的原子 (Moving) - 直接使用文件里给的 start/end
        moving_atoms = get_atoms_by_residue_range(moving_struct, in_start, in_end)

        # 原子数量检查与截断
        min_len = min(len(fixed_atoms), len(moving_atoms))
        if min_len < 3:
            print(f"Skipping {uid}: Not enough matching atoms.")
            continue
        
        fixed_atoms = fixed_atoms[:min_len]
        moving_atoms = moving_atoms[:min_len]

        # C. 结构对齐 (SVD) - 仅计算 Motif 原子的最佳旋转矩阵
        super_imposer = Superimposer()
        try:
            super_imposer.set_atoms(fixed_atoms, moving_atoms)
            # 应用旋转到整个结构 (虽然我们只关心 Motif，但保存全结构便于观察上下文)
            super_imposer.apply(moving_struct.get_atoms())
            print(f"Aligned {uid} Motif | RMSD: {super_imposer.rms:.3f} | Len: {min_len}")
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

        # E. PyMOL 命令
        obj_name = f"{uid}_m{task['motif_idx']}"
        pymol_script.append(f"load {os.path.abspath(out_path)}, {obj_name}")
        
        # 这里的颜色策略：只显示 Motif 的 Sticks，隐藏其他部分或者把其他部分变淡
        if 'oxidoreductase' in label:
            stick_color = "cyan"
        else:
            stick_color = "magenta"
            
        # 先把整个蛋白隐藏或设为极淡
        pymol_script.append(f"color gray80, {obj_name}")
        pymol_script.append(f"hide cartoon, {obj_name}") # 默认隐藏 Cartoon
        
        # 选中 Motif 区域
        sel_name = f"sel_{obj_name}"
        pymol_script.append(f"select {sel_name}, {obj_name} and resi {in_start}-{in_end}")
        
        # 以漂亮的 Sticks 展示 Motif
        pymol_script.append(f"show sticks, {sel_name}")
        pymol_script.append(f"color {stick_color}, {sel_name}")
        pymol_script.append(f"util.cnc {sel_name}") # 让碳原子上色，氮氧保持蓝红
        
        # 可选：显示 Ribbon 只要 Motif 部分
        pymol_script.append(f"show cartoon, {sel_name}")

        pymol_script.append(f"group {label.replace(' ', '_')}, {obj_name}")

    # 保存脚本
    pml_file = "view_motif_align.pml"
    with open(pml_file, "w") as f:
        f.write("\n".join(pymol_script))
        f.write("\nzoom target\n")
    
    print(f"\nDone. Run 'pymol {pml_file}'")

if __name__ == "__main__":
    main()