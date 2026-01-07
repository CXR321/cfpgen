import os
import ast
import requests
import warnings
from Bio import BiopythonWarning
from Bio.PDB import MMCIFIO, MMCIFParser, PDBParser, Superimposer, PDBIO
from Bio.Align import PairwiseAligner
from Bio.SeqUtils import seq1

# 忽略 PDB 解析的一般警告
warnings.simplefilter('ignore', BiopythonWarning)

# --- 全局配置 ---
TARGET_ID = "Q9M1K9"  # 目标参考蛋白
INPUT_FILE = "watch_train_data.out" # 数据源文件
OUTPUT_DIR = "alignment_output"
PDB_DIR = "pdbs"

# 允许的 GO Term 白名单
ALLOWED_GO_TERMS = {
    'oxidoreductase activity', 
    'identical protein binding'
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. 下载模块 (用户指定) ---

def download_alphafill_cif(uniprot_id, output_dir=PDB_DIR):
    """从 AlphaFill 下载 CIF 文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = f"{uniprot_id}.cif"
    filepath = os.path.join(output_dir, filename)
    
    if os.path.exists(filepath): 
        return filepath
    
    print(f"Downloading {uniprot_id} from AlphaFill...")
    url = f"https://alphafill.eu/v1/aff/{uniprot_id}"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            with open(filepath, "wb") as f: 
                f.write(r.content)
            return filepath
        else:
            print(f"Failed to download {uniprot_id}, Status Code: {r.status_code}")
    except Exception as e:
        print(f"下载异常 {uniprot_id}: {e}")
    return None

# --- 2. 文件解析模块 ---

def parse_watch_file(filepath):
    """解析 watch_train_data.out 文件内容"""
    tasks = []
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return tasks

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line.startswith("protein:"):
                continue
            
            # 去掉 "protein: " 前缀，解析 Tuple
            content_str = line.replace("protein: ", "").strip()
            try:
                # 使用 ast.literal_eval 安全地将字符串转换为 Python 对象
                pdb_id, motif_list = ast.literal_eval(content_str)
                
                # 筛选符合条件的 Motif
                for i, m in enumerate(motif_list):
                    if m.get('go_term') in ALLOWED_GO_TERMS:
                        tasks.append({
                            'id': pdb_id,
                            'motif_seq': m['motif_segment'],
                            'start': m['start'],
                            'end': m['end'],
                            'go_term': m['go_term'],
                            'motif_idx': i # 用于区分同一个蛋白的不同片段
                        })
            except Exception as e:
                print(f"Error parsing line: {line[:50]}... -> {e}")
    
    return tasks

# --- 3. 结构处理辅助函数 ---

def get_structure_sequence(structure, chain_id='A'):
    """提取结构序列字符串（仅用于定位）"""
    # 假设 AlphaFold/AlphaFill 主要是 A 链
    try:
        chain = structure[0][chain_id]
        # 过滤掉非标准氨基酸或水分子
        seq = "".join([seq1(r.get_resname()) for r in chain if r.id[0] == ' '])
        return seq
    except KeyError:
        return ""

def get_atoms_for_alignment(structure, chain_id, start_res_id, end_res_id):
    """提取指定残基范围的 CA 原子"""
    atoms = []
    try:
        chain = structure[0][chain_id]
        for residue in chain:
            # residue.id[1] 是 PDB 序列号
            if start_res_id <= residue.id[1] <= end_res_id:
                if 'CA' in residue:
                    atoms.append(residue['CA'])
    except KeyError:
        pass
    return atoms

def find_motif_in_target(target_seq, motif_seq):
    """在 Target 序列中找到 Motif 的最佳比对位置"""
    aligner = PairwiseAligner()
    aligner.mode = 'local'
    # 提高 gap penalty 尽量让 motif 连续匹配
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -1
    
    alignments = aligner.align(target_seq, motif_seq)
    if not alignments:
        return None, None
        
    best_aln = alignments[0]
    # aligned 返回的是 tuple list，例如 [(start, end)]
    # 这里我们只取匹配上的最外层范围
    target_indices = best_aln.aligned[0]
    
    # 转换为 1-based PDB 坐标 (假设 target 是连续的 AlphaFold 结构)
    # 注意：best_aln.aligned 是 0-based，且 end 是不包含的
    start_res = target_indices[0][0] + 1
    end_res = target_indices[-1][1] 
    
    return start_res, end_res

# --- 新增：清理结构以兼容 PDB 格式 ---
def sanitize_structure(structure):
    """
    强制将 Chain ID 改为单字符，解决 PDB 保存时的 %c 错误。
    CIF 可能有长链名，但 PDB 不支持。
    """
    for model in structure:
        for chain in model:
            if len(chain.id) > 1:
                # 如果链名超过1个字符，强制改为 'A' (假设单链)
                # 或者截取第一个字符
                new_id = 'A' 
                print(f"  [Fix] Renaming chain '{chain.id}' to '{new_id}' for PDB compatibility.")
                chain.id = new_id

# --- 4. 主逻辑 ---

def main():
    print(f"--- Step 1: Loading Tasks from {INPUT_FILE} ---")
    tasks = parse_watch_file(INPUT_FILE)
    print(f"Found {len(tasks)} valid motifs to align.")

    if not tasks:
        return

    print(f"\n--- Step 2: Preparing Target {TARGET_ID} ---")
    # target_path = download_alphafill_cif(TARGET_ID)
    target_path = 'SEQUENCE_ID=Q9M1K9_L=297_plddt_94.46477508544922_ptm_0.857.pdb'
    if not target_path:
        print("Target download failed.")
        return

    cif_parser = MMCIFParser(QUIET=True)
    # target_struct = cif_parser.get_structure(TARGET_ID, target_path)
    # target_seq = get_structure_sequence(target_struct)

    parser_pdb = PDBParser(QUIET=True)
    target_struct = parser_pdb.get_structure(TARGET_ID, target_path)
    target_seq = get_structure_sequence(target_struct)

    # PyMOL 脚本初始化
    pymol_script = [
        "reinitialize", 
        "bg_color white",
        f"load {os.path.abspath(target_path)}, target",
        "color gray80, target",
        "set transparency, 0.5, target" # 让 Target 稍微透明一点
    ]

    print(f"\n--- Step 3: Aligning Structures ---")
    
    # 用于防止重复加载同一个文件
    loaded_structures = {} 

    for task in tasks:
        uid = task['id']
        motif_seq = task['motif_seq']
        # 注意：这里的数据是 1-based start/end，可以直接用于提取 Input 的原子
        # 但 AlphaFill 的 CIF 有时可能序号会有偏移，通常 AF 预测结构是从 1 开始的
        in_start, in_end = task['start'], task['end']
        label = task['go_term']
        
        print(f"Processing {uid} | Motif len: {len(motif_seq)} | {label}")
        
        # 下载输入蛋白
        input_path = download_alphafill_cif(uid)
        if not input_path: continue

        # 解析输入蛋白 (为了性能，缓存一下 parser 结果)
        if uid not in loaded_structures:
            loaded_structures[uid] = cif_parser.get_structure(uid, input_path)
        moving_struct = loaded_structures[uid]

        # A. 在 Target 上找位置
        t_start, t_end = find_motif_in_target(target_seq, motif_seq)
        if t_start is None:
            print(f"  -> Could not map motif to target. Skipping.")
            continue

        # B. 获取原子 (Input 的 Motif vs Target 的 对应区域)
        # 注意：每次对齐都是一个新的变换，所以我们需要深拷贝 moving_struct 或者重新加载
        # 但 BioPython 的 Superimposer 是修改原子坐标的。
        # 为了避免同一个蛋白被多次旋转打乱，我们这里必须克隆一个 structure 或者重新解析
        # 简单起见，我们重新解析一次，或者只保存变换后的结果
        temp_struct = cif_parser.get_structure(f"{uid}_temp", input_path)
        
        fixed_atoms = get_atoms_for_alignment(target_struct, 'A', t_start, t_end)
        moving_atoms = get_atoms_for_alignment(temp_struct, 'A', in_start, in_end)

        # 检查原子数量
        min_len = min(len(fixed_atoms), len(moving_atoms))
        if min_len < 3:
            print(f"  -> Not enough atoms ({min_len}) to align.")
            continue
            
        fixed_atoms = fixed_atoms[:min_len]
        moving_atoms = moving_atoms[:min_len]

        # C. 计算 RMSD 并旋转
        super_imposer = Superimposer()
        try:
            super_imposer.set_atoms(fixed_atoms, moving_atoms)
            super_imposer.apply(temp_struct.get_atoms())
            print(f"  -> Aligned RMSD: {super_imposer.rms:.4f}")
        except Exception as e:
            print(f"  -> Alignment calculation failed: {e}")
            continue

        # D. 保存结果
        # 使用 motif_idx 防止同一个蛋白有多个 motif 时文件覆盖
        out_filename = f"{uid}_m{task['motif_idx']}_{label.split()[0]}.cif"
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        
        # sanitize_structure(temp_struct)

        io = MMCIFIO()
        io.set_structure(temp_struct)
        io.save(out_path)

        # E. 写入 PyMOL 命令
        obj_name = f"{uid}_m{task['motif_idx']}"
        pymol_script.append(f"load {os.path.abspath(out_path)}, {obj_name}")
        
        # 颜色策略：Oxidoreductase 用一种色系，Binding 用另一种
        if 'oxidoreductase' in label:
            color = "cyan"
            stick_color = "blue"
        else:
            color = "salmon"
            stick_color = "red"
            
        pymol_script.append(f"color {color}, {obj_name}")
        # 高亮 Motif
        pymol_script.append(f"select sel_{obj_name}, {obj_name} and resi {in_start}-{in_end}")
        pymol_script.append(f"show sticks, sel_{obj_name}")
        pymol_script.append(f"color {stick_color}, sel_{obj_name}")
        # 添加 Label
        pymol_script.append(f"label {obj_name} and name CA and resi {in_start}, '{uid}-{label}'")
        pymol_script.append(f"group {label.replace(' ', '_')}, {obj_name}")

    # 4. 保存 PyMOL 脚本
    pml_file = "view_results.pml"
    with open(pml_file, "w") as f:
        f.write("\n".join(pymol_script))
        f.write("\nzoom target\n")
        f.write("set seq_view, 1\n") # 打开序列视图方便看
    
    print(f"\nDone! Results saved in '{OUTPUT_DIR}'.\nRun 'pymol {pml_file}' to view.")

if __name__ == "__main__":
    main()