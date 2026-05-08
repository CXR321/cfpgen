import os
import pickle
import numpy as np
import requests
import warnings
from tqdm import tqdm
from Bio import PDB
from Bio.PDB import PDBParser, MMCIFParser, Superimposer

# 忽略 Biopython 读取 PDB/CIF 的一些警告
warnings.filterwarnings('ignore')

# ================= 配置区域 =================
# 文件夹配置
PDB_REF_DIR = 'pdbs'                    # 参考蛋白结构下载目录 (CIF)
GEN_PDB_FOLDER = 'generation-results-dplm2-goonly-new-unseen/esmfold_pdb/' # 生成的 PDB 文件夹
OUTPUT_LOG = 'structure_align_results.txt'
OUTPUT_DOUBLE_HIT = 'double_hit_structures.txt'

# 数据路径
TRAIN_DATASET_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
UNSEEM_DATASET_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/generated_candidates_motif_emb.pkl'

# RMSD 阈值
RMSD_THRESHOLD = 2

# ================= 1. [已替换] 下载函数 =================
def download_alphafill_cif(uniprot_id, output_dir=PDB_REF_DIR):
    """
    用户提供的下载函数：下载 AlphaFill 的 CIF 文件
    """
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

# ================= 2. [升级版] 提取 CA 原子坐标 =================
def get_ca_coords(file_path, start_residue=None, end_residue=None):
    """
    提取结构文件中的 CA 原子坐标。
    自动识别 .pdb 和 .cif 格式。
    """
    if not file_path or not os.path.exists(file_path):
        return None

    # 根据后缀选择解析器
    if file_path.endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    try:
        # structure_id 随便起一个
        structure = parser.get_structure('struct', file_path)
    except Exception as e:
        # print(f"Parse error {file_path}: {e}")
        return None

    ca_atoms = []
    try:
        # 获取第一个模型
        model = next(iter(structure)) 
        # 获取第一条链
        chain = next(iter(model))
        
        for residue in chain:
            # 过滤非标准残基 (HETATM)，Biopython 中 id[0] 为 ' ' 表示标准残基
            if residue.id[0] != ' ': continue
            
            # 获取残基编号
            res_id = residue.id[1]
            
            # 范围过滤 (start/end 是 inclusive 的)
            if start_residue and res_id < start_residue: continue
            if end_residue and res_id > end_residue: continue
            
            if 'CA' in residue:
                ca_atoms.append(residue['CA'])
                
    except Exception as e:
        return None
    
    return ca_atoms

# ================= 3. 滑动窗口 RMSD 计算 =================
def calc_sliding_window_rmsd(ref_atoms, target_atoms):
    """
    将 ref_atoms (Motif) 在 target_atoms (生成的全长蛋白) 上滑窗比对。
    """
    len_ref = len(ref_atoms)
    len_target = len(target_atoms)
    
    # 基础检查
    if len_ref == 0 or len_target == 0: return 999.9
    if len_ref > len_target: return 999.9 

    min_rmsd = 999.9
    sup = Superimposer()

    # 滑动窗口
    # 我们只关心 CA 原子的几何匹配，不关心序列是否一致
    for i in range(len_target - len_ref + 1):
        target_window = target_atoms[i : i + len_ref]
        
        try:
            sup.set_atoms(ref_atoms, target_window)
            # sup.apply(target_window) # 不需要实际移动，只要 rms 值
            current_rmsd = sup.rms
            
            if current_rmsd < min_rmsd:
                min_rmsd = current_rmsd
            
            # 极速退出：如果已经非常完美，没必要继续滑了
            if min_rmsd < 0.1: break
            
        except Exception:
            continue

    return min_rmsd

# ================= 主程序 =================
def main():
    print("Loading datasets...")
    # 加载 Unseen Dataset (用于查找生成的 ID 对应的目标 GO)
    with open(UNSEEM_DATASET_PATH, 'rb') as f:
        unseen_data = pickle.load(f)
    # 构建 Unseen 索引: UniprotID -> Entry
    unseen_map = {item['uniprot_id']: item for item in unseen_data}

    # 加载 Train Dataset (用于查找 GO 对应的真实 Motif 结构)
    with open(TRAIN_DATASET_PATH, 'rb') as f:
        train_data = pickle.load(f)

    # 1. 构建训练集索引
    print("Indexing training motifs...")
    go_idx_to_ref_motifs = {} 

    for entry in train_data:
        uid = entry['uniprot_id']
        if 'motif' not in entry or not entry['motif']: continue
        if 'motif_go_number' not in entry: continue
        
        motifs = entry['motif']
        motif_go_idxs = entry.get('motif_go_number', [])
        
        min_len = min(len(motifs), len(motif_go_idxs))
        
        for i in range(min_len):
            go_idx = motif_go_idxs[i]
            motif_info = motifs[i]
            
            if go_idx not in go_idx_to_ref_motifs:
                go_idx_to_ref_motifs[go_idx] = []
            
            go_idx_to_ref_motifs[go_idx].append({
                'uid': uid,
                's': motif_info['start'],
                'e': motif_info['end']
            })

    print(f"Indexed motifs for {len(go_idx_to_ref_motifs)} GO terms.")
    # print(go_idx_to_ref_motifs)

    # 2. 遍历生成的 PDB 文件
    if not os.path.exists(GEN_PDB_FOLDER):
        print(f"Folder {GEN_PDB_FOLDER} not found.")
        return

    pdb_files = [f for f in os.listdir(GEN_PDB_FOLDER) if f.endswith('.pdb')]
    print(f"Found {len(pdb_files)} generated PDB files. Starting alignment...")

    results = []
    double_hits = []

    for pdb_file in tqdm(pdb_files):
        # 解析文件名: SEQUENCE_ID=GEN_000000_0_L=363...
        try:
            gen_id = pdb_file.split('SEQUENCE_ID=')[1].split('_L=')[0]
        except:
            continue
            
        if gen_id not in unseen_map:
            continue
            
        target_entry = unseen_map[gen_id]
        target_go_idxs = target_entry.get('go_f_mapped', [])
        
        # 获取生成蛋白的结构 (PDB格式)
        gen_pdb_path = os.path.join(GEN_PDB_FOLDER, pdb_file)
        gen_ca_atoms = get_ca_coords(gen_pdb_path)
        
        if not gen_ca_atoms: continue

        hits = [] # 记录当前蛋白成功的 GO Term

        # print(gen_id)
        # print(target_go_idxs)
        # exit()
        
        # 3. 对每个目标 GO，寻找参考 Motif 并比对
        for go_idx in target_go_idxs:
            if go_idx not in go_idx_to_ref_motifs:
                continue
            
            candidates = go_idx_to_ref_motifs[go_idx]
            
            best_rmsd = 999.9
            
            # 尝试前 3 个参考蛋白
            for ref_info in candidates[:3]:
                ref_uid = ref_info['uid']
                start = ref_info['s']
                end = ref_info['e']

                if end - start < 30: 
                    continue
                
                # 下载参考结构 (CIF格式)
                ref_cif_path = download_alphafill_cif(ref_uid)
                if not ref_cif_path: continue
                
                # 提取参考 Motif 坐标 (自动识别CIF)
                ref_ca_atoms = get_ca_coords(ref_cif_path, start, end)
                if not ref_ca_atoms or len(ref_ca_atoms) < 5: continue 
                
                # 计算滑窗 RMSD
                rmsd = calc_sliding_window_rmsd(ref_ca_atoms, gen_ca_atoms)
                
                if rmsd < best_rmsd:
                    best_rmsd = rmsd
                
                if best_rmsd < RMSD_THRESHOLD:
                    break
            
            # 判定
            if best_rmsd < RMSD_THRESHOLD:
                print(f"{gen_id} hit {go_idx} with RMSD={best_rmsd:.2f}A")
                print(f"candidate: {ref_uid} ({start}-{end})")
                hits.append((go_idx, best_rmsd))
                
        # 4. 记录结果
        if len(hits) > 0:
            log_str = f"File: {pdb_file} | Hits: {hits}"
            results.append(log_str)
            
            # 这里的逻辑是：如果该蛋白有两个目标GO，且这两个都命中了
            if len(target_go_idxs) >= 2 and len(hits) == len(target_go_idxs):
                double_hits.append(log_str)

    # 5. 保存结果
    with open(OUTPUT_LOG, 'w') as f:
        f.write('\n'.join(results))
    
    with open(OUTPUT_DOUBLE_HIT, 'w') as f:
        f.write('\n'.join(double_hits))
        
    print(f"\nProcessing complete.")
    print(f"Total files with at least one motif match (<{RMSD_THRESHOLD}A): {len(results)}")
    print(f"Total files with ALL target motif matches: {len(double_hits)}")
    print(f"Results saved to {OUTPUT_LOG} and {OUTPUT_DOUBLE_HIT}")

if __name__ == '__main__':
    main()