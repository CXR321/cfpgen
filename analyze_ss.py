import os
import pandas as pd
from Bio.PDB import PDBParser, PPBuilder

# -------------------------------
# 配置
# -------------------------------
PDB_FOLDER = "./generation-results-dplm2-goonly-alldata-dm-ca-weight-headclloss-2.0_sn-pn-8wstep/esmfold_pdb/"

# -------------------------------
# 二级结构分类函数
# -------------------------------
def classify_phi_psi(phi, psi):
    if phi is None or psi is None:
        return "C"
    if -160 < phi < -40 and -90 < psi < -10:
        return "H"  # alpha helix
    if -180 < phi < -40 and 90 < psi < 180:
        return "E"  # beta sheet
    return "C"

# -------------------------------
# 单个 PDB 统计函数
# -------------------------------
def ss_stats_fast(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)

    counts = {"H": 0, "E": 0, "C": 0}

    for pp in PPBuilder().build_peptides(structure):
        for phi, psi in pp.get_phi_psi_list():
            ss = classify_phi_psi(phi, psi)
            counts[ss] += 1

    total = sum(counts.values())
    fractions = {k+"_frac": counts[k]/total if total>0 else 0 for k in counts}
    return {**counts, **fractions, "n_res": total}

# -------------------------------
# 批量处理 folder
# -------------------------------
def main():
    records = []
    pdb_files = [f for f in os.listdir(PDB_FOLDER) if f.endswith(".pdb")]

    for fname in pdb_files:
        pdb_path = os.path.join(PDB_FOLDER, fname)
        try:
            stats = ss_stats_fast(pdb_path)
            stats["pdb"] = fname
            records.append(stats)
        except Exception as e:
            print(f"[WARN] Failed on {fname}: {e}")

    df = pd.DataFrame(records)
    df.to_csv("ss_stats_fast_results.csv", index=False)

    print("Per-PDB statistics saved to ss_stats_fast_results.csv")
    print("Folder-level mean fractions:")
    print(df[["H_frac", "E_frac", "C_frac"]].mean())

if __name__ == "__main__":
    # main()
    print("bad! use analyze_plddt.py first")
