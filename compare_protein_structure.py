import sys
import numpy as np
from Bio import pairwise2
from Bio.PDB import PDBParser, Superimposer, PPBuilder
from Bio.SeqUtils import seq1


def extract_sequence_and_ca_atoms(pdb_file):
    """
    从 PDB 中提取：
    1) 蛋白一级序列
    2) 对应的 Cα 原子（按残基顺序）
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    ppb = PPBuilder()
    peptides = ppb.build_peptides(structure)

    if len(peptides) == 0:
        raise ValueError(f"No peptide found in {pdb_file}")

    # 假设单链、主链
    seq = "".join(str(p.get_sequence()) for p in peptides)

    ca_atoms = []
    for model in structure:
        for chain in model:
            for res in chain:
                if "CA" in res:
                    ca_atoms.append(res["CA"])
        break  # 只取第一个 model

    return seq, ca_atoms


def compute_sequence_identity(seq1, seq2):
    """
    全局序列比对并计算 identity
    """
    alignments = pairwise2.align.globalxx(seq1, seq2)
    best = alignments[0]

    aligned_seq1, aligned_seq2 = best.seqA, best.seqB

    matches = sum(
        a == b and a != "-"
        for a, b in zip(aligned_seq1, aligned_seq2)
    )
    aligned_length = sum(
        a != "-" and b != "-"
        for a, b in zip(aligned_seq1, aligned_seq2)
    )

    identity = matches / aligned_length if aligned_length > 0 else 0.0
    return identity, aligned_seq1, aligned_seq2


def compute_structural_rmsd(ca_atoms_1, ca_atoms_2, aligned_seq1, aligned_seq2):
    """
    根据序列对齐结果，筛选对应残基的 Cα 原子，
    使用 Bio.PDB.Superimposer 进行结构对齐并计算 RMSD
    """
    atoms_1 = []
    atoms_2 = []

    idx1 = 0
    idx2 = 0

    for a, b in zip(aligned_seq1, aligned_seq2):
        if a != "-" and b != "-":
            atoms_1.append(ca_atoms_1[idx1])
            atoms_2.append(ca_atoms_2[idx2])
        if a != "-":
            idx1 += 1
        if b != "-":
            idx2 += 1

    if len(atoms_1) < 3:
        raise ValueError("Not enough aligned residues for RMSD calculation")

    sup = Superimposer()
    sup.set_atoms(atoms_1, atoms_2)

    return sup.rms, len(atoms_1)



def evaluate_similarity(seq_id, rmsd):
    """
    给出简单定性评价
    """
    if seq_id > 0.9 and rmsd < 2.0:
        return "Structures are highly similar; prediction is very reliable."
    elif seq_id > 0.7 and rmsd < 3.5:
        return "Structures are moderately similar; overall fold is consistent."
    else:
        return "Noticeable differences observed; structural deviation exists."


def main(pdb1, pdb2):
    seq1, ca1 = extract_sequence_and_ca_atoms(pdb1)
    seq2, ca2 = extract_sequence_and_ca_atoms(pdb2)

    print(seq1)
    print()
    print(seq2)

    print()

    seq_id, aln1, aln2 = compute_sequence_identity(seq1, seq2)

    print(aln1)
    print()
    print(aln2)


    rmsd, n_aligned = compute_structural_rmsd(ca1, ca2, aln1, aln2)

    print("=" * 60)
    print("Protein Similarity Comparison")
    print("=" * 60)
    print(f"PDB 1: {pdb1}")
    print(f"PDB 2: {pdb2}")
    print("-" * 60)
    print(f"Sequence identity      : {seq_id * 100:.2f}%")
    print(f"Aligned residues (Cα)  : {n_aligned}")
    print(f"Structural RMSD (Å)    : {rmsd:.3f}")
    print("-" * 60)
    print("Evaluation:")
    print(evaluate_similarity(seq_id, rmsd))
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_protein_similarity.py pdb1.pdb pdb2.pdb")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
