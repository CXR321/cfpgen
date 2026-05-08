import pickle
import gzip
from tqdm import tqdm

def get_pdb_uniprot_ids(tsv_file):
    print(f"Loading PDB to UniProt mapping from {tsv_file}...")
    uniprot_ids_with_pdb = set()
    with gzip.open(tsv_file, 'rt') as f:
        for line in f:
            if line.startswith('#') or line.startswith('PDB'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                uniprot_ids_with_pdb.add(parts[2])
    print(f"Found {len(uniprot_ids_with_pdb)} unique UniProt IDs with PDB structures.")
    return uniprot_ids_with_pdb

def calculate_proportion(pkl_file, uniprot_ids_with_pdb):
    print(f"Loading dataset from {pkl_file}...")
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    total_proteins = len(data)
    pdb_count = 0
    
    for item in tqdm(data, desc="Checking proteins"):
        uniprot_id = item.get('uniprot_id')
        if uniprot_id in uniprot_ids_with_pdb:
            pdb_count += 1
            
    proportion = pdb_count / total_proteins if total_proteins > 0 else 0
    
    print("\n--- Results ---")
    print(f"Total proteins in dataset: {total_proteins}")
    print(f"Proteins with PDB structure: {pdb_count}")
    print(f"Proportion: {proportion:.2%} ({proportion})")
    
if __name__ == "__main__":
    tsv_file = "pdb_chain_uniprot.tsv.gz"
    pkl_file = "/AIRvePFS/dair/chenxr-data/repo/cfpgen/data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl"
    
    uniprot_ids_with_pdb = get_pdb_uniprot_ids(tsv_file)
    calculate_proportion(pkl_file, uniprot_ids_with_pdb)
