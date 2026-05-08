import pickle
import requests
import gzip

def get_pdb_uniprot_ids(tsv_file):
    uniprot_ids_with_pdb = set()
    with gzip.open(tsv_file, 'rt') as f:
        for line in f:
            if line.startswith('#') or line.startswith('PDB'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                uniprot_ids_with_pdb.add(parts[2])
    return uniprot_ids_with_pdb

uniprot_ids_with_pdb = get_pdb_uniprot_ids("pdb_chain_uniprot.tsv.gz")

with open('/AIRvePFS/dair/chenxr-data/repo/cfpgen/data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl', 'rb') as f:
    data = pickle.load(f)

for item in data:
    uid = item['uniprot_id']
    if not item.get('pdb_ids') and uid in uniprot_ids_with_pdb:
        print(f"Checking {uid} which has empty pdb_ids but is in mapping...")
        url = f"https://rest.uniprot.org/uniprotkb/{uid}.json"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            pdb_entries = []
            if 'uniProtKBCrossReferences' in data:
                for ref in data['uniProtKBCrossReferences']:
                    if ref['database'] == 'PDB':
                        pdb_entries.append(ref['id'])
            print(f"UniProt REST API found PDBs: {pdb_entries}")
        break
