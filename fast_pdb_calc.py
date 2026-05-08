import pickle
import requests
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

def fetch_pdb_batch(uids):
    query = " OR ".join([f"accession:{u}" for u in uids])
    url = f"https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": query,
        "fields": "accession,xref_pdb",
        "size": 500
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                results = {}
                for entry in data.get('results', []):
                    acc = entry['primaryAccession']
                    xrefs = entry.get('uniProtKBCrossReferences', [])
                    pdb_ids = [ref['id'] for ref in xrefs if ref['database'] == 'PDB']
                    results[acc] = pdb_ids
                return results
            elif response.status_code == 429:
                time.sleep(2 ** attempt)
            else:
                return {}
        except Exception as e:
            time.sleep(2 ** attempt)
    return {}

def main():
    pkl_file = "/AIRvePFS/dair/chenxr-data/repo/cfpgen/data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl"
    print(f"Loading {pkl_file}...")
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
        
    uids = list(set([item['uniprot_id'] for item in data if 'uniprot_id' in item]))
    print(f"Total unique UniProt IDs: {len(uids)}")
    
    batch_size = 100
    batches = [uids[i:i + batch_size] for i in range(0, len(uids), batch_size)]
    
    pdb_mapping = {}
    
    print(f"Fetching PDB data in {len(batches)} batches...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_pdb_batch, batch): batch for batch in batches}
        for future in tqdm(as_completed(futures), total=len(batches)):
            result = future.result()
            if result:
                pdb_mapping.update(result)
            
    total_proteins = len(data)
    pdb_count = 0
    for item in data:
        uid = item.get('uniprot_id')
        if uid in pdb_mapping and len(pdb_mapping[uid]) > 0:
            pdb_count += 1
            
    proportion = pdb_count / total_proteins if total_proteins > 0 else 0
    print("\n=== Results ===")
    print(f"Total proteins in dataset: {total_proteins}")
    print(f"Proteins with PDB structure: {pdb_count}")
    print(f"Proportion: {proportion:.2%} ({proportion})")

if __name__ == "__main__":
    main()
