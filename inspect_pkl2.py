import pickle
with open('/AIRvePFS/dair/chenxr-data/repo/cfpgen/data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl', 'rb') as f:
    data = pickle.load(f)
c = sum(1 for item in data if item.get('pdb_ids'))
print(f"Items with non-empty pdb_ids: {c}")
