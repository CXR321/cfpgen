import pickle
with open('/AIRvePFS/dair/chenxr-data/repo/cfpgen/data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl', 'rb') as f:
    data = pickle.load(f)
print(f"Type of data: {type(data)}")
if isinstance(data, list) and len(data) > 0:
    print(f"Number of items: {len(data)}")
    print(f"First item keys: {data[0].keys()}")
    print(f"First item: {data[0]}")
elif isinstance(data, dict):
    print(f"Keys: {list(data.keys())[:10]}")
