from datasets import load_dataset
import pickle
import os
import requests
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# Login using e.g. `huggingface-cli login` to access this dataset
swiss_prot = load_dataset("airkingbd/pdb_swissprot", "train", cache_dir="data-bin")
# print(ds['train'][1])
# ds = load_dataset("airkingbd/pdb_swissprot", "valid", cache_dir="data-bin")
# print(ds.keys())
# print(ds['train'][1])

swiss_train = swiss_prot["train"]

pdb_map = {}
for entry in swiss_train:
    pdb_name = entry["pdb_name"]
    if pdb_name.startswith("AF-") and "-model_v" in pdb_name:
        uniprot_id = pdb_name.split("-")[1]  # "AF-Q60888-model_v4" -> "Q60888"
        pdb_map[uniprot_id] = pdb_name

# with open("data-bin/uniprotKB/cfpgen_general_dataset/train.pkl", 'rb') as f:
#     train_data = pickle.load(f)

# with open("data-bin/uniprotKB/cfpgen_general_dataset/valid.pkl", 'rb') as f:
#     valid_data = pickle.load(f)

with open("data-bin/uniprotKB/cfpgen_general_dataset/test.pkl", 'rb') as f:
    test_data = pickle.load(f)


# print(f"len train data: {len(train_data)}")
# print(f"len valid data: {len(valid_data)}")
print(f"len test data: {len(test_data)}")

# exit()

save_dir = "data-bin/missed_pdb"

# 单个下载任务
def download_pdb(uniprot_id, save_dir):
    save_path = os.path.join(save_dir, f"AF-{uniprot_id}-F1-model_v4.pdb")
    
    # 跳过已存在文件
    if os.path.exists(save_path):
        return (uniprot_id, "already_exists")

    pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    try:
        response = requests.get(pdb_url, timeout=15)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return (uniprot_id, "success")
        else:
            return (uniprot_id, f"failed_{response.status_code}")
    except Exception as e:
        return (uniprot_id, f"error_{str(e)}")

# 下载所有 missing ids
def download_all_with_multiprocessing(uniprot_ids, save_dir, num_workers=None):
    if num_workers is None:
        num_workers = min(16, cpu_count())  # 限制最大进程数，防止过载

    with Pool(processes=num_workers) as pool:
        # 使用 partial 传参
        job_fn = partial(download_pdb, save_dir=save_dir)
        results = list(tqdm(pool.imap(job_fn, uniprot_ids), total=len(uniprot_ids), desc="Downloading PDBs"))

    # 处理结果
    failed = [(uid, status) for uid, status in results if status != "success" and status != "already_exists"]

    if failed:
        with open("failed_ids.txt", "w") as f:
            for uid, status in failed:
                f.write(f"{uid}\t{status}\n")

    print(f"✅ Done. {len(uniprot_ids) - len(failed)} succeeded, {len(failed)} failed.")
    if failed:
        print("❌ Failed IDs saved to failed_ids.txt")


# missing = [item["uniprot_id"] for item in train_data if item["uniprot_id"] not in pdb_map]
# print(f"Missing PDB for {len(missing)} entries:", missing[:5])

# download_all_with_multiprocessing(missing, save_dir, num_workers=128)

# missing = [item["uniprot_id"] for item in valid_data if item["uniprot_id"] not in pdb_map]
# print(f"Missing PDB for {len(missing)} entries:", missing[:5])

# download_all_with_multiprocessing(missing, save_dir, num_workers=128)

missing = [item["uniprot_id"] for item in test_data if item["uniprot_id"] not in pdb_map]
print(f"Missing PDB for {len(missing)} entries:", missing[:5])

download_all_with_multiprocessing(missing, save_dir, num_workers=128)

