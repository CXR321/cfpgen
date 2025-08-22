import struct
from datasets import load_dataset
import pickle
import os
import requests
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from biotite.sequence.io import fasta

# Login using e.g. `huggingface-cli login` to access this dataset

with open("data-bin/uniprotKB/cfpgen_general_dataset/train_expanded.pkl", "rb") as f:
    train_data_expanded = pickle.load(f)

with open("data-bin/uniprotKB/cfpgen_general_dataset/valid_expanded.pkl", "rb") as f:
    valid_data_expanded = pickle.load(f)

# test = train_data_expanded[0]
# print(test)
# print(len(test["sequence"]))
# print(test["sequence"])
# print(len(test["aa_seq"]))
# print(test["aa_seq"])
# print(len(test["struct_seq"]))
# print(test["struct_seq"])

# length = [len(item["sequence"]) for item in train_data_expanded]
# length = sorted(length)

# valid_length = [len(item["aa_seq"]) for item in valid_data_expanded]
# valid_length = sorted(valid_length)

# print(length)
# print(valid_length)

# print(train_data_expanded[1])

def find_motif_in_aa_seq(aa_seq, motif_segment, seq, name):
    """在 aa_seq 中查找 motif_segment 的准确位置"""
    index = aa_seq.find(motif_segment)
    if index == -1:
        print(f"❌ Motif segment {motif_segment} not found in aa_seq!")
        print(f"aa_seq: {aa_seq}")
        print(f"seq: {seq}")
        print(f"name: {name}")

        # raise ValueError("❌ Motif segment not found in aa_seq!")
        return None, None
    return index, index + len(motif_segment) - 1  # 返回 start, end (0-based)

for data in train_data_expanded:

    aa_seq = data['aa_seq']
    struct_seq = data['struct_seq'].split(',')
    try:
        motif_info = data['motif'][0]  # 假设只有一个 motif
        motif_segment = motif_info['motif_segment']
    except:
        print(data['uniprot_id'], "has no motif info")
        print(data['motif'])
        continue

    # 重新计算 motif 在 aa_seq 中的位置
    start, end = find_motif_in_aa_seq(aa_seq, motif_segment, data['sequence'], data['uniprot_id'])

    if start is None:
        continue

    # 提取 struct_seq 对应位置的 tokens
    struct_tokens = struct_seq[start : end + 1]  # Python 切片是 [start, end+1)

    # print(f"✅ Motif segment found at aa_seq positions: {start+1}-{end+1} (1-based)")
    # print(f"Extracted {len(struct_tokens)} struct tokens:")
    # print(struct_tokens[:10], "...", struct_tokens[-10:])  # 显示前10和后10个 token
    terms = [term['go_term'] for term in data['motif']]
    print(f"motif_num: {terms} {motif_info['go_term']}: {struct_tokens} {data['uniprot_id']} {aa_seq[start:end+1]}")

exit()

swiss_prot = load_dataset("airkingbd/pdb_swissprot", "train", cache_dir="data-bin")
# print(ds['train'][1])
# ds = load_dataset("airkingbd/pdb_swissprot", "valid", cache_dir="data-bin")
# print(ds.keys())
# print(ds['train'][1])

swiss_train = swiss_prot["train"]

print(swiss_train[0])

# pdb_map = {}
# for entry in swiss_train:
#     pdb_name = entry["pdb_name"]
#     if pdb_name.startswith("AF-") and "-model_v" in pdb_name:
#         uniprot_id = pdb_name.split("-")[1]  # "AF-Q60888-model_v4" -> "Q60888"
#         pdb_map[uniprot_id] = pdb_name

# 构建 swiss_train 的索引：{uniprot_id: entry}
swiss_index = {}
for entry in swiss_train:
    pdb_name = entry["pdb_name"]
    if pdb_name.startswith("AF-") and "-model_v" in pdb_name:
        uniprot_id = pdb_name.split("-")[1]
        swiss_index[uniprot_id] = entry

with open("data-bin/uniprotKB/cfpgen_general_dataset/train.pkl", 'rb') as f:
    train_data = pickle.load(f)

with open("data-bin/uniprotKB/cfpgen_general_dataset/valid.pkl", 'rb') as f:
    valid_data = pickle.load(f)

with open("data-bin/uniprotKB/cfpgen_general_dataset/test.pkl", 'rb') as f:
    test_data = pickle.load(f)


aa_seq = fasta.FastaFile.read("/AIRvePFS/dair/chenxr-data/repo/cfpgen/data-bin/tokenized_missed_pdb/aa_seq.fasta")
struct_seq = fasta.FastaFile.read("/AIRvePFS/dair/chenxr-data/repo/cfpgen/data-bin/tokenized_missed_pdb/struct_seq.fasta")

aa_seq_dict = dict(aa_seq.items())
struct_seq_dict = dict(struct_seq.items())



# 通用处理函数，返回：扩展后数据 + 缺失ID列表
def expand_and_filter_dataset(dataset, dataset_name):
    expanded = []
    missing_ids = []

    for item in tqdm(dataset, desc=f"Processing {dataset_name}"):
        uniprot_id = item.get("uniprot_id")
        pdb_key = f"AF-{uniprot_id}-F1-model_v4"

        aa_seq = None
        struct_seq = None

        # 优先从 swiss_train 中找
        if uniprot_id in swiss_index:
            entry = swiss_index[uniprot_id]
            aa_seq = entry.get("aa_seq")
            struct_seq = entry.get("struct_seq")
        # 其次尝试从 fasta 文件中找
        elif pdb_key in aa_seq_dict and pdb_key in struct_seq_dict:
            # print("in")
            aa_seq = aa_seq_dict[pdb_key]
            struct_seq = struct_seq_dict[pdb_key]

        # 判断是否找到序列
        if aa_seq and struct_seq:
            item["aa_seq"] = aa_seq
            item["struct_seq"] = struct_seq
            expanded.append(item)
        else:
            missing_ids.append(uniprot_id)

    return expanded, missing_ids

# 处理三个数据集
train_data_expanded, train_missing = expand_and_filter_dataset(train_data, "train")
valid_data_expanded, valid_missing = expand_and_filter_dataset(valid_data, "valid")
test_data_expanded, test_missing = expand_and_filter_dataset(test_data, "test")

# 保存扩展后的数据集（已剔除缺失项）
with open("data-bin/uniprotKB/cfpgen_general_dataset/train_expanded.pkl", "wb") as f:
    pickle.dump(train_data_expanded, f)

with open("data-bin/uniprotKB/cfpgen_general_dataset/valid_expanded.pkl", "wb") as f:
    pickle.dump(valid_data_expanded, f)

with open("data-bin/uniprotKB/cfpgen_general_dataset/test_expanded.pkl", "wb") as f:
    pickle.dump(test_data_expanded, f)

# 汇总所有缺失项并保存
all_missing = set(train_missing + valid_missing + test_missing)
with open("missing_uniprot_ids.txt", "w") as f:
    for uid in sorted(all_missing):
        f.write(uid + "\n")

print(f"Total entries removed due to missing seq: {len(all_missing)}")
print(f"Final train: {len(train_data_expanded)}  valid: {len(valid_data_expanded)}  test: {len(test_data_expanded)}")

exit()


# print(dict(aa_seq.items()))
# print(dict(struct_seq.items())["AF-Q3KI22-F1-model_v4"])

# print(f"len train data: {len(train_data)}")
# print(f"len valid data: {len(valid_data)}")
# print(f"len test data: {len(test_data)}")

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

# missing = [item["uniprot_id"] for item in test_data if item["uniprot_id"] not in pdb_map]
# print(f"Missing PDB for {len(missing)} entries:", missing[:5])

# download_all_with_multiprocessing(missing, save_dir, num_workers=128)

