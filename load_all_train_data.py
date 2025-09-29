import struct
from datasets import load_dataset
import pickle
import os
import requests
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from biotite.sequence.io import fasta
from get_emb_cal import load_embeddings
from get_emb import find_pfam_in_aaseq, find_motif_in_aa_seq
import torch
import numpy as np
from motif_search_web import parse_pfam2go_id2go


def get_ori_all_data():

    swiss_prot = load_dataset("airkingbd/pdb_swissprot", "train", cache_dir="data-bin")
    # print(ds['train'][1])
    # ds = load_dataset("airkingbd/pdb_swissprot", "valid", cache_dir="data-bin")
    # print(ds.keys())
    # print(ds['train'][1])

    swiss_train = swiss_prot["train"]

    print(swiss_train[0])

    pdb_map = {}
    for entry in swiss_train:
        pdb_name = entry["pdb_name"]
        if pdb_name.startswith("AF-") and "-model_v" in pdb_name:
            uniprot_id = pdb_name.split("-")[1]  # "AF-Q60888-model_v4" -> "Q60888"
            pdb_map[uniprot_id] = pdb_name

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
                item["aa_seq"] = item.get("sequence")
                item["struct_seq"] = ""
                expanded.append(item)

        return expanded, missing_ids

    # 处理三个数据集
    train_data_expanded, train_missing = expand_and_filter_dataset(train_data, "train")
    valid_data_expanded, valid_missing = expand_and_filter_dataset(valid_data, "valid")
    test_data_expanded, test_missing = expand_and_filter_dataset(test_data, "test")

    # 保存扩展后的数据集（已剔除缺失项）
    with open("data-bin/uniprotKB/cfpgen_general_dataset/train_all_expanded.pkl", "wb") as f:
        pickle.dump(train_data_expanded, f)

    with open("data-bin/uniprotKB/cfpgen_general_dataset/valid_all_expanded.pkl", "wb") as f:
        pickle.dump(valid_data_expanded, f)

    with open("data-bin/uniprotKB/cfpgen_general_dataset/test_all_expanded.pkl", "wb") as f:
        pickle.dump(test_data_expanded, f)

    # 汇总所有缺失项并保存
    all_missing = set(train_missing + valid_missing + test_missing)
    with open("missing_uniprot_ids.txt", "w") as f:
        for uid in sorted(all_missing):
            f.write(uid + "\n")

    print(f"Total entries removed due to missing seq: {len(all_missing)}")
    print(f"Final train: {len(train_data_expanded)}  valid: {len(valid_data_expanded)}  test: {len(test_data_expanded)}")

def get_ori_all_motif(name):
    with open(f"data-bin/uniprotKB/cfpgen_general_dataset/{name}_data_motif_emb.pkl", "rb") as f:
        my_data = pickle.load(f)

    with open(f"data-bin/uniprotKB/cfpgen_general_dataset/{name}_all_expanded.pkl", "rb") as f:
        all_data = pickle.load(f)

    map_dict = {}
    for item in my_data:
        uniprot_id = item['uniprot_id']
        map_dict[uniprot_id] = item

    motif_added_all_data = []

    print(f"alldata: {len(all_data)}; olddata: {len(my_data)}")

    for data in tqdm(all_data, desc=f"Processing {name}"):

        
        pdb_id = data['uniprot_id']
        aa_seq = data['aa_seq']

        if pdb_id in map_dict:
            data['motif_mask'] = map_dict[pdb_id]['motif_mask']
            data['motif_struct_emb'] = map_dict[pdb_id]['motif_struct_emb']         
        else:
            data['motif_mask'] = torch.zeros(len(aa_seq), dtype=torch.bool)
            data['motif_struct_emb'] = torch.zeros(7, 1280, dtype=torch.float32)

        motif_added_all_data.append(data)

    with open(f"data-bin/uniprotKB/cfpgen_general_dataset/{name}_all_old_motif_added.pkl", "wb") as f:
        pickle.dump(motif_added_all_data, f)

def load_all_motif_data(name):
    with open(f"data-bin/uniprotKB/cfpgen_general_dataset/{name}_all_old_motif_added.pkl", "rb") as f:
        all_data = pickle.load(f)
    return all_data

def load_all_pfam_data(name):
    with open(f"data-bin/uniprotKB/cfpgen_general_dataset/{name}_all_old_motif_added_pfamMotif.pkl", "rb") as f:
        all_data = pickle.load(f)
    return all_data
def save_all_pfam_data(name, data):
    with open(f"data-bin/uniprotKB/cfpgen_general_dataset/{name}_all_old_motif_added_pfamMotif_esmfold.pkl", "wb") as f:
        pickle.dump(data, f)

def load_all_pfam_esm_data(name):
    with open(f"data-bin/uniprotKB/cfpgen_general_dataset/{name}_all_old_motif_added_pfamMotif_esmfold.pkl", "rb") as f:
        all_data = pickle.load(f)
    return all_data

def save_all_pfam_emb_data(name, data):
    with open(f"data-bin/uniprotKB/cfpgen_general_dataset/{name}_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl", "wb") as f:
        pickle.dump(data, f)

def load_all_pfam_emb_data(name):
    with open(f"data-bin/uniprotKB/cfpgen_general_dataset/{name}_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl", "rb") as f:
        all_data = pickle.load(f)
    return all_data

def load_pfam_emb_data():
    with open(f"data-bin/uniprotKB/cfpgen_general_dataset/train_pfam_cls_emb_pfamMotif.pkl", "rb") as f:
        pfam_data = pickle.load(f)

    filtered_go_terms_emb = {}

    for key, value_list in pfam_data.items():
        filtered_values = []
        for value_set in value_list:

            feature_vector, e_value = value_set 
            if e_value <= 0.05:
                filtered_values.append(torch.tensor(feature_vector))
                # 如果数据结构不同，可以根据实际情况调整
        
        # 只有当过滤后还有数据时才保留这个key
        if filtered_values:
            filtered_go_terms_emb[key] = torch.stack(filtered_values)

    return filtered_go_terms_emb
    return pfam_data

def load_pfam_emb_data_goterm():
    with open(f"data-bin/uniprotKB/cfpgen_general_dataset/train_pfam_cls_emb_pfamMotif.pkl", "rb") as f:
        pfam_data = pickle.load(f)

    pfamid2gomap = parse_pfam2go_id2go()

    filtered_go_terms_emb = {}

    for key, value_list in pfam_data.items():
        filtered_values = []
        for value_set in value_list:

            feature_vector, e_value = value_set 
            if e_value <= 0.05:
                filtered_values.append(torch.tensor(feature_vector))
                # 如果数据结构不同，可以根据实际情况调整
        
        # 只有当过滤后还有数据时才保留这个key
        if filtered_values:
            filtered_go_terms_emb[key] = filtered_values

    go_id2pfam_emb = {}

    for key, value in filtered_go_terms_emb.items():
        go_ids = pfamid2gomap.get(key, [])
        for go_id, go_desc in go_ids:
            if go_id not in go_id2pfam_emb:
                go_id2pfam_emb[go_id] = []
            go_id2pfam_emb[go_id].extend(value)
    
    return go_id2pfam_emb

def load_test_data_motif_emb():
    with open(f"data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added_pfamMotif.pkl", "rb") as f:
        test_data = pickle.load(f)
    return test_data


def fasta_no_struct_data():
    train_data = load_all_pfam_data("train")
    valid_data = load_all_pfam_data("valid")

    # 合并训练和验证数据
    all_data = train_data + valid_data
    
    # 筛选出struct_seq长度为0的蛋白质
    no_struct_proteins = []
    for data in all_data:
        # 检查struct_seq字段是否存在且长度为0
        struct_seq = data.get('struct_seq', '')
        if struct_seq == '' or (isinstance(struct_seq, str) and len(struct_seq.strip()) == 0):
            no_struct_proteins.append(data)
    
    # 创建FASTA格式的内容
    fasta_content = ""
    for protein in no_struct_proteins:
        uniprot_id = protein.get('uniprot_id', '')
        sequence = protein.get('sequence', '')
        
        # 确保有有效的ID和序列
        if uniprot_id and sequence:
            fasta_content += f">SEQUENCE_ID={uniprot_id}\n{sequence}\n"
    
    # 写入FASTA文件
    output_filename = "data-bin/uniprotKB/cfpgen_general_dataset/no_struct_proteins.fasta"
    with open(output_filename, 'w') as fasta_file:
        fasta_file.write(fasta_content)
    
    print(f"找到 {len(no_struct_proteins)} 个没有结构数据的蛋白质")
    print(f"结果已保存到 {output_filename}")
    
    return no_struct_proteins


def load_struct_token():
    aa_seq = fasta.FastaFile.read("/AIRvePFS/dair/chenxr-data/repo/cfpgen/data-bin/uniprotKB/cfpgen_general_dataset/esmfold_pdb/aa_seq.fasta")
    struct_seq = fasta.FastaFile.read("/AIRvePFS/dair/chenxr-data/repo/cfpgen/data-bin/uniprotKB/cfpgen_general_dataset/esmfold_pdb/struct_seq.fasta")

    aa_seq_dict = dict(aa_seq.items())
    struct_seq_dict = dict(struct_seq.items())

    # 创建映射字典：原始ID -> 简化ID
    id_mapping = {}
    plddt_mapping = {}
    for key in aa_seq.keys():
        if key.startswith('SEQUENCE_ID='):
            # 提取"A7F996"部分
            simplified_id = key.split('=')[1].split('_')[0]
            id_mapping[simplified_id] = key
            plddt_mapping[simplified_id] = float(key.split('=')[1].split('_')[-1])
    
    aa_seq_dict = dict(aa_seq.items())
    struct_seq_dict = dict(struct_seq.items())

    

    def get_struct_seq(data):
        new_data = []
        for item in tqdm(data):
            if len(item['struct_seq']) != 0:
                new_data.append(item)
                continue
            pdb_key = item['uniprot_id']

            if id_mapping.get(pdb_key) is None:
                print(f"Cannot find {pdb_key} in mapping dict")
                new_data.append(item)
                continue

            item['struct_seq'] = struct_seq_dict[id_mapping[pdb_key]]
            item['esmfold_plddt'] = plddt_mapping[pdb_key]
            new_data.append(item)
        return new_data

    data = load_all_pfam_data("train")
    new_data = get_struct_seq(data)
    save_all_pfam_data("train", new_data)

    data = load_all_pfam_data("valid")
    new_data = get_struct_seq(data)
    save_all_pfam_data("valid", new_data)

def load_pfam_emb():
    train_data = load_all_pfam_esm_data("train")
    valid_data = load_all_pfam_esm_data("valid")
    test_data = load_test_data_motif_emb()

    pfam_data = load_pfam_emb_data()
    pfam_data_go = load_pfam_emb_data_goterm()

    def analyze_pfam_coverage(data):
        """
        分析pfam_motif的evalue小于0.05的pfam是否完全覆盖GO F项
        """
        # 目标GO F项
        target_go_f = set(data['go_numbers']['F'])  # {'GO:0030145', 'GO:0070006'}
        
        # 筛选evalue < 0.05的pfam
        significant_pfams = []
        covered_go_ids = set()
        try:
            for pfam in data['pfam_motif']:
                # 检查evalue条件
                if pfam['evalue'] < 0.05:
                    # 收集该pfam的strong_go_id
                    if 'strong_go_id' in pfam:

                        go_ids_set = set(pfam['strong_go_id'])

                        if not go_ids_set.issubset(target_go_f) or len(go_ids_set) == 0:
                            continue

                        s, e = find_pfam_in_aaseq(data['aa_seq'], data['sequence'], pfam['start'], pfam['end'])

                        if s == None or e == None:
                            continue

                        covered_go_ids.update(pfam['strong_go_id'])

                        pfam['aa_s'] = s
                        pfam['aa_e'] = e

                        significant_pfams.append(pfam)
            
            # 检查是否完全覆盖
            is_fully_covered = target_go_f.issubset(covered_go_ids)

            # 记录符合条件的pfam信息
            result = {
                'is_fully_covered': is_fully_covered,
                'target_go_f': list(target_go_f),
                'covered_go_ids': list(covered_go_ids),
                'significant_pfams': []
            }
            
            for pfam in significant_pfams:
                result['significant_pfams'].append({
                    'pfam_id': pfam['pfam_id'],
                    'aa_s': pfam['aa_s'],
                    'aa_e': pfam['aa_e'],
                    'strong_go_id': pfam.get('strong_go_id', [])
                })
            
            return result
        except:
            print(f"Error in {data['uniprot_id']}")
            # print(data)
            # exit()
            return {"is_fully_covered": False}

    def add_pfam_emb(all_data, pfam_emb):
        new_data = []
        for data in all_data:
            res = analyze_pfam_coverage(data)
            if res['is_fully_covered']:
                # 完全覆盖，直接添加pfam_emb
                aa_seq = data['aa_seq']
                motif_num = 0

                motif_mask = torch.zeros(len(aa_seq), dtype=torch.bool)

                motif_struct_emb = torch.zeros(7, 1280, dtype=torch.float32)

                pfam_position_s = []
                pfam_position_e = []

                for pfam in res['significant_pfams']:
                    s, e = pfam['aa_s'], pfam['aa_e']
                    motif_mask[s:e] = True

                    pfam_position_s.append(s)
                    pfam_position_e.append(e)

                    if pfam['pfam_id'] not in pfam_emb:
                        continue

                    motif_struct_emb[motif_num] = pfam_emb[pfam['pfam_id']].mean(dim=0)
                    motif_num += 1
                    if motif_num == 7:
                        break


                data['pfam_mask'] = motif_mask
                data['pfam_emb'] = motif_struct_emb
                data['pfam_position_s'] = pfam_position_s
                data['pfam_position_e'] = pfam_position_e

                if motif_num == 0:
                    data['pfam_emb'] = None
                    data['pfam_mask'] = None

                new_data.append(data)
            else:
                # 不完整，添加None
                data['pfam_emb'] = None
                data['pfam_mask'] = None
                new_data.append(data)
        return new_data  

    def add_pfam_emb_go(all_data, pfam_emb_go):
        new_data = []
        for data in all_data:
            res = analyze_pfam_coverage(data)
            if res['is_fully_covered']:
                # 完全覆盖，直接添加pfam_emb
                aa_seq = data['aa_seq']
                motif_num = 0

                motif_mask = torch.zeros(len(aa_seq), dtype=torch.bool)

                motif_struct_emb = torch.zeros(7, 1280, dtype=torch.float32)

                pfam_position_s = []
                pfam_position_e = []

                for pfam in res['significant_pfams']:
                    s, e = pfam['aa_s'], pfam['aa_e']
                    motif_mask[s:e] = True

                    pfam_position_s.append(s)
                    pfam_position_e.append(e)

                for go_id in data['go_numbers']['F']:
                    motif_struct_emb[motif_num] = torch.stack(pfam_emb_go[go_id]).mean(dim=0)
                    motif_num += 1
                    if motif_num == 7:
                        break


                data['pfam_mask'] = motif_mask
                data['pfam_emb'] = motif_struct_emb
                data['pfam_position_s'] = pfam_position_s
                data['pfam_position_e'] = pfam_position_e

                if motif_num == 0:
                    data['pfam_emb'] = None
                    data['pfam_mask'] = None

                new_data.append(data)
            else:
                # 不完整，添加None
                data['pfam_emb'] = None
                data['pfam_mask'] = None
                new_data.append(data)
        return new_data  

    # new_train = add_pfam_emb(train_data, pfam_data)
    # save_all_pfam_emb_data("train", new_train)

    # new_valid = add_pfam_emb(valid_data, pfam_data)
    # save_all_pfam_emb_data("valid", new_valid)

    # new_test = add_pfam_emb(test_data, pfam_data)
    # save_all_pfam_emb_data("test", new_test)

    new_train = add_pfam_emb_go(train_data, pfam_data_go)
    save_all_pfam_emb_data("train", new_train)

    new_valid = add_pfam_emb_go(valid_data, pfam_data_go)
    save_all_pfam_emb_data("valid", new_valid)

    new_test = add_pfam_emb_go(test_data, pfam_data_go)
    save_all_pfam_emb_data("test", new_test)

    print(new_test[0])

def load_motif_segment_mask_data():


    def modify(data):
        new_data = []
        for meta_data in data:
            item = meta_data
            motif_position_s = []
            motif_position_e = []
            motif_mask = torch.zeros(len(item['aa_seq']), dtype=torch.bool)
            if item.get('motif'):
                for motif in item['motif']:
                    try:
                        s, e = find_motif_in_aa_seq(item['aa_seq'], motif['motif_segment'])
                    except:
                        print(f"Error in {item['uniprot_id']}")
                        # print(item)
                    
                    if s != None and e != None:
                        motif_position_s.append(s)
                        motif_position_e.append(e)
                        motif_mask[s:e] = True
                        continue
                    else:
                        s, e = find_pfam_in_aaseq(item['aa_seq'], item['sequence'], motif['start'], motif['end'])
                        if s != None and e != None:
                            motif_position_s.append(s)
                            motif_position_e.append(e)
                            motif_mask[s:e] = True
                            continue
                        else:
                            continue
                if len(motif_position_s) != 0:
                    item['motif_mask'] = motif_mask
                    item['motif_position_s'] = motif_position_s
                    item['motif_position_e'] = motif_position_e
                else:
                    pass
            new_data.append(item)
        return new_data




    data = load_all_pfam_emb_data("train") # 加载pfam_emb数据
    data = modify(data)
    print(f"example: {data[0]}")
    save_all_pfam_emb_data("train", data)

    data = load_all_pfam_emb_data("valid") # 加载pfam_emb数据
    data = modify(data)
    save_all_pfam_emb_data("valid", data)

    data = load_all_pfam_emb_data("test") # 加载pfam_emb数据
    data = modify(data)
    save_all_pfam_emb_data("test", data)





if __name__ == "__main__":
    # load_struct_token()
    load_pfam_emb()
    # load_motif_segment_mask_data()

# d = load_all_motif_data("train")
# d = load_all_pfam_data("train")

# for i in d:
#     if i.get('uniprot_id') == "A1AAN1":
#         print(i)
#         exit()

# print(d[0])



# get_ori_all_motif("train")
# get_ori_all_motif("valid")
# get_ori_all_motif("test")
# fasta_no_struct_data()