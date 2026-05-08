import pickle

import requests
from tqdm import tqdm

def create_prompt(descriptions, style='instruction'):
    """
    将描述列表组合成一个字符串 Prompt。
    
    Args:
        descriptions (list): 描述字符串列表，如 ['activity A', 'activity B']
        style (str): 组合风格 ('simple', 'sentence', 'tags')
    
    Returns:
        str: 组合后的 Prompt
    """
    # 过滤掉 None 或空字符串，去重（可选）
    clean_desc = [d for d in descriptions if d]
    # clean_desc = list(set(clean_desc)) # 如果需要去重，取消注释这一行
    
    if not clean_desc:
        return "Protein with unknown function."

    if style == 'simple':
        # 风格 1: 简单的逗号分隔 (适合只需关键词的模型)
        # 输出: "G protein-coupled receptor activity, olfactory receptor activity"
        return ", ".join(clean_desc)

    elif style == 'sentence':
        # 风格 2: 自然语言句子 (适合 T5, GPT 等文本生成模型)
        # 输出: "This protein exhibits G protein-coupled receptor activity and olfactory receptor activity."
        if len(clean_desc) == 1:
            joined = clean_desc[0]
        else:
            # 处理最后一个 "and"
            joined = ", ".join(clean_desc[:-1]) + " and " + clean_desc[-1]
        
        return f"This protein exhibits {joined}."

    elif style == 'instruction':
        # 风格 3: 指令格式 (适合指令微调模型)
        # 输出: "Generate a protein sequence that functions as: activity A, activity B."
        return f"Generate a protein that functions as: {', '.join(clean_desc)}."

    return ""

go_id2name = {}

def get_go_name_from_web(go_id):
    """
    访问 QuickGO API 获取指定 GO ID 的 name。
    例如: 输入 'GO:0004933' -> 返回 'mating-type a-factor pheromone receptor activity'
    """
    # 构建 URL
    url = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{go_id}/complete"

    if go_id in go_id2name:
        return go_id2name[go_id]
    
    try:
        # 发送 GET 请求
        response = requests.get(url, headers={"Accept": "application/json"})
        
        # 检查请求是否成功 (状态码 200)
        if response.status_code == 200:
            data = response.json()
            
            # 检查是否有结果
            if data['numberOfHits'] > 0:
                # 提取 'results' 列表中第一项的 'name' 字段
                go_id2name[go_id] = data['results'][0]['name']
                return data['results'][0]['name']
            else:
                return "No results found"
        else:
            return f"Error: Status code {response.status_code}"
            
    except Exception as e:
        return f"Request failed: {e}"

with open('go_mapping.pkl', 'rb') as f:
    go_mapping = pickle.load(f)

print(go_mapping)

exit()

with open('data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl', 'rb') as f:
    test_data = pickle.load(f)

# print(test_data[0])

index_to_go = {v: k for k, v in go_mapping.items()}

test_data_desc = []

for data in tqdm(test_data):
    protein_id = data['uniprot_id']

    condition_ids = data['go_f_mapped']

    condition_go_ids = [index_to_go[i] for i in condition_ids]

    # print(protein_id, condition_go_ids)

    condition_go_descs = [get_go_name_from_web(go_id) for go_id in condition_go_ids]

    # print(condition_go_descs)
    test_data_desc.append({
            'id': protein_id,
            'conditions': condition_go_ids,
            'conditions_desc': condition_go_descs,
            'conditions_prompt': create_prompt(condition_go_descs, style='instruction')
        })
    
    # print(test_data_desc[-1])
    # exit()

with open('test_data_desc.pkl', 'wb') as f:
    pickle.dump(test_data_desc, f)

print(f"Saved {len(test_data_desc)} entries to test_data_desc.pkl")
print(f"Example entry: {test_data_desc[0]}")