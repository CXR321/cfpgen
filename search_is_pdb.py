import requests

def has_experimental_structure(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error fetching data for {uniprot_id}")
        return False
        
    data = response.json()
    
    # 遍历交叉引用数据库，寻找 PDB
    pdb_entries = []
    if 'uniProtKBCrossReferences' in data:
        for ref in data['uniProtKBCrossReferences']:
            if ref['database'] == 'PDB':
                # 提取 PDB ID 和解析方法
                method = next((prop['value'] for prop in ref['properties'] if prop['key'] == 'Method'), 'Unknown')
                resolution = next((prop['value'] for prop in ref['properties'] if prop['key'] == 'Resolution'), 'N/A')
                chains = next((prop['value'] for prop in ref['properties'] if prop['key'] == 'Chains'), 'N/A')
                
                pdb_entries.append({
                    'pdb_id': ref['id'],
                    'method': method,
                    'resolution': resolution,
                    'chains': chains
                })
                
    if pdb_entries:
        print(f"✅ {uniprot_id} 拥有 {len(pdb_entries)} 个真实结构数据。")
        for entry in pdb_entries:
            print(f"  - PDB ID: {entry['pdb_id']} | 方法: {entry['method']} | 分辨率: {entry['resolution']} | 覆盖范围: {entry['chains']}")
        return True
    else:
        print(f"❌ {uniprot_id} 目前没有 PDB 实验结构。")
        return False

# 测试示例
has_experimental_structure("P04637")  # 人类 p53 蛋白，有大量真实结构
has_experimental_structure("A0A024RBG1") # 某些无特征的 TrEMBL 蛋白，通常没有