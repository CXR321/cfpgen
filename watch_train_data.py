import pickle
import matplotlib.pyplot as plt
import numpy as np
from sympy import intersection
import random
from load_all_train_data import load_all_pfam_emb_data
import requests
import time
from collections import defaultdict
from collections import Counter
import math

# exit()

def plot_coverage_distribution(data):
    """
    绘制motif覆盖率的分布直方图
    """
    # 收集所有蛋白质的motif覆盖率
    motif_coverages = []
    go_motif_coverages = []
    fully_covered_proteins = []  # 记录哪些蛋白质的GO标签被完全覆盖
    partially_covered_count = []  # 记录哪些蛋白质的GO标签被部分覆盖

    partially_covered_proteins = []  # 记录哪些蛋白质的GO标签被部分覆盖
    no_coverage_proteins = []  # 记录哪些蛋白质的GO标签没有被覆盖
    
    for protein in data:
        seq_length = len(protein['sequence'])
        go_f_terms = protein['go_numbers'].get('F', [])
        
        # 计算motif_segment覆盖率
        motif_segments = protein.get('motif', [])
        if motif_segments:
            motif_lengths = [motif['end'] - motif['start'] + 1 for motif in motif_segments]
            avg_motif_coverage = sum(motif_lengths) / len(motif_lengths) / seq_length
            motif_coverages.append(avg_motif_coverage)
        
        # 计算有GO注释的pfam_motif覆盖率
        pfam_motifs = protein.get('pfam_motif', [])
        go_motif_lengths = []
        pfam_go_terms = set()  # 记录当前蛋白质的所有GO注释
        
        for pfam in pfam_motifs:
            if pfam.get('strong_go_id') and len(pfam['strong_go_id']) >= 1:
                motif_length = pfam['end'] - pfam['start'] + 1
                go_motif_lengths.append(motif_length)
                pfam_go_terms.update(pfam['strong_go_id'])
        
        if go_motif_lengths:
            avg_go_coverage = sum(go_motif_lengths) / len(go_motif_lengths) / seq_length
            go_motif_coverages.append(avg_go_coverage)

        # 检查GO标签是否被完全覆盖
        is_fully_covered = False
        if go_f_terms and pfam_go_terms:
            is_fully_covered = set(go_f_terms).issubset(pfam_go_terms)
            if is_fully_covered:
                fully_covered_proteins.append(1)
            else:
                fully_covered_proteins.append(0)
        else:
            fully_covered_proteins.append(0)

        if set(go_f_terms).intersection(pfam_go_terms):
            partially_covered_count.append(1)
            if not is_fully_covered:
                partially_covered_proteins.append(protein)
        else:
            partially_covered_count.append(0)
            no_coverage_proteins.append(protein)


    print(f"fully covered proteins: {sum(fully_covered_proteins)} {sum(fully_covered_proteins) / len(fully_covered_proteins) * 100:.2f}")
    print(f"partially covered proteins: {sum(partially_covered_count)} {sum(partially_covered_count) / len(partially_covered_count) * 100:.2f}")

    print(f"partially covered protein example: {partially_covered_proteins[100]}")
    print(f"no coverage protein example: {no_coverage_proteins[100]}")

    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 绘制motif覆盖率分布
    if motif_coverages:
        bins = np.linspace(0, 1, 11)  # 0-1之间分成10个区间
        counts, bin_edges = np.histogram(motif_coverages, bins=bins)
        percentages = counts / len(motif_coverages) * 100
        
        ax1.bar(bin_edges[:-1], percentages, width=0.1, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Motif Coverage')
        ax1.set_ylabel('Percentage of Proteins (%)')
        ax1.set_title('Distribution of Motif Segment Coverage')
        ax1.grid(True, alpha=0.3)
        
        # 在柱子上添加百分比标签
        for i, (count, percentage) in enumerate(zip(counts, percentages)):
            if count > 0:
                ax1.text(bin_edges[i] + 0.05, percentage + 1, f'{percentage:.1f}%', 
                        ha='center', va='bottom', fontsize=9)
    
    # 绘制GO motif覆盖率分布
    if go_motif_coverages:
        bins = np.linspace(0, 1, 11)
        counts, bin_edges = np.histogram(go_motif_coverages, bins=bins)
        percentages = counts / len(go_motif_coverages) * 100
        
        ax2.bar(bin_edges[:-1], percentages, width=0.1, alpha=0.7, 
                edgecolor='black', color='orange')
        ax2.set_xlabel('GO Motif Coverage')
        ax2.set_ylabel('Percentage of Proteins (%)')
        ax2.set_title('Distribution of GO-annotated PFAM Motif Coverage')
        ax2.grid(True, alpha=0.3)
        
        # 在柱子上添加百分比标签
        for i, (count, percentage) in enumerate(zip(counts, percentages)):
            if count > 0:
                ax2.text(bin_edges[i] + 0.05, percentage + 1, f'{percentage:.1f}%', 
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()

    plt.savefig('motif_coverage_distribution.png')
    
    # 返回统计数据
    return {
        'motif_coverages': motif_coverages,
        'go_motif_coverages': go_motif_coverages,
        'total_proteins': len(data),
        'proteins_with_motif': len(motif_coverages),
        'proteins_with_go_motif': len(go_motif_coverages)
    }


def extract_motif_info(train_data, max_motif_length=-1):
    motif_dict = {}
    
    for i, protein_data in enumerate(train_data):
        # 获取蛋白质基本信息
        uniprot_id = protein_data.get('uniprot_id', f'protein_{i}')
        aa_seq = protein_data.get('aa_seq', '')
        struct_seq = protein_data.get('struct_seq', '')
        struct_seq = struct_seq.split(',')
        
        # 获取motif信息
        # motif_position_s = protein_data.get('motif_position_s', [])
        # motif_position_e = protein_data.get('motif_position_e', [])
        motif_desc_list = protein_data.get('motif_desc', [])

        motif_position_s = []
        motif_position_e = []

        for motif_desc in motif_desc_list:
            for motif in protein_data.get('motif', []):
                if motif['go_term'] == motif_desc:
                    motif_position_s.append(motif['start'])
                    motif_position_e.append(motif['end'])
                    break


        
        # 确保所有列表长度一致
        assert len(motif_position_s) == len(motif_position_e) == len(motif_desc_list)
        
        for j in range(len(motif_position_s)):
            # 获取motif信息
            start = motif_position_s[j]
            end = motif_position_e[j]
            motif_desc = motif_desc_list[j]

            if max_motif_length > 0 and end - start + 1 > max_motif_length:
                continue
            
            # 提取对应的氨基酸序列和结构序列片段
            aa_motif = aa_seq[start:end+1]  # 转换为0-based索引
            struct_motif = struct_seq[start:end+1]
            
            # 创建motif信息字典
            motif_info = {
                'protein_name': uniprot_id,
                'aa_sequence': aa_motif,
                'struct_sequence': struct_motif,
                'start_position': start,
                'end_position': end,
                'motif_length': end - start + 1
            }
            
            # 按motif描述分类存储
            if motif_desc not in motif_dict:
                motif_dict[motif_desc] = []
            
            motif_dict[motif_desc].append(motif_info)
    
    return motif_dict


def validate_go_mapping(data_list):
    """
    验证go_f_mapped中的GO编号是否与motif中的GO描述对应
    
    Args:
        data_list: 包含所有训练数据的列表
    
    Returns:
        dict: 包含验证结果的字典
    """
    go_mapping = {}
    mismatches = []
    
    for i, data in enumerate(data_list):
        # 获取go_f_mapped中的GO编号
        go_numbers = data.get('go_f_mapped', [])
        # 获取motif中的GO描述
        motif_go_descs = []
        for motif_item in data.get('motif', []):
            if 'go_term' in motif_item:
                motif_go_descs.append(motif_item['go_term'])
        
        # 验证对应关系
        if len(go_numbers) == len(motif_go_descs):
            for go_num, go_desc in zip(go_numbers, motif_go_descs):
                if go_num not in go_mapping:
                    go_mapping[go_num] = go_desc
                elif go_mapping[go_num] != go_desc:
                    mismatches.append({
                        'index': i,
                        'go_number': go_num,
                        'expected_desc': go_mapping[go_num],
                        'actual_desc': go_desc
                    })
                    exit()
        else:
            # 数量不匹配的情况
            mismatches.append({
                'index': i,
                'go_numbers': go_numbers,
                'motif_descs': motif_go_descs,
                'issue': '数量不匹配'
            })
            # print(data)
            # exit()
    return {
        'go_mapping': go_mapping,
        'mismatches': mismatches,
        'total_mappings': len(go_mapping),
        'total_mismatches': len(mismatches)
    }

def fetch_go_term_info(go_id):
    """
    从QuickGO API获取GO term的详细信息
    """
    url = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{go_id}/complete"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if data.get('numberOfHits', 0) > 0 and data.get('results'):
            result = data['results'][0]
            return {
                'id': result.get('id'),
                'name': result.get('name'),
                'definition': result.get('definition', {}).get('text', ''),
                'aspect': result.get('aspect')
            }
    except Exception as e:
        print(f"获取 {go_id} 信息失败: {e}")
    return None

def build_go_mapping_from_api(data_list):
    """
    通过API建立GO term编号与名称的映射关系
    """
    # 收集所有唯一的GO term编号
    all_go_terms = set()
    for data in data_list:
        go_f_terms = data.get('go_numbers', {}).get('F', [])
        all_go_terms.update(go_f_terms)
    
    print(f"发现 {len(all_go_terms)} 个唯一的GO term编号")
    
    # 通过API获取每个GO term的信息
    go_mapping = {}
    for i, go_id in enumerate(all_go_terms):
        print(f"获取 {go_id} 的信息 ({i+1}/{len(all_go_terms)})")
        info = fetch_go_term_info(go_id)
        if info:
            go_mapping[go_id] = info['name']
        time.sleep(0.1)  # 避免请求过于频繁
    
    return go_mapping


def build_go_mappings(data_list, save_path=None):
    """
    建立并保存两个映射字典：
    1. GO_term到go_f_mapped_id的映射
    2. motif_desc到go_f_mapped_id的映射
    
    Args:
        data_list: 训练数据列表
        save_path: 保存路径（可选）
    
    Returns:
        tuple: (go_term_to_id_map, motif_desc_to_id_map)
    """
    # 初始化映射字典

    print("开始建立GO term到ID的映射...")
    
    go_term2desc_dict = build_go_mapping_from_api(data_list)
    
    # 建立motif_desc到go_f_mapped_id的映射
    go_term2go_id_dict = {}
    for data in data_list:
        go_f_mapped = data.get('go_f_mapped', [])
        go_term = data.get('go_numbers', {})['F']

        for id, term in zip(go_f_mapped, go_term):
            if term not in go_term2go_id_dict:
                go_term2go_id_dict[term] = id
            else:
                if go_term2go_id_dict[term] != id:
                    print(f"GO term {term} 映射到两个不同ID: {go_term2go_id_dict[term]} 和 {id}")
                    exit()
    
    motif_desc2go_id_dict = {}
    for go_term, desc in go_term2desc_dict.items():
        motif_desc2go_id_dict[desc] = go_term2go_id_dict.get(go_term, None)
    
    print(f"建立 {len(go_term2go_id_dict)} 个GO term到ID的映射")
    print(f"建立 {len(motif_desc2go_id_dict)} 个Motif描述到ID的映射")
    
    with open(save_path, "wb") as f:
        pickle.dump({
            'go_term_to_id': go_term2go_id_dict,
            'motif_desc_to_id': motif_desc2go_id_dict
        }, f)

    return go_term2go_id_dict, motif_desc2go_id_dict

def count_labels(train_data, key="motif_go_number", top_k=50):
    """
    统计整个数据集中每个label的出现次数
    """
    all_labels = []
    for item in train_data:
        all_labels.extend(item[key])  # 每个样本可能有多个label
    
    label_counter = Counter(all_labels)
    total_labels = sum(label_counter.values())

    print(f"总共有 {len(label_counter)} 个不同的 label，累计出现 {total_labels} 次。")
    print(f"出现次数最多的前 {top_k} 个 label：\n")
    for label, count in label_counter.most_common(top_k):
        print(f"{label:<15} 出现 {count:>6} 次")

    print(f"出现最少的前 {top_k} 个 label：\n")
    for label, count in label_counter.most_common()[:-top_k-1:-1]:
        print(f"{label:<15} 出现 {count:>6} 次")
    
    return label_counter

def analyze_positive_ratio(train_data, batch_size=6, key="motif_go_number"):
    """
    统计每个batch里可构成正例的label比例
    """
    n_batches = math.ceil(len(train_data) / batch_size)
    total_labels = 0
    total_pos_labels = 0

    random.shuffle(train_data)

    for i in range(n_batches):
        batch = train_data[i * batch_size : (i + 1) * batch_size]
        # 收集所有label
        labels = []
        for item in batch:
            labels.extend(item[key])  # 每个样本可能有多个label
        count = Counter(labels)
        # 能形成正例的label：出现次数≥2
        pos_labels = [l for l, c in count.items() if c >= 2]
        
        total_labels += len(count)
        total_pos_labels += len(pos_labels)

        # print(f"Batch {i+1}:")
        # print(f"  总label数: {len(count)}")
        # print(f"  可形成正例的label数: {len(pos_labels)}")
        # print(f"  比例: {len(pos_labels)/len(count):.2f}")
        # print(f"  正例label示例: {pos_labels[:5]}")
        # print("")

    print("==== 全局统计 ====")
    print(f"总batch数: {n_batches}")
    print(f"平均每batch正例label比例: {total_pos_labels / total_labels:.2f}")

if __name__ == '__main__':

    # path = "data-bin/uniprotKB/cfpgen_general_dataset/train_data_motif_emb_<200.pkl"
    # path = "data-bin/uniprotKB/cfpgen_general_dataset/train.pkl"
    # path = "data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added.pkl"
    # path = "data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif.pkl"

    # path = "data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl"

    # with open(path, "rb") as f:
    #     train_data = pickle.load(f)


    # stats = plot_coverage_distribution(train_data)  # 替换为您的完整数据集

    # print(f"总蛋白质数量: {stats['total_proteins']}")
    # print(f"包含motif的蛋白质数量: {stats['proteins_with_motif']}")
    # print(f"包含GO注释motif的蛋白质数量: {stats['proteins_with_go_motif']}")

    # # print(train_data[0])
    # # exit()
    # # function_name = "GO:0003723"
    # function_name = "GO:0000034"
    # function_name = "GO:0046789"
    # function_name = "GO:0039660"

    # match_pdb = []

    # random.seed(42)

    # # for pdb in train_data:
    # #     if function_name in pdb["go_numbers"]['F']:
    # #         match_pdb.append(pdb)

    # for pdb in train_data:
    #     if pdb.get("pfam_emb", None) is not None:
    #         match_pdb.append(pdb)

    # random_samples = random.sample(match_pdb, min(10, len(match_pdb)))
    # for i, sample in enumerate(random_samples):
    #     print(f"样本 {i+1}: {sample}")
    # print("============================")
    # print(train_data[0])

# 
    # train_data = load_all_pfam_emb_data("train")
    # valid_data = load_all_pfam_emb_data("valid")
    test_data = load_all_pfam_emb_data("test")
    
    # for data in train_data:
    for data in test_data:
        # if data['uniprot_id'] == 'Q48KZ8':
        #     # print(data)
        #     print(data['motif'])
        #     exit()
        # print(data)
        # exit()
        motifs = data['motif']
        # print(data)
        # exit()

        # for motif in motifs:
        #     if motif['go_term'] == 'methylenetetrahydrofolate dehydrogenase (NADP+) activity':
        #         # if motif['end'] - motif['start'] < 50:
        #         print(f"protein: {data['uniprot_id'], data['motif']}")
        #         break

        # if 27 in data['go_f_mapped']:
        #     print(data['motif'])
        #     continue
        # if 4 in data['go_f_mapped']:
        #     print(data['motif'])
        #     continue
        # oxidoreductase activity / identical protein binding
        for motif in motifs:
            # if motif['go_term'] == 'mannitol-1-phosphate 5-dehydrogenase activity' or motif['go_term'] == 'NAD binding':
            # if motif['go_term'] == 'methylenetetrahydrofolate dehydrogenase (NADP+) activity':
            if len(motif['motif_segment']) >= 40 and len(motif['motif_segment']) <= 60:
                # if motif['end'] - motif['start'] < 50:
                print(f"protein: {data['uniprot_id'], data['motif']}")
                break
        #     if motif['go_term'] == 'obsolete 2,3-bisphosphoglycerate-dependent phosphoglycerate mutase activity':
        #         print(f"protein: {data['uniprot_id'], data['motif']}")
        #         break

        # for go_number in data['go_numbers'] ['F']:
        #     if go_number == "GO:0008176":
        #         print(data)

    exit()




    # print(f"训练集样本数: {len(train_data)}")
    # print(f"验证集样本数: {len(valid_data)}")
    # print(f"测试集样本数: {len(test_data)}")

    # all_data = train_data + valid_data + test_data
    # for i in range(len(train_data)):
    #     if train_data[i].get("motif_position_s", None) is not None:
    #         print(train_data[i])
    #         exit()
    # print(train_data[1])

    # motif_dict = extract_motif_info(train_data, max_motif_length=50)

    # sorted_top_motifs = sorted(motif_dict.items(), key=lambda x: len(x[1]), reverse=True)

    # print("\n按频率排序:")
    # for i, (motif_desc, motifs) in enumerate(sorted_top_motifs, 1):
    #     # avg_length = sum(m['motif_length'] for m in motifs) / len(motifs)
    #     # print(f"{i:2d}. {motif_desc:<40} 出现{len(motifs):3d}次, 平均长度{avg_length:.1f}")
    #     print(motifs[0:3])
    #     exit()


    # print(motif_dict.keys())
    # print(motif_dict['3-dehydroquinate dehydratase activity'][0:2])

    # print(train_data[0])

    # result = validate_go_mapping(all_data)

    # print(f"\n总共 {result['total_mappings']} 个GO映射")
    # print(f"发现 {result['total_mismatches']} 个不匹配")

    # desc2map_dict = {}
    # for go_num, go_desc in result['go_mapping'].items():
    #     desc2map_dict[go_desc] = go_num
    
    # with open("desc2map_dict_statics.pkl", "wb") as f:
    #     pickle.dump(desc2map_dict, f)

    # build_go_mappings(all_data, save_path="go_id_mapping.pkl")

    # analyze_positive_ratio(train_data, batch_size=6, key="motif_go_number")
    # count_labels(train_data, key="motif_go_number", top_k=20)
