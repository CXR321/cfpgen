import requests
from bs4 import BeautifulSoup
from requests_toolbelt.multipart.encoder import MultipartEncoder
import pickle
from Bio import SeqIO
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm
import numpy as np
from pfam2go import pfam2go  

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((requests.exceptions.Timeout, 
                                 requests.exceptions.ConnectionError)),
    reraise=True
)
def make_request_with_retry(url, data, headers):
    response = requests.post(url, data=data, headers=headers, timeout=30)
    return response


def extract_motif_info(html_content):
    """
    从HTML内容中提取Pfam基序信息
    
    参数:
        html_content: HTML内容字符串
    
    返回:
        包含Pfam信息的列表
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 查找结果表格
    table = soup.find('table', {'class': 'result'})
    if not table:
        return []
    
    # 提取所有行（跳过表头）
    rows = table.find_all('tr')[1:]
    
    motifs = []
    
    for row in rows:
        columns = row.find_all('td')
        if len(columns) < 3:
            continue
            
        # 提取Pfam ID
        pfam_id_elem = columns[0].find('a')
        pfam_id = pfam_id_elem.text if pfam_id_elem else columns[0].text.strip()
        
        # 提取位置和E值
        # position_elem = columns[1].find('nobr')
        # position = position_elem.text if position_elem else columns[1].text.strip()

        nobr_elements = columns[1].find_all('nobr')
        positions = []
        for nobr_elem in nobr_elements:
            if nobr_elem.text:
                if '(' in nobr_elem.text and ')' in nobr_elem.text:
                    positions.append(nobr_elem.text.strip())
        
        # 提取描述
        description = columns[4].text
        
        for position in positions:
            motifs.append({
                'pfam_name': pfam_id,
                'pfam_id': description.split(',')[0],
                'position': position,
                'description': description
            })
    
    return motifs

def read_fasta_biopython(filename):
    """
    使用Biopython读取FASTA文件
    
    参数:
        filename: FASTA文件路径
    
    返回:
        序列字典 {序列ID: 序列}
    """
    sequences = {}
    for record in SeqIO.parse(filename, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences

failed_id = []

def search_and_extract_motifs(sequence="MDLNPSTFVLEIVNFLVLVWLLKRFLYQPVSAAIEERRRQIARTVAEARDTQTAAETLRMQYESRLADWESEKRQAREAFKQEIEAERQRALDELEKALDAEREKARVLIERQRRDMESDLERQALRLSRQFASRFLERLAGPEMEAALLRMFGEDLAAMSPEQWQALTRALEEQEHPEAEIASAFPLKPESCAELTEMIEARTGRAVAWRFREDPALICGIRLRAGHRVLAANVGEELKFFADGAENSLGGG"):
    """
    自动搜索基序并提取结果
    
    参数:
        sequence: 要搜索的氨基酸序列
    
    返回:
        提取的基序信息列表
    """
    # 发送请求
    url = "https://www.genome.jp/tools-bin/search_motif_lib"
    
    multipart_data = MultipartEncoder(
        fields={
            'seqid': '',
            'upload_file': ('', '', 'application/octet-stream'),
            'seq': sequence,
            'pfam': 'on',
            'pfam_cuteval': '1.0',
            'cdd_filter': '',
            'cdd_cuteval': '1.0',
            'skip_entry': 'on',
            'skip_unspecific_profile': 'on',
            'user_cutscore': '',
            'profile_file': ('', '', 'application/octet-stream'),
            'FORMAT': 'PROSITE'
        },
        boundary='----WebKitFormBoundaryWXBETVtOPaGNbjfv'
    )
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Content-Type': multipart_data.content_type,
        'Referer': 'https://www.genome.jp/tools/motif/'
    }
    
    try:
        response = make_request_with_retry(url, multipart_data, headers)
        
        if response.status_code == 200:
            # 提取信息
            motifs = extract_motif_info(response.text)
            return motifs
        else:
            print(f"请求失败，状态码: {response.status_code}")
            return ["bad"]
            
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return ["bad"]

def search_fasta_motifs(fasta_file_path: str):
    sequences = read_fasta_biopython(fasta_file_path)

    pred_seq = {}
    pred_oriname_seq = {}

    for seq_id, sequence in tqdm(sequences.items()):
        # print(f"ID: {seq_id}")
        # print(f"Sequence: {sequence[:50]}...")  # 只显示前50个字符
        # print(f"Length: {len(sequence)}")
        # print("-" * 50)
        # exit()
        

        motifs = search_and_extract_motifs(sequence)
        pred_oriname_seq[seq_id] = motifs

        seq_id = seq_id.split('_L=')[0] if '_L=' in seq_id else seq_id
        pred_seq[seq_id] = motifs

    with open(fasta_file_path.replace(".fasta", "_motifs.pkl"), "wb") as f:
        pickle.dump(pred_seq, f)

    with open(fasta_file_path.replace(".fasta", "_motifs.txt"), "w") as f:
        for seq_id, motifs in pred_oriname_seq.items():
            f.write(f"{seq_id}\n{motifs}\n")

def parse_position_and_evalue(position_str):
    """
    解析position字符串，返回起始位置、结束位置、长度和e-value
    格式示例: '39..286(8.9e-29)' -> (39, 286, 247, 8.9e-29)
    """
    try:
        # 提取位置部分和e-value部分
        if '(' in position_str and ')' in position_str:
            pos_part = position_str.split('(')[0].strip()
            evalue_part = position_str.split('(')[1].split(')')[0].strip()
            
            start, end = map(int, pos_part.split('..'))
            length = end - start
            
            # 解析e-value
            if 'e-' in evalue_part:
                base, exponent = evalue_part.split('e-')
                evalue = float(base) * (10 ** -float(exponent))
            else:
                evalue = float(evalue_part)
                
            return start, end, length, evalue
        else:
            # 如果没有括号，尝试直接解析位置
            if '..' in position_str:
                start, end = map(int, position_str.split('..'))
                length = end - start
                return start, end, length, None
            else:
                return None, None, None, None
    except (ValueError, IndexError, AttributeError):
        return None, None, None, None

def calculate_motif_hit_rate_with_evalue_threshold(test_motifs, compare_motifs, evalue_threshold=0.05):
    """
    计算每个蛋白质的motif匹配率，考虑e-value阈值，并统计motif长度分布
    """
    results = {
        'total_proteins': 0,
        'hit_0': 0,
        'hit_1': 0,
        'hit_2': 0,
        'hit_3_or_more': 0,
        'individual_hit_rates': [],
        'average_hit_rate': 0.0,
        'hit_lengths': [],        # 命中的motif长度
        'missed_lengths': [],     # 未命中的motif长度
        'hit_evalues': [],        # 命中的motif的e-value
        'missed_evalues': [],     # 未命中的motif的e-value
        'detailed_results': {},
        'evalue_threshold': evalue_threshold,
        'hit_0_names': [],
        'hit_1_names': [],
    }

    with open("data-bin/uniprotKB/cfpgen_general_dataset/test_data_motif_emb.pkl", "rb") as f:
        original_test_data = pickle.load(f)

    # print(original_test_data[0])
    # print(original_test_data[100])

    # 创建从protein_id到GO标签的映射
    go_mapping = {}
    motif_go_mapping = {}
    for protein_data in original_test_data:
        if 'uniprot_id' in protein_data:
            protein_id = "SEQUENCE_ID=" + protein_data['uniprot_id']
            go_mapping[protein_id] = protein_data.get('go_numbers', {})
            motif_go_mapping[protein_id] = protein_data['motif']

    for protein_id, test_motif_list in test_motifs.items():
        protein_id = "SEQUENCE_ID=" + protein_id
        if protein_id not in compare_motifs:
            # 如果比较数据中没有该蛋白质，跳过
            continue
            
        results['total_proteins'] += 1
        

        
        # 获取比较数据中满足e-value阈值的pfam_id集合
        valid_compare_motifs = []
        for motif in compare_motifs[protein_id]:
            _, _, _, evalue = parse_position_and_evalue(motif['position'])
            if evalue is not None and evalue < evalue_threshold:
                valid_compare_motifs.append(motif['pfam_id'])

        valid_test_motifs = []
        for motif in test_motif_list:
            _, _, _, evalue = parse_position_and_evalue(motif['position'])
            if evalue is not None and evalue < evalue_threshold:
                valid_test_motifs.append(motif)

        test_motif_list = valid_test_motifs
        # 获取test motifs的pfam_id集合
        test_pfam_ids = {motif['pfam_id'] for motif in test_motif_list}  

        compare_pfam_ids = set(valid_compare_motifs)
        
        # 计算匹配的motif数量（pfam_id相同且e-value满足阈值）
        matched_count = len(test_pfam_ids.intersection(compare_pfam_ids))
        
        # 计算该蛋白质的匹配率
        hit_rate = matched_count / len(test_pfam_ids) if test_pfam_ids else 0
        results['individual_hit_rates'].append(hit_rate)

        go_tags = go_mapping.get(protein_id, {})

        protein_info = {
            'name': protein_id, 
            # 'test_motifs': test_pfam_ids, 
            'test_motifs': test_motif_list,
            'compare_motifs': compare_pfam_ids,
            'go_tags': go_tags,  # 添加GO标签
            'go_motif': motif_go_mapping.get(protein_id, [])
        }

        # 统计命中情况
        if matched_count == 0:
            results['hit_0'] += 1
            results['hit_0_names'].append(protein_info)
        elif matched_count == 1:
            results['hit_1'] += 1
            if len(test_pfam_ids) > 1:
                results['hit_1_names'].append(protein_info)
        elif matched_count == 2:
            results['hit_2'] += 1
        else:
            results['hit_3_or_more'] += 1
        
        # 统计命中和未命中的motif长度及e-value
        for test_motif in test_motif_list:
            start, end, length, evalue = parse_position_and_evalue(test_motif['position'])
            if length is not None:
                # 检查是否匹配（pfam_id相同且比较数据中有满足e-value阈值的相同motif）
                is_hit = (test_motif['pfam_id'] in compare_pfam_ids)
                
                if is_hit:
                    results['hit_lengths'].append(length)
                    if evalue is not None:
                        results['hit_evalues'].append(evalue)
                else:
                    results['missed_lengths'].append(length)
                    if evalue is not None:
                        results['missed_evalues'].append(evalue)
            
        # 保存详细结果
        results['detailed_results'][protein_id] = {
            'matched_count': matched_count,
            'total_test_motifs': len(test_pfam_ids),
            'hit_rate': hit_rate,
            'test_motifs': test_pfam_ids,
            'compare_motifs': compare_pfam_ids,
            'matched_motifs': test_pfam_ids.intersection(compare_pfam_ids),
            'valid_compare_count': len(compare_pfam_ids)  # 满足e-value阈值的比较motif数量
        }
    
    # 计算平均命中率
    if results['individual_hit_rates']:
        results['average_hit_rate'] = sum(results['individual_hit_rates']) / len(results['individual_hit_rates'])
    
    return results

def print_statistics(results, dataset_name):
    """
    打印统计结果
    """
    print(f"\n=== {dataset_name} 统计结果 ===")
    print(f"总蛋白质数量: {results['total_proteins']}")
    print(f"命中0个motif: {results['hit_0']} ({results['hit_0']/results['total_proteins']*100:.1f}%)")
    print(f"命中1个motif: {results['hit_1']} ({results['hit_1']/results['total_proteins']*100:.1f}%)")
    print(f"命中2个motif: {results['hit_2']} ({results['hit_2']/results['total_proteins']*100:.1f}%)")
    print(f"命中3个或以上motif: {results['hit_3_or_more']} ({results['hit_3_or_more']/results['total_proteins']*100:.1f}%)")
    print(f"平均motif命中率: {results['average_hit_rate']*100:.1f}%")

def analyze_length_distribution(lengths, name):
    if lengths:
        print(f"{name}:")
        print(f"  中位数: {np.median(lengths):.1f}")
        print(f"  标准差: {np.std(lengths):.1f}")
        print(f"  25%分位数: {np.percentile(lengths, 25):.1f}")
        print(f"  75%分位数: {np.percentile(lengths, 75):.1f}")

def parse_pfam2go(filename="pfam2go.txt"):
    """
    解析 pfam2go 文件，返回一个映射字典。
    
    Args:
        filename (str): pfam2go 文件的路径
        
    Returns:
        dict: 一个字典，格式为 {pfam_accession: [(go_id, go_term)]}
    """
    pfam2go_map = {}
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            # 跳过注释行和空行
            if line.startswith('!') or not line:
                continue
                
            # 解析一行
            # 示例: Pfam:7tm_1 PF00001 > GO:G protein-coupled receptor activity ; GO:0004930
            parts = line.split(' > ')
            if len(parts) < 2:
                continue
                
            # 提取 Pfam 部分
            pfam_section = parts[0].split()
            pfam_acc = pfam_section[1] # 获取 7tm_1
                
            # 提取 GO 部分
            go_section = parts[1].split(' ; ')
            go_desc = go_section[0][3:]  # 移除前面的 "GO:"，获取描述
            go_id = go_section[1]        # 获取 GO:0004930
            
            # 将映射添加到字典中
            if pfam_acc not in pfam2go_map:
                pfam2go_map[pfam_acc] = []
            pfam2go_map[pfam_acc].append((go_id, go_desc))
            
    return pfam2go_map


def data_build_pfam_motif(file_path):
    with open(file_path, "rb") as f:
        ori_data = pickle.load(f)


    for id, data in tqdm(enumerate(ori_data)):
        pfam_motif = search_and_extract_motifs(data['sequence'])

        go_numbers = data['go_numbers']['F']
        
        try: 
            for motif_id, motif in enumerate(pfam_motif):

                s, e, l, evalue = parse_position_and_evalue(motif['position'])
                pfam_motif[motif_id]['start'] = s
                pfam_motif[motif_id]['end'] = e
                pfam_motif[motif_id]['seq'] = data['sequence'][s-1:e]
                pfam_motif[motif_id]['evalue'] = evalue
                pfam_motif[motif_id]['strong_go_id'] = []
                pfam_motif[motif_id]['strong_go_desc'] = []

                if motif['pfam_name'] in pfam2go_map:
                    for go_id, go_desc in pfam2go_map[motif['pfam_name']]:
                        if go_id in go_numbers:
                            pfam_motif[motif_id]['strong_go_id'].append(go_id) 
                            pfam_motif[motif_id]['strong_go_desc'].append(go_desc)

            ori_data[id]['pfam_motif'] = pfam_motif
            # print(f"pfam_motif in {data['uniprot_id']}: {ori_data[id]['pfam_motif']}")
        except Exception as e:
            print(f"err in {data['uniprot_id']}")
            ori_data[id]['pfam_motif'] = ["bad"]

    with open(file_path.replace(".pkl", "_pfamMotif.pkl"), "wb") as f:
        pickle.dump(ori_data, f)



if __name__ == "__main__":

    # print(search_and_extract_motifs("MVQKFMSRYQAALLGLGLLLVFLLYMGLPGPPEQTSRLWRGPNVTVLTGLTRGNSRIFYREVLPIQQACRAEVVFLHGKAFNSHTWEQLGTLQLLSERGYRAVAIDLPGFGNSAPSEEVSTEAGRVELLERVFQDLQVQNTVLVSPSLSGSYALPFLMQNHHQLRGFVPIAPTYTRNYAQEQFRAVKTPTLILYGELDHTLARESLQQLRHLPNHSMVKLRDAGHACYLHKPEAFHLALLAFLDHLP"))
    # exit()

    pfam2go_map = parse_pfam2go()

    # t = search_and_extract_motifs()
    # print(t)
    # exit()

    # with open("data-bin/uniprotKB/cfpgen_general_dataset/test_data_motif_emb.pkl", "rb") as f:
    #     test_data = pickle.load(f)

    # print(test_data[0])

    # gt_seq = {}

    

    # for item in tqdm(test_data):
    #     seq_id = item['uniprot_id']
    #     sequence = item['sequence']
        
    #     motifs = search_and_extract_motifs(sequence)

    #     gt_seq[seq_id] = motifs

    # with open("data-bin/uniprotKB/cfpgen_general_dataset/test_motifs.pkl", "wb") as f:
    #     pickle.dump(gt_seq, f)

    # exit()
    # path = "generation-results-dplm2-diff-cross-dc-5.5wstep/cfpgen_general_dataset_stage1_dplm2_dm_ca_dc_pow2weight_wandb_go-ipr-500iter-repeat_cut.fasta"
    # path = "generation-results-dplm2-diff-cross-me-drop6-3wstep/cfpgen_general_dataset_stage1_dplm2_dm_ca_me_drop6_wandb_go-ipr-500iter-repeat_cut.fasta"
    # search_fasta_motifs(path)
    # exit()


    # path = "gen_single/generated_sequence_B6HV35.fasta"
    # path = "gen_single/generated_sequence_Q8S339.fasta"
    # path = "gen_single/generated_sequence_A3PN82.fasta"
    # path = "generation-results-dplm2-diff-cross/cfpgen_general_dataset_stage1_dplm2_diff-modulation_func-cross-attn_wandb_go-ipr-500iter-repeat_cut-debug.fasta"
    # search_fasta_motifs(path)
    # exit()


    # data_build_pfam_motif("data-bin/uniprotKB/cfpgen_general_dataset/train_data_motif_emb.pkl")
    # data_build_pfam_motif("data-bin/uniprotKB/cfpgen_general_dataset/valid_data_motif_emb.pkl")
    # data_build_pfam_motif("data-bin/uniprotKB/cfpgen_general_dataset/test_data_motif_emb.pkl")

    data_build_pfam_motif("data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added.pkl")
    data_build_pfam_motif("data-bin/uniprotKB/cfpgen_general_dataset/valid_all_old_motif_added.pkl")
    data_build_pfam_motif("data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added.pkl")


    exit()


    with open(path.replace(".fasta", "_motifs.pkl"), "rb") as f:
        my_model_motifs = pickle.load(f)

    
    path = "generation-results-cfpgen_650m/cfpgen_650m_go-ipr.fasta"
    # search_fasta_motifs(path)
    with open(path.replace(".fasta", "_motifs.pkl"), "rb") as f:
        baseline_motifs = pickle.load(f)

    path = "data-bin/uniprotKB/cfpgen_general_dataset/test_motifs.pkl"
    with open(path, "rb") as f:
        test_motifs = pickle.load(f)

    print(f"my_model_motifs[0]: {my_model_motifs[list(my_model_motifs.keys())[0]]}")
    print(f"baseline_motifs[0]: {baseline_motifs[list(baseline_motifs.keys())[0]]}")
    print(f"test_motifs[0]: {test_motifs[list(test_motifs.keys())[0]]}")

# 计算统计结果
    my_model_stats = calculate_motif_hit_rate_with_evalue_threshold(test_motifs, my_model_motifs)

    with open('my_model_hit_0_names.txt', 'w') as f:
        for item in my_model_stats['hit_0_names']:
            f.write(str(item) + '\n')
    with open('my_model_hit_1_names.txt', 'w') as f:
        for item in my_model_stats['hit_1_names']:
            f.write(str(item) + '\n')
    exit()


    baseline_stats = calculate_motif_hit_rate_with_evalue_threshold(test_motifs, baseline_motifs)

    # 打印结果
    print_statistics(my_model_stats, "My Model")
    print_statistics(baseline_stats, "Baseline")

    analyze_length_distribution(my_model_stats['hit_lengths'], "My Model 命中motif长度")
    analyze_length_distribution(my_model_stats['missed_lengths'], "My Model 未命中motif长度")
    analyze_length_distribution(baseline_stats['hit_lengths'], "Baseline 命中motif长度")
    analyze_length_distribution(baseline_stats['missed_lengths'], "Baseline 未命中motif长度")    

    # # 使用默认序列"MKMK"
    # motifs = search_and_extract_motifs()
    
    # # 使用自定义序列
    # # motifs = search_and_extract_motifs("YOUR_SEQUENCE_HERE")
    
    # if motifs:
    #     print("找到的基序信息:")
    #     print("=" * 80)
    #     for motif in motifs:
    #         print(f"{motif['pfam_id']}\t{motif['position']}\t{motif['description']}")
    # else:
    #     print("未找到基序信息")

    with open("data-bin/uniprotKB/cfpgen_general_dataset/train_data_motif_emb.pkl", "rb") as f:
        train_data = pickle.load(f)
        # print(train_data[0])

    bins = [0, 50, 100, 150, 200, 300, 500, float('inf')]
    bin_labels = ['0-50', '50-100', '100-150', '150-200', '200-300', '300-500', '500+']
    protein_count_by_bin = {label: 0 for label in bin_labels}
    protein_count_by_bin['no_motif'] = 0

    # 提取所有 motif_segment 的长度
    motif_lengths = []
    for item in train_data:
        if 'motif' in item and item['motif']:  # 检查是否存在 motif 字段且不为空
            temp_ls = []
            for motif in item['motif']:
                if 'motif_segment' in motif:
                    motif_lengths.append(len(motif['motif_segment']))
                    temp_ls.append(len(motif['motif_segment']))

            if temp_ls:
                # 找出该蛋白涉及的所有区间
                covered_bins = set()
                for length in temp_ls:
                    for i in range(len(bins) - 1):
                        if bins[i] < length <= bins[i + 1]:
                            covered_bins.add(bin_labels[i])
                            break
                
                # 统计每个涉及的区间
                for bin_label in covered_bins:
                    protein_count_by_bin[bin_label] += 1
            else:
                protein_count_by_bin['no_motif'] += 1
        else:
            protein_count_by_bin['no_motif'] += 1


    # 转换为 numpy 数组
    motif_lengths = np.array(motif_lengths)
    if len(motif_lengths) > 0:
        # 计算分位数
        quantiles = np.quantile(motif_lengths, [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        print(f"25% 分位数: {quantiles[0]}")
        print(f"50% 分位数: {quantiles[1]}")
        print(f"75% 分位数: {quantiles[2]}")
        print(f"90% 分位数: {quantiles[3]}")
        print(f"95% 分位数: {quantiles[4]}")
        print(f"99% 分位数: {quantiles[5]}")
        
        # 统计不同长度区间的数量
        bins = [0, 50, 100, 150, 200, 300, 500, 1000, np.inf]
        hist, _ = np.histogram(motif_lengths, bins=bins)
        print("\n长度区间统计:")
        for i in range(len(hist)):
            if i == len(hist)-1:
                print(f">{bins[i]}: {hist[i]} 个 motif")
            else:
                print(f"{bins[i]}-{bins[i+1]}: {hist[i]} 个 motif")

        print("\n各长度区间的蛋白数量:")
        for bin_label in bin_labels:
            print(f"{bin_label}: {protein_count_by_bin[bin_label]} 个蛋白")