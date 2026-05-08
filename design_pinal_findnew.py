import pandas as pd
import glob
import os
import numpy as np
from tqdm import tqdm

def find_best_sequence():
    # 1. 文件路径配置
    input_csv = 'final_design_result.csv'
    output_csv = 'final_design_result_random.csv'
    generated_dir = 'curl_generated_proteins'

    # 检查输入文件是否存在
    if not os.path.exists(input_csv):
        print(f"错误: 找不到文件 {input_csv}")
        return

    # 读取主 CSV
    print(f"正在读取 {input_csv}...")
    df = pd.read_csv(input_csv)

    # 准备列表存储更新后的数据
    updated_sequences = []
    updated_scores = []

    # 统计计数器
    count_updated = 0
    count_missing = 0

    # 2. 遍历每一行数据
    for index, row in tqdm(df.iterrows()):
        uid = str(row['id']).strip()
        target_len = int(row['target_length'])
        
        # 获取当前默认的值，以防找不到新文件时回退
        current_best_seq = row['best_sequence']
        current_best_score = row['protrek_score']

        # 3. 在文件夹中查找匹配的文件 (例如: A0A0F7TXA1_*.tsv)
        # 使用 glob 匹配该 ID 开头的所有 .tsv 文件
        search_pattern = os.path.join(generated_dir, f"{uid}_*.tsv")
        found_files = glob.glob(search_pattern)

        candidates = []

        if not found_files:
            # 如果没找到生成的文件，保持原样
            updated_sequences.append(current_best_seq)
            updated_scores.append(current_best_score)
            count_missing += 1
            continue

        # 4. 读取所有找到的文件中的候选序列
        for file_path in found_files:
            try:
                # 读取 TSV 文件，分隔符为制表符
                # 根据你的示例，列名包含空格，pandas 会自动处理
                sub_df = pd.read_csv(file_path, sep='\t')
                
                # 去除列名可能存在的首尾空格
                sub_df.columns = sub_df.columns.str.strip()

                # 检查必要的列是否存在
                if 'Protein Sequence' in sub_df.columns and 'Protrek Score' in sub_df.columns:
                    for _, sub_row in sub_df.iterrows():
                        seq = str(sub_row['Protein Sequence']).strip()
                        score = sub_row['Protrek Score']
                        candidates.append({'seq': seq, 'score': score})
            except Exception as e:
                print(f"  读取文件出错 {file_path}: {e}")

        # 5. 如果有候选序列，进行筛选
        if candidates:
            best_candidate = None
            min_len_diff = float('inf')
            min_pre_score = float('inf')


            # choose length diff min
            # for cand in candidates:
            #     seq_len = len(cand['seq'])
            #     # 计算长度差距 (绝对值)
            #     len_diff = abs(seq_len - target_len)

            #     # 逻辑：找差距最小的
            #     if len_diff < min_len_diff:
            #         min_len_diff = len_diff
            #         best_candidate = cand
            #     elif len_diff == min_len_diff:
            #         # 如果长度差距一样，保留分数更小的那个 (可选优化)
            #         if float(cand['score']) < float(best_candidate['score']):
            #             best_candidate = cand
            
            # updated_sequences.append(best_candidate['seq'])
            # updated_scores.append(best_candidate['score'])
            # count_updated += 1

            # random choice one candidate
            # best_candidate = np.random.choice(candidates)
            # updated_sequences.append(best_candidate['seq'])
            # updated_scores.append(best_candidate['score'])
            # count_updated += 1

            # choose score min
            # for cand in candidates:
            #     if float(cand['score']) < float(min_score): 
            #         min_score = cand['score']
            #         best_candidate = cand
            # updated_sequences.append(best_candidate['seq'])
            # updated_scores.append(best_candidate['score'])
            # count_updated += 1
        else:
            # 文件找到了但没内容或解析失败
            updated_sequences.append(current_best_seq)
            updated_scores.append(current_best_score)

    # 6. 更新 DataFrame 并保存
    df['best_sequence'] = updated_sequences
    df['protrek_score'] = updated_scores

    df.to_csv(output_csv, index=False)
    print("-" * 30)
    print(f"处理完成！")
    print(f"共处理 ID 数: {len(df)}")
    print(f"成功更新序列数: {count_updated}")
    print(f"未找到生成文件数: {count_missing}")
    print(f"结果已保存至: {output_csv}")

if __name__ == "__main__":
    find_best_sequence()