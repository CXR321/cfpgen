import csv
import os

# 定义输入和输出文件名
input_csv_file = 'final_design_result_random.csv'
# 输出文件自动命名为 .fasta 后缀
output_fasta_file = 'generation-pinal/final_design_result_random.fasta'

def convert_csv_to_fasta(input_path, output_path):
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误: 找不到文件 {input_path}，请确保文件在当前目录下。")
        return

    try:
        count = 0
        with open(input_path, mode='r', encoding='utf-8') as csv_file:
            # 使用 DictReader 自动处理带引号的复杂CSV格式
            reader = csv.DictReader(csv_file)
            
            with open(output_path, mode='w', encoding='utf-8') as fasta_file:
                for row in reader:
                    # 1. 获取 ID
                    seq_id = row['id'].strip()
                    
                    # 2. 获取序列 (列名变更为 best_sequence)
                    sequence = row['best_sequence'].strip()
                    
                    # 3. 直接计算序列的实际长度 (忽略 target_length)
                    real_length = len(sequence)
                    
                    # 4. 构造 FASTA header
                    # 格式: >SEQUENCE_ID={id}_L={real_length}
                    header = f">SEQUENCE_ID={seq_id}_L={real_length}"
                    
                    # 5. 写入文件
                    fasta_file.write(f"{header}\n")
                    fasta_file.write(f"{sequence}\n")
                    
                    count += 1
                    
        print(f"转换成功！共处理了 {count} 条序列。")
        print(f"结果已保存至: {output_path}")

    except KeyError as e:
        print(f"错误: CSV文件中缺少列 {e}。")
        print("请检查 CSV 表头是否包含 'id' 和 'best_sequence'。")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    convert_csv_to_fasta(input_csv_file, output_fasta_file)