import sys

def add_sequence_numbers(input_file, output_file=None):
    """
    为FASTA文件中的序列ID添加序号
    
    参数:
    input_file: 输入FASTA文件路径
    output_file: 输出文件路径（如果为None，则覆盖原文件）
    """
    
    # 读取输入文件
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    modified_lines = []
    sequence_count = 0
    
    for line in lines:
        if line.startswith('>'):
            # 移除换行符，添加序号，再加回换行符
            line = line.rstrip() + f'_{sequence_count}\n'
            sequence_count += 1
        modified_lines.append(line)
    
    # 确定输出文件路径
    if output_file is None:
        output_file = input_file
    
    # 写入输出文件
    with open(output_file, 'w') as f:
        f.writelines(modified_lines)
    
    print(f"处理完成！共处理了 {sequence_count} 个序列")
    print(f"结果已保存到: {output_file}")

if __name__ == "__main__":
    # 使用方法示例
    if len(sys.argv) < 2:
        print("使用方法: python fasta_renumber.py <输入文件> [输出文件]")
        print("示例: python fasta_renumber.py sequences.fasta sequences_renumbered.fasta")
        print("如果只提供输入文件，将覆盖原文件")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        add_sequence_numbers(input_file, output_file)
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{input_file}'")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")