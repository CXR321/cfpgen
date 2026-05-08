import os

def split_fasta(input_file, max_seqs=999):
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        return

    # 获取输入文件的前缀和扩展名，用于命名输出文件
    file_prefix, file_ext = os.path.splitext(input_file)
    if not file_ext:
        file_ext = ".fasta"

    file_idx = 1
    seq_count = 0
    out_f = None

    try:
        with open(input_file, 'r') as f:
            for line in f:
                # 遇到 '>' 说明是一条新序列的开始
                if line.startswith(">"):
                    # 如果当前文件已经装满了 999 条序列，就关掉它，准备开新文件
                    if seq_count == max_seqs:
                        out_f.close()
                        file_idx += 1
                        seq_count = 0
                        out_f = None # 重置文件指针
                    
                    seq_count += 1

                    # 如果当前没有打开的文件，就新建一个
                    if out_f is None:
                        out_name = f"{file_prefix}_part{file_idx}{file_ext}"
                        out_f = open(out_name, 'w')
                        print(f"正在生成文件: {out_name} ...")

                # 如果文件指针存在（处理第一行不是 '>' 的异常情况也会被过滤），则写入当前行
                if out_f is not None:
                    out_f.write(line)
                    
    finally:
        # 确保最后一个文件被正确关闭
        if out_f and not out_f.closed:
            out_f.close()
            
    print("FASTA 切割完成！")

# ==========================================
# 使用方法：在这里修改你的输入文件名
# ==========================================
if __name__ == "__main__":
    # 替换为你实际的 fasta 文件路径
    INPUT_FASTA = "./evaluat_fasta/cfpgen_650m_go.fasta" 
    # INPUT_FASTA = "./evaluat_fasta/codefp.fasta" 
    # INPUT_FASTA = "./evaluat_fasta/pinal_baseline_random_selection.fasta" 
    
    # 执行切割，默认每个文件最大 999 条
    split_fasta(INPUT_FASTA, max_seqs=999)