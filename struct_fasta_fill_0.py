import sys

def fix_fasta_padding(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            # 如果是 Header 行 (> 开头)，直接写入
            if line.startswith('>'):
                f_out.write(line + '\n')
            else:
                # 如果是数据行，按逗号分割
                parts = line.split(',')
                # 对每个数字进行补零 (zfill(4) 或 {:04d})
                padded_parts = [p.strip().zfill(4) for p in parts if p.strip()]
                # 重新组合并写入
                f_out.write(','.join(padded_parts) + '\n')

    print(f"处理完成! 结果已保存至: {output_file}")

# 使用示例
# 将您的文件路径替换在这里
input_path = '/AIRvePFS/dair/chenxr-data/repo/cfpgen/generation-results-dplm2-goonly-struct/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_struct_only_go-ipr-500iter-repeat_cut.fasta'  
output_path = '/AIRvePFS/dair/chenxr-data/repo/cfpgen/generation-results-dplm2-goonly-struct/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_struct_only_go-ipr-500iter-repeat_cut_fix.fasta'

# 或者直接在运行时调用逻辑 (如果你直接运行脚本)
if __name__ == "__main__":
    # 如果通过命令行传参: python fix_fasta.py input.fasta output.fasta
    # if len(sys.argv) >= 3:
    #     fix_fasta_padding(sys.argv[1], sys.argv[2])
    # else:
    #     # 如果没有传参，使用硬编码路径或提示
    #     print("请在代码中设置文件路径或使用命令行参数: python script.py <input> <output>")

    fix_fasta_padding(input_path, output_path)