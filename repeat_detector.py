import re
from collections import defaultdict
import sys

def analyze_repeats(sequence, min_repeat_length=5):
    """
    检测序列中的氨基酸重复模式
    """
    repeat_patterns = []
    
    # 检测单一氨基酸重复 (如: PPPPPP, GGGGGG)
    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        pattern = f"{aa}{{3,}}"
        matches = re.finditer(pattern, sequence)
        for match in matches:
            if len(match.group()) >= min_repeat_length:
                repeat_patterns.append({
                    'type': 'single_aa_repeat',
                    'pattern': match.group(),
                    'position': match.start(),
                    'length': len(match.group()),
                    'amino_acid': aa
                })
    
    # 检测短肽重复 (如: PGPGPG, GGGPGGG)
    short_peptide_patterns = [
        r'(.{2,4})\1{2,}',  # 2-4个氨基酸的重复
    ]
    
    for pattern in short_peptide_patterns:
        matches = re.finditer(pattern, sequence)
        for match in matches:
            repeat_unit = match.group(1)
            full_repeat = match.group()
            if len(full_repeat) >= min_repeat_length:
                repeat_patterns.append({
                    'type': 'peptide_repeat',
                    'pattern': full_repeat,
                    'position': match.start(),
                    'length': len(full_repeat),
                    'repeat_unit': repeat_unit,
                    'repeat_count': len(full_repeat) // len(repeat_unit)
                })
    
    return repeat_patterns

def process_fasta_file(filename, min_repeat_length=30, min_total_repeat_length=1):
    """
    处理整个FASTA文件，检测含有重复序列的蛋白质
    """
    proteins_with_repeats = []
    current_header = ""
    current_sequence = ""
    
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('>'):
                # 处理前一个序列
                if current_sequence:
                    repeats = analyze_repeats(current_sequence, min_repeat_length)
                    if repeats:
                        total_repeat_length = sum(repeat['length'] for repeat in repeats)
                        if total_repeat_length >= min_total_repeat_length:
                            proteins_with_repeats.append({
                                'header': current_header,
                                'sequence_length': len(current_sequence),
                                'repeats': repeats,
                                'total_repeat_length': total_repeat_length,
                                'repeat_percentage': (total_repeat_length / len(current_sequence)) * 100
                            })
                
                # 开始新序列
                current_header = line.strip()[1:]
                current_sequence = ""
            else:
                current_sequence += line.strip()
        
        # 处理最后一个序列
        if current_sequence:
            repeats = analyze_repeats(current_sequence, min_repeat_length)
            if repeats:
                total_repeat_length = sum(repeat['length'] for repeat in repeats)
                if total_repeat_length >= min_total_repeat_length:
                    proteins_with_repeats.append({
                        'header': current_header,
                        'sequence_length': len(current_sequence),
                        'repeats': repeats,
                        'total_repeat_length': total_repeat_length,
                        'repeat_percentage': (total_repeat_length / len(current_sequence)) * 100
                    })
    
    return proteins_with_repeats

def main():
    if len(sys.argv) != 2:
        print("使用方法: python repeat_detector.py <fasta文件>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    print(f"分析文件: {filename}")
    print("检测含有重复序列的蛋白质...")
    
    results = process_fasta_file(filename)
    
    print(f"\n检测结果:")
    print(f"总蛋白质数量: 需要从文件中统计")
    print(f"含有显著重复序列的蛋白质数量: {len(results)}")
    # print(f"\n详细信息:")
    
    # for i, protein in enumerate(results, 1):
    #     print(f"\n{i}. {protein['header']}")
    #     print(f"   序列长度: {protein['sequence_length']}")
    #     print(f"   重复序列总长度: {protein['total_repeat_length']}")
    #     print(f"   重复比例: {protein['repeat_percentage']:.2f}%")
    #     print(f"   重复模式:")
        
    #     for repeat in protein['repeats']:
    #         if repeat['type'] == 'single_aa_repeat':
    #             print(f"     - {repeat['amino_acid']}重复: {repeat['length']}个氨基酸 (位置 {repeat['position']})")
    #         else:
    #             print(f"     - 肽段重复: '{repeat['repeat_unit']}' × {repeat['repeat_count']}")
    #             print(f"       全长: {repeat['length']}氨基酸 (位置 {repeat['position']})")

if __name__ == "__main__":
    main()