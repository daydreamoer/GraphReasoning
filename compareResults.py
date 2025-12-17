def extract_answer_lines(filepath):
    """
    提取所有包含“模型回答为:”的完整行
    """
    lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if "模型回答为:" in line:
                lines.append(line.strip())
    return lines


def compare_answer_lines(file1, file2):
    lines1 = extract_answer_lines(file1)
    lines2 = extract_answer_lines(file2)

    min_len = min(len(lines1), len(lines2))
    mismatch_found = False

    for i in range(min_len):
        if lines1[i] != lines2[i]:
            print(f"第 {i + 1} 处不一致:")
            print(f"文件1: {lines1[i]}")
            print(f"文件2: {lines2[i]}")
            print("-" * 40)
            mismatch_found = True

    if len(lines1) != len(lines2):
        print(f"\n两个文件中“模型回答为:”行数不同：文件1有 {len(lines1)} 行，文件2有 {len(lines2)} 行")
        mismatch_found = True

    if not mismatch_found:
        print("所有包含“模型回答为:”的行完全一致。")


filePath1 = 'dataset/hitab/output/BGEoutput1000-1100.log'
filePath2 = 'dataset/hitab/output/output1000-1100.log'

# 使用示例（请替换文件路径）
compare_answer_lines(filePath1, filePath2)
