import re
from compute_score import eval_ex_match  # 确保 compute_score.py 与此脚本在同一目录，或设置好导入路径

def extract_predictions_and_answers(log_path):
    """
    从日志中提取“模型回答为:” 和 “答案为:” 后的内容
    """
    pairs = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "模型回答为:" in line and "答案为:" in line:
                try:
                    pred_match = re.search(r"模型回答为:\s*(.*?)\s*\t", line)
                    gold_match = re.search(r"答案为:\s*(.*)", line)

                    if pred_match and gold_match:
                        pred = pred_match.group(1).strip()
                        gold = gold_match.group(1).strip()
                        pairs.append((pred, gold))
                except Exception as e:
                    print(f"解析失败：{e}")
    return pairs

def compute_accuracy(pairs, N=None):
    """
    计算前 N 条数据的准确率；N=None 时计算全部
    """
    if N is not None:
        pairs = pairs[:N]          # 仅取前 N 条
    correct = 0
    total = len(pairs)
    for pred, gold in pairs:
        if eval_ex_match(pred, gold) == 1:
            correct += 1
    accuracy = correct / total if total > 0 else 0.0
    return correct, total, accuracy

def main():
    log_path = "dataset/hitab/output/20251111BGEoutput0-1000.log"
    N = 100                     # <—— 在这里设置你需要的 N
    pairs = extract_predictions_and_answers(log_path)
    correct, total, accuracy = compute_accuracy(pairs, N=N)

    print(f"前 {N} 条样本数: {total}")
    print(f"匹配正确数: {correct}")
    print(f"准确率: {accuracy:.2%}")

if __name__ == "__main__":
    main()
