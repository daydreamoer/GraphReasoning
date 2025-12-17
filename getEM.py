import re
import argparse  # 导入 argparse 模块
from compute_score import eval_ex_match  # 确保 compute_score.py 与此脚本在同一目录


# # 计算第1条到第100条数据的准确率
# python getEM.py --start 1 --end 100
#
# # 计算第101条到第200条数据的准确率
# python getEM.py --start 101 --end 200
#
# # 你也可以指定不同的日志文件
# python getEM.py --start 1 --end 50 --log_path "path/to/your/other.log"
def extract_predictions_and_answers(log_path):
    """
    从日志中提取“模型回答为:” 和 “答案为:” 后的内容
    (此函数保持不变)
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

def compute_accuracy(pairs):
    """
    计算给定数据对的准确率
    (此函数保持不变)
    """
    correct = 0
    total = len(pairs)
    for pred, gold in pairs:
        if eval_ex_match(pred, gold) == 1:
            correct += 1
    accuracy = correct / total if total > 0 else 0.0
    return correct, total, accuracy

def main():
    # --- 新增：设置命令行参数解析 ---
    parser = argparse.ArgumentParser(description="计算指定范围内日志的EM准确率")
    parser.add_argument('--start', type=int, required=True, help="开始的数据索引 (从1开始)")
    parser.add_argument('--end', type=int, required=True, help="结束的数据索引 (包含)")
    parser.add_argument(
        '--log_path',
        type=str,
        default="dataset/hitab/output/20251114_BGE_qwen25Instruct72b_output0-1500.log",
        help="日志文件路径"
    )

    args = parser.parse_args()

    log_path = args.log_path
    start = args.start
    end = args.end

    # --- 新增：参数校验 ---
    if start < 1:
        print(f"错误: 'start' ( {start} ) 必须 >= 1")
        return
    if end < start:
        print(f"错误: 'end' ( {end} ) 必须 >= 'start' ( {start} )")
        return

    # 1. 提取所有数据
    all_pairs = extract_predictions_and_answers(log_path)
    total_extracted = len(all_pairs)
    print(f"从日志 [ {log_path} ] 中总共提取到 {total_extracted} 条数据。")

    if total_extracted == 0:
        print("未提取到数据，退出。")
        return

    # --- 新增：根据 start 和 end 对数据进行切片 ---
    # 用户输入的 start=1, end=100 对应 Python 索引 all_pairs[0:100]
    slice_start = start - 1
    slice_end = end

    if slice_start >= total_extracted:
        print(f"错误: 'start' ( {start} ) 超出了总数据条数 ( {total_extracted} )。")
        return

    # 如果 end 超出范围，自动调整为到末尾
    if slice_end > total_extracted:
        print(f"警告: 'end' ({end}) 超出了总数据条数 ({total_extracted})。将计算到数据末尾。")
        slice_end = total_extracted
        end = total_extracted # 更新 end 的值以便打印

    # 2. 获取切片后的数据
    sliced_pairs = all_pairs[slice_start:slice_end]

    # 3. 仅计算切片数据的准确率
    correct, total, accuracy = compute_accuracy(sliced_pairs)

    # 4. 打印结果
    print(f"--- 计算范围: 第 {start} 条 到 第 {end} 条 ---")
    print(f"总样本数 (该范围内): {total}")
    print(f"匹配正确数 (该范围内): {correct}")
    print(f"准确率 (该范围内): {accuracy:.2%}")

if __name__ == "__main__":
    main()