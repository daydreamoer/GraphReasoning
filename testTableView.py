import json
import pandas as pd
from tabulate import tabulate

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os

def visualize_json_table(json_path):
    # 1. 读取JSON文件
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：文件 {json_path} 不存在")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 {json_path} 不是有效的JSON格式")
        return

    # 2. 提取表格数据
    if 'texts' not in data:
        print("错误：JSON文件中缺少'table_data'字段")
        return

    texts = data['texts']

    # 3. 创建DataFrame
    # 第一行作为列名，其余作为数据
    columns = texts[0] if texts else []
    table_data = texts[1:] if len(texts) > 1 else []

    df = pd.DataFrame(table_data, columns=columns)

    # 4. 可视化设置
    plt.figure(figsize=(14, 8))

    # 如果有标题则添加
    title = data.get('title', '表格可视化')
    plt.suptitle(title, fontsize=14, y=1.02)

    ax = plt.gca()
    ax.axis('off')  # 隐藏坐标轴

    # 5. 创建表格
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     loc='center',
                     cellLoc='center')

    # 6. 样式调整
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # 缩放表格

    # 设置表头样式
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # 表头行
            cell.set_facecolor('#f2f2f2')
            cell.set_text_props(fontweight='bold')
        if col == 0 and row > 0:  # 第一列数据
            cell.set_facecolor('#f8f8f8')

    # 7. 显示图表
    plt.tight_layout()
    plt.show()

    # 8. 同时打印控制台版本
    print("\n控制台表格视图：")
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

# 使用示例
json_path = 'dataset/hitab/raw/491.json'

# 检查路径是否存在
if not os.path.exists(json_path):
    print(f"错误：路径 {json_path} 不存在")
else:
    visualize_json_table(json_path)