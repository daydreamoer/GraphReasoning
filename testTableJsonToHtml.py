import os
import json
import pandas as pd

# 输入和输出路径配置
input_path = "dataset/hitab/raw/491.json"
output_dir = "dataset/hitab/tableDisplay"
output_file = os.path.join(output_dir, "491.html")

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 读取JSON数据
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取文本数据创建DataFrame
texts = data["texts"]
df = pd.DataFrame(texts[1:], columns=texts[0])

# 生成HTML内容（添加基础样式）
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{data.get('title', 'Table Visualization')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; position: sticky; top: 0; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        h2 {{ color: #333; }}
    </style>
</head>
<body>
    <h2>{data.get('title', '')}</h2>
    {df.to_html(index=False, classes='dataframe')}
</body>
</html>
"""

# 写入HTML文件
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"表格可视化已成功保存到: {os.path.abspath(output_file)}")