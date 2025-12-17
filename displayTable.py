import pandas as pd
import json
from IPython.display import display, HTML

# 加载JSON数据
with open('data.json') as f:
    data = json.load(f)

# 提取文本数据
texts = data['texts']

# 转换为DataFrame
df = pd.DataFrame(texts[1:], columns=texts[0])

# 处理合并单元格（需安装`openpyxl`）
def apply_merged_regions(df, merged_regions):
    for region in merged_regions:
        first_row, last_row = region['first_row'], region['last_row']
        first_col, last_col = region['first_column'], region['last_column']
        df.iloc[first_row:last_row+1, first_col:last_col+1] = df.iloc[first_row, first_col]

apply_merged_regions(df, data['merged_regions'])

# 美化显示（Jupyter环境）
display(HTML(df.to_html(index=False)))