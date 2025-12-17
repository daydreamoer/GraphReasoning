import pandas as pd
import matplotlib.pyplot as plt

# 示例JSON数据
json_data = [
    {"id": 1, "name": "张三", "age": 25, "city": "北京"},
    {"id": 2, "name": "李四", "age": 30, "city": "上海"},
    {"id": 3, "name": "王五", "age": 28, "city": "广州"}
]

# 转换为DataFrame
df = pd.DataFrame(json_data)

# 打印表格
print(df.to_string(index=False))

# 或者使用matplotlib绘制表格
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.show()