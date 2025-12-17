import os
from transformers import AutoTokenizer, AutoModel

# --- 配置 ---
# 1. Hugging Face 上的模型 ID
model_name = "BAAI/bge-reranker-base"

# 2. 您想保存到的本地目录
save_directory = "./bge-reranker-base"
# ----------------

print(f"开始下载模型: {model_name}")

# 检查并创建目标目录
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
    print(f"已创建目录: {save_directory}")
else:
    print(f"目录已存在: {save_directory}")

try:
    # 步骤 1: 从 Hub 加载模型和分词器 (这会自动下载到缓存)
    print("正在加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("正在加载模型 (这可能需要一些时间)...")
    model = AutoModel.from_pretrained(model_name)

    print("模型加载完毕。")

    # 步骤 2: 将加载的模型和分词器保存到您的指定目录
    print(f"正在将文件保存到 {save_directory} ...")
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)

    print("-" * 50)
    print(f"成功! 模型和分词器已完整下载并保存到:")
    print(f"{os.path.abspath(save_directory)}") # 打印绝对路径以便查看
    print("-" * 50)

except Exception as e:
    print(f"发生错误: {e}")
    print("请检查您的网络连接、磁盘空间或模型名称是否正确。")