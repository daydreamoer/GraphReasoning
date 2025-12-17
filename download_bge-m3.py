from sentence_transformers import SentenceTransformer
import os

# 指定模型名称
model_name = "BAAI/bge-m3"

# 指定本地保存路径
save_path = "./bge-m3-sbert"

# 下载模型（含 Transformer 模块 + Pooling 模块）
model = SentenceTransformer(model_name)

# 保存到本地（包含所有 sentence-transformers 所需结构）
model.save(save_path)

print(f"模型已成功下载并保存到: {save_path}")
