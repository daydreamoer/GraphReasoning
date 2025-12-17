from transformers import AutoTokenizer, AutoModel
import torch

# 加载本地模型
model_path = "./bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# 模拟一个文本输入
text = "你好，世界"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# 前向计算（不求梯度）
with torch.no_grad():
    outputs = model(**inputs)

# 获取 CLS 向量（第一位 token 的输出）
embedding = outputs.last_hidden_state[:, 0, :]
print("模型输出向量维度:", embedding.shape)
