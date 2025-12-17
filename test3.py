from transformers import AutoTokenizer, AutoModel
import torch

# 本地模型路径
model_path = "./GraphOTTER/gte-base"  # 注意是服务器上该脚本相对路径

# 明确指定只用本地文件
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModel.from_pretrained(model_path, local_files_only=True)

# 测试文本
texts = ["Graph neural networks are powerful.", "Transformers revolutionized NLP."]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0]  # 使用 [CLS] token

for i, text in enumerate(texts):
    print(f"\nText: {text}\nEmbedding norm: {embeddings[i].norm().item():.4f}")
