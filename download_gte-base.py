# from transformers import AutoModel, AutoTokenizer
#
# # 加载模型和分词器
# model = AutoModel.from_pretrained("thenlper/gte-base")
# tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base")
#
# # 可选：指定本地路径
# model.save_pretrained('./GraphOTTER/gte-base')
# tokenizer.save_pretrained('./GraphOTTER/gte-base')
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('thenlper/gte-base')
model.save('/home/amax/public/lyh/GraphOTTER/gte-base')  # 保存位置
