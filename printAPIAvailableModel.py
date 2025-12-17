from openai import OpenAI

# 替换为你的 Base URL 和 API Key
client = OpenAI(
    base_url="https://aifast.site/v1",
    api_key="sk-rpN9f01K0gHT4rGCTO0w3asAB1sKLBcukBbnLi4R68qnRH0B"
)

try:
    models = client.models.list()
    print("可用模型列表：")
    for model in models:
        # 打印包含 'qwen' 的模型，方便查找
        if 'qwen' in model.id.lower():
            print(model.id)
except Exception as e:
    print(f"获取模型列表失败: {e}")