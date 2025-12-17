import requests

key = "sk-6a79e9ba99914248ade7e1ee8c393ceb"
url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json"
}
data = {
    "model": "qwen-turbo",
    "input": {"prompt": "你好，介绍一下阿里云百炼"},
    "parameters": {"temperature": 0.7}
}
resp = requests.post(url, headers=headers, json=data)
print(resp.status_code)
print(resp.text)
