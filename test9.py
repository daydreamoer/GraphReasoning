import requests
import json

def test_vveai_api(api_key):
    url = "https://api.vveai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "qwen-max",
        "messages": [{"role": "user", "content": "Hello, can you say 'API test successful'?"}],
        "temperature": 0.7
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            timeout=10  # 10秒超时
        )

        print(f"HTTP Status Code: {response.status_code}")

        if response.status_code == 200:
            print("API is working! Response:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"API Error: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {type(e).__name__}: {e}")

# 替换成你的API Key
test_vveai_api("sk-9hH8AnLwRzpcLevh1948Ac5487E34cEdB07eB8A68d904468")