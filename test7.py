import re

text = """模型行动步骤为 ```json
[
  {
    "Function": {
      "function_name": "Answer",
      "parameters": []
    },
    "Explanation": "The data is sufficient to calculate the percentage of federally funded academic S
&E obligations provided by HHS, NSF, and DOD in fiscal year 2015. The total amount for all agencies i
n 2015 is $30,494 million, with HHS contributing $17,008 million, NSF contributing $5,295 million, and DOD contributing $3,501 million. I will now compute the required percentage."
    }
  }
]}
```"""

# 1. 提取 ```json [...]``` 之间的内容
json_pattern = r'```json\s*([\s\S]*?)\s*```'
match = re.search(json_pattern, text)
if not match:
    print("未找到JSON部分")
    exit()

json_content = match.group(1)

# 2. 检查括号是否匹配，并修复多余的 }
stack = []
fixed_json = []
for char in json_content:
    if char == '{':
        stack.append(char)
        fixed_json.append(char)
    elif char == '}':
        if stack and stack[-1] == '{':
            stack.pop()
            fixed_json.append(char)
        else:
            # 多余的 }，跳过不添加
            continue
    else:
        fixed_json.append(char)

# 如果栈未清空，说明缺少 `}`
if stack:
    fixed_json.extend(['}'] * len(stack))  # 补全缺少的 `}`

fixed_json_str = ''.join(fixed_json)

# 3. 重新组合文本
fixed_text = re.sub(json_pattern, f'```json\n{fixed_json_str}\n```', text)

print(fixed_text)