import re

def fix_json_brackets(text):
    """
    修复文本中JSON部分的括号匹配问题（检查多余的 `}` 或 `{`）

    参数:
        text (str): 包含JSON的原始文本（可能有多余的括号）

    返回:
        str: 修复后的文本，JSON部分括号匹配，其他内容不变
    """
    # 1. 提取 ```json [...]``` 之间的内容
    json_pattern = r'```json\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, text)
    if not match:
        return text  # 如果没有JSON部分，直接返回原文本

    json_content = match.group(1)

    # 2. 检查括号是否匹配，并修复多余的括号
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
                # 多余的 `}`，跳过不添加
                continue
        else:
            fixed_json.append(char)

    # 如果栈未清空，说明缺少 `}`，自动补全
    if stack:
        fixed_json.extend(['}'] * len(stack))

    fixed_json_str = ''.join(fixed_json)

    # 3. 替换原JSON部分，保留其他内容
    fixed_text = re.sub(json_pattern, f'```json\n{fixed_json_str}\n```', text)
    return fixed_text


# ===== 测试用例 =====
if __name__ == "__main__":
    test_text = """模型行动步骤为 ```json
[
  {
    "Function": {
      "function_name": "Answer",
      "parameters": []
    },
    "Explanation": "Some explanation..."
    }
  }
]}
```"""

    print(fix_json_brackets(test_text))