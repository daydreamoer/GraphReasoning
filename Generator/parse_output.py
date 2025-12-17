import ast
import os
import re
import json

def split_checks(input_string):
    # pattern = r'[\w]+\(.*?\)'
    pattern = r'[\w]+\[.*?\]'
    # Use re.findall to get all matches
    result = re.findall(pattern, input_string)
    return result
def output_json_parser(input_string):
    input_string = input_string.strip().strip('\n').strip().replace('\n','')
    json_match = re.search(r'```json(.+)```', input_string, re.DOTALL)
    json_match2 = re.search(r'\[.+\]', input_string) # 匹配 [...]
    json_match3 = re.search(r'\{.+\}', input_string) # 匹配 {...}

    parsed_data = None # 1. 使用临时变量

    # --- 统一的解析逻辑 ---
    try:
        if json_match:
            # 优先使用 ```json ... ```
            json_string = json_match.group(1).strip().strip('\n')
            parsed_data = ast.literal_eval(json_string)
        elif input_string.strip('```').strip('"').strip("'").strip().startswith('{') and input_string.strip(
                '```').strip('"').strip("'").strip().endswith('}'):
            # 尝试解析裸露的 {...}
            parsed_data = ast.literal_eval(input_string.strip('```').strip('"').strip("'").strip())
        elif input_string.strip('```').strip('"').strip("'").strip().startswith('[') and input_string.strip(
                '```').strip('"').strip("'").strip().endswith(']'):
            # 尝试解析裸露的 [...]
            parsed_data = ast.literal_eval(input_string.strip('```').strip('"').strip("'").strip())
        elif json_match3:
            # 回退到正则匹配 {...}
            parsed_data = ast.literal_eval(json_match3.group())
        elif json_match2:
            # 回退到正则匹配 [...]
            parsed_data = ast.literal_eval(json_match2.group())
    except Exception as e:
        # 解析失败
        print(f"JSON a_st.literal_eval failed: {e}")
        parsed_data = None

    # --- 2. 规范化 (关键修复) ---
    match_str = [] # 默认值
    if isinstance(parsed_data, dict):
        # 如果解析出的是字典，把它包装成列表
        match_str = [parsed_data]
    elif isinstance(parsed_data, list):
        # 如果解析出的是列表，直接使用
        match_str = parsed_data

    # --- 3. 安全检查 (现在是安全的) ---
    # 检查列表是否为空，或者第一个元素是否为无效的字典
    if len(match_str) == 0 or not isinstance(match_str[0], dict) or "Function" not in match_str[0].keys():
        return [], [] # 解析失败或格式无效，返回空


    # --- 4. 循环处理 (现在是安全的) ---
    func_list = []
    exlanation_lsit = []
    for match in match_str:
        try:
            # 增加一个检查，确保列表中的每一项都是字典
            if not isinstance(match, dict):
                continue

            func_list.append(match['Function'])
            exlanation_lsit.append(
                f"Function: {match['Function']['function_name']}({', '.join([str(i) for i in match['Function']['parameters']])}), Explanation: {match['Explanation']}")
        except KeyError as k:
            print('LLM action 输出解析报错',k.__str__())
            # raise Exception(f'LLM action 输出解析报错 {k.__str__()}') # 建议不要在这里抛出异常，而是跳过错误的条目
            continue
    return func_list, exlanation_lsit

def get_action_list(string):
    # if string[:len('Finish')] == 'Finish':
    #     return [string]
    # else:
    #     # return string.split(', ')
    # return split_checks(string)
    return output_json_parser(string)



def parse_action_json(function_dict):
    action_type = function_dict['function_name']
    argument = function_dict['parameters']
    parameters = []
    for item in argument:
        if type(item) == list and len(item) > 0 and (type(item[0]) == tuple or (type(item[0]) == str and item[0].startswith('(') and item[0].endswith(')') and item[0].count(',') >= 2)):
            for it in item:
                parameters.append(ast.literal_eval(it) if type(it) == str and it.startswith('(') and it.endswith(')') and it.count(',') >= 2 else it )
        elif type(item) == list and 'answer' in (action_type.lower()):
            for it in item:
                parameters.append(it)
        elif type(item) == list:
            parameters.append(tuple(item))
        elif type(item) == str and  item.startswith('(') and item.endswith(')') and item.count(',') >= 2:
            try:
                parameters.append(ast.literal_eval(item))
            except Exception as e:
                print('LLM输出action无法解析',item,e.__str__())
                item_list = [i.strip().strip('(').strip(')').strip('"').strip("'").strip() for i in item.split(',')]
                parameters.append((int(item_list[0]),int(item_list[1]),''.join(item_list[2:])))
        else:
            parameters.append(item)
    return action_type,parameters

def LLM_json_output_parse(output):
    output = output.strip().strip('\n').strip().replace('\n','')
    json_match = re.search(r'```json(.+)```', output, re.DOTALL)
    # 使用正则表达式提取JSON
    json_pattern = r'\{.+\}'
    json_match2 = re.search(json_pattern, output)
    json_match3 = re.search(r'\[.+\]', output)
    if json_match:
        json_string = json_match.group(1)
        try:
            result = ast.literal_eval(json_string.strip().strip('\n'))
        except Exception as e:
            print('LLM输出解析报错', output, e.__str__())
            raise UserWarning('LLM输出解析报错 ' + output + e.__str__())
    elif output.strip('```').strip('"').strip("'").strip().startswith('{') and output.strip(
            '```').strip('"').strip("'").strip().endswith('}'):
        try:
            result = ast.literal_eval(output.strip('```').strip('"').strip("'").strip())
        except Exception as e:
            print('LLM输出解析报错', output, e.__str__())
            raise UserWarning('LLM输出解析报错 ' + output + e.__str__())
    elif output.strip('```').strip('"').strip("'").strip().startswith('[') and output.strip(
            '```').strip('"').strip("'").strip().endswith(']'):
        try:
            result = ast.literal_eval(output.strip('```').strip('"').strip("'").strip())
        except Exception as e:
            print('LLM输出解析报错', output, e.__str__())
            raise UserWarning('LLM输出解析报错 ' + output + e.__str__())
    elif json_match2:
        json_str = json_match2.group()
        # 将JSON字符串转换为字典
        result = ast.literal_eval(json_str)
    elif json_match3:
        json_str = json_match3.group()
        # 将JSON字符串转换为字典
        result = ast.literal_eval(json_str)
    else:
        raise UserWarning('LLM输出解析报错 ' + output)

    return result
def remove_quotes(s):
    s = s.strip().strip('\n')
    if s.startswith(("'", '"','‘','’','“','”')):
        s = s[1:]
    if s.endswith(("'", '"','‘','’','“','”')):
        s = s[:-1]
    return s

# import os
# import re
# import json
#
# def safe_json_parse(json_str):
#     """
#     安全解析 JSON 字符串，自动清除常见格式错误，解析失败返回 None。
#     """
#     def fix_common_errors(s):
#         # 修复 Explanation 后多余的 }
#         s = re.sub(r'("Explanation"\s*:\s*".*?")\s*}', r'\1', s, flags=re.DOTALL)
#         return s.strip().strip('```').strip('"').strip("'")
#
#     try:
#         return json.loads(json_str)
#     except json.JSONDecodeError:
#         try:
#             fixed = fix_common_errors(json_str)
#             return json.loads(fixed)
#         except json.JSONDecodeError as e:
#             print("[警告] JSON解析失败：", e)
#             return None
#
# def split_checks(input_string):
#     pattern = r'[\w]+\[.*?\]'
#     return re.findall(pattern, input_string)
#
# def output_json_parser(input_string):
#     input_string = input_string.strip().replace('\n', '')
#     json_match = re.search(r'```json(.+?)```', input_string, re.DOTALL)
#     json_match2 = re.search(r'\[.+\]', input_string)
#     json_match3 = re.search(r'\{.+\}', input_string)
#
#     match_str = []
#
#     def try_parse(s):
#         result = safe_json_parse(s)
#         if isinstance(result, dict):
#             return [result]
#         if isinstance(result, list):
#             return result
#         return []
#
#     if json_match:
#         match_str = try_parse(json_match.group(1))
#     elif input_string.startswith('[') and input_string.endswith(']'):
#         match_str = try_parse(input_string)
#     elif json_match2:
#         match_str = try_parse(json_match2.group())
#     elif input_string.startswith('{') and input_string.endswith('}'):
#         match_str = try_parse(input_string)
#     elif json_match3:
#         match_str = try_parse(json_match3.group())
#     else:
#         return [], []
#
#     if not match_str or not isinstance(match_str, list) or not isinstance(match_str[0], dict) or "Function" not in match_str[0]:
#         return [], []
#
#     func_list = []
#     explanation_list = []
#     for match in match_str:
#         try:
#             func_list.append(match['Function'])
#             explanation_list.append(
#                 f"Function: {match['Function']['function_name']}({', '.join([str(i) for i in match['Function']['parameters']])}), Explanation: {match['Explanation']}"
#             )
#         except KeyError as k:
#             print('LLM action 输出解析报错:', k)
#             continue
#     return func_list, explanation_list
#
# def get_action_list(string):
#     return output_json_parser(string)
#
# def parse_action_json(function_dict):
#     action_type = function_dict['function_name']
#     argument = function_dict['parameters']
#     parameters = []
#
#     for item in argument:
#         if isinstance(item, list) and len(item) > 0 and isinstance(item[0], str) and item[0].startswith('(') and item[0].endswith(')') and item[0].count(',') >= 2:
#             for it in item:
#                 try:
#                     parsed = safe_json_parse(it.replace("(", "[").replace(")", "]"))
#                     parameters.append(tuple(parsed) if parsed else it)
#                 except Exception as e:
#                     print("参数解析失败:", it, e)
#                     parameters.append(it)
#         elif isinstance(item, list) and 'answer' in action_type.lower():
#             parameters.extend(item)
#         elif isinstance(item, list):
#             parameters.append(tuple(item))
#         elif isinstance(item, str) and item.startswith('(') and item.endswith(')') and item.count(',') >= 2:
#             try:
#                 parsed = safe_json_parse(item.replace("(", "[").replace(")", "]"))
#                 parameters.append(tuple(parsed) if parsed else item)
#             except Exception as e:
#                 print('LLM输出action无法解析:', item, e)
#                 parts = [i.strip().strip('(').strip(')').strip('"').strip("'") for i in item.split(',')]
#                 parameters.append((int(parts[0]), int(parts[1]), ''.join(parts[2:])))
#         else:
#             parameters.append(item)
#
#     return action_type, parameters
#
# def LLM_json_output_parse(output):
#     output = output.strip().replace('\n', '')
#     json_match = re.search(r'```json(.+?)```', output, re.DOTALL)
#     json_match2 = re.search(r'\{.+\}', output)
#     json_match3 = re.search(r'\[.+\]', output)
#
#     json_str = None
#     if json_match:
#         json_str = json_match.group(1)
#     elif output.startswith('{') or output.startswith('['):
#         json_str = output
#     elif json_match2:
#         json_str = json_match2.group()
#     elif json_match3:
#         json_str = json_match3.group()
#
#     if json_str:
#         result = safe_json_parse(json_str)
#         if result is not None:
#             return result
#
#     print('LLM输出解析失败:', output)
#     return {}
#
# def remove_quotes(s):
#     s = s.strip().strip('\n')
#     if s.startswith(("'", '"', '‘', '’', '“', '”')):
#         s = s[1:]
#     if s.endswith(("'", '"', '‘', '’', '“', '”')):
#         s = s[:-1]
#     return s
