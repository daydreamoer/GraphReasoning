import os
import time

from dotenv import load_dotenv
from openai import OpenAI
import openai


class ChatGPTTool(object):
    def __init__(self,args):
        load_dotenv()
        #self.API_SECRET_KEY = args.key
        self.API_SECRET_KEY = os.getenv("OPENAI_API_KEY")  # 从 .env 文件中读取 API 密钥
        self.BASE_URL = args.base_url
        self.model_name = args.model
        self.args = args

        # chat
        if self.BASE_URL:
            self.client = OpenAI(api_key=self.API_SECRET_KEY, base_url=self.BASE_URL)
        else:
            self.client = OpenAI(api_key=self.API_SECRET_KEY)

#    修改源代码参数response_mime_type（gemini）为response_format
#    def generate(self, prompt, system_instruction='You are a helpful AI bot.', isrepeated=0.0, response_mime_type=None):
    def generate(self, prompt, system_instruction='You are a helpful AI bot.', isrepeated=0.0, response_format=None):
        if isrepeated > 0.0:
            temperature = self.args.temperature + isrepeated
        else:
            temperature = self.args.temperature

        error = 3
        resp = None  # 初始化resp变量

        while error > 0:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": '\n'.join(prompt)}
                    ],
                    temperature=temperature,
                    seed=42,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    response_format=response_format
                )
                break
            except openai.RateLimitError as r:
                print('openai限流了', r.__str__())
                error -= 1
                time.sleep(4.0)
            except openai.InternalServerError as r:
                print('openai奔溃了', r.__str__())
                error -= 1
                time.sleep(2.0)
            except openai.APITimeoutError as a:
                print('openai超时', a.__str__())
                raise UserWarning(f'openai超时 {a.__str__()}')
            except Exception as r:
                print('openai报错了', r.__str__())
                error -= 1
                time.sleep(2.0)

        # 所有重试失败后处理
        if resp is None:
            raise RuntimeError("所有重试失败，无法获取OpenAI API响应")

        output = resp.choices[0].message.content
        return output


# import time
# import os
# from openai import OpenAI
# from dotenv import load_dotenv
#
# load_dotenv()  # 默认读取当前目录下的 .env 文件
#
#
# class DashScopeChatTool(object):
#     def __init__(self, args):
#         self.API_SECRET_KEY = args.key
#         self.BASE_URL = args.base_url
#         self.model_name = args.model
#
#         self.args = args
#         # 初始化客户端
#         if self.BASE_URL:
#             self.client = OpenAI(api_key=self.API_SECRET_KEY, base_url=self.BASE_URL)
#         else:
#             self.client = OpenAI(api_key=self.API_SECRET_KEY)
#
#     def generate(self, prompt, system_instruction='You are a helpful AI bot.', isrepeated=0.0):
#         global resp
#         if isrepeated > 0.0:
#             temperature = self.args.temperature + isrepeated
#         else:
#             temperature = self.args.temperature
#         error = 3
#         while error > 0:
#             try:
#                 resp = self.client.chat.completions.create(
#                     model=self.model_name,
#                     messages=[
#                         {"role": "system", "content": system_instruction},
#                         {"role": "user", "content": '\n'.join(prompt)}
#                     ],
#                     temperature=temperature,
#                     top_p=0.95,
#                     frequency_penalty=0,
#                     presence_penalty=0,
#                 )
#                 break
#             except Exception as e:
#                 print(f"Error: {e}")
#                 error -= 1
#                 time.sleep(2.0)
#
#         output = resp.model_dump_json()  # Use model_dump_json to extract formatted result
#         return output
# import requests
# import time
# from dotenv import load_dotenv
# import os
#
#
# class DashScopeChatTool(object):
#
#     def __init__(self, args):
#
#         load_dotenv()  # 加载 .env 文件
#         self.API_SECRET_KEY = os.getenv("OPENAI_API_KEY")  # 从 .env 文件中读取 API 密钥
#         if not self.API_SECRET_KEY:  # 如果没有找到密钥，抛出异常
#             raise ValueError("API KEY 未设置，请在 .env 文件中配置 API_KEY")
#         self.model_name = args.model  # 示例: "qwen-turbo"
#         self.args = args
#
#     def generate(self, prompt, system_instruction='You are a helpful AI bot.', isrepeated=0.0):
#         temperature = self.args.temperature + isrepeated if isrepeated > 0.0 else self.args.temperature
#         error = 3
#         while error > 0:
#             try:
#                 url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
#                 headers = {
#                     "Authorization": f"Bearer {self.API_SECRET_KEY}",
#                     "Content-Type": "application/json"
#                 }
#                 messages = [
#                     {"role": "system", "content": system_instruction},
#                     {"role": "user", "content": '\n'.join(prompt)}
#                 ]
#                 data = {
#                     "model": self.model_name,
#                     "input": {"messages": messages},
#                     "parameters": {
#                         "temperature": temperature,
#                         "top_p": 0.95,
#                         "seed": 42
#                     }
#                 }
#
#                 resp = requests.post(url, headers=headers, json=data)
#
#                 if resp.status_code == 200:
#                     result = resp.json()
#                     output = result["output"]["text"]
#                     return output
#                 else:
#                     # print("API KEY:", self.API_SECRET_KEY)
#
#                     print(f"调用百炼失败: {resp.status_code} {resp.text}")
#                     error -= 1
#                     time.sleep(2.0)
#             except Exception as e:
#                 print(f"Error: {e}")
#                 error -= 1
#                 time.sleep(2.0)
#
#         raise RuntimeError("调用百炼接口失败，超过最大重试次数")
