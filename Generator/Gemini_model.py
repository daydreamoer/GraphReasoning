import google.generativeai as genai
from configs import GEMINI_KEYS
import random
import time
from google.api_core.exceptions import ResourceExhausted

class GeminiTool:
    def __init__(self, key, args):
        self.key_index = random.randint(0, len(GEMINI_KEYS) - 1)
        # 确保这里正确获取模型名称
        self.model_name = args.model
        self.args = args

        # Create the model
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUAL", "threshold": "BLOCK_NONE"},
        ]

    # 修改：增加 response_format 参数以兼容 OpenAI 风格的调用
    def generate(self, prompt, system_instruction='You are a helpful AI bot.', isrepeated=0.0, response_mime_type=None, response_format=None):
        genai.configure(api_key=GEMINI_KEYS[self.key_index])

        # --- [新增逻辑]：兼容 response_format 参数 ---
        # 如果外部传入了 OpenAI 风格的 response_format={"type": "json_object"}，
        # 自动转换为 Gemini 需要的 response_mime_type="application/json"
        if response_mime_type is None and response_format is not None:
            if isinstance(response_format, dict) and response_format.get("type") == "json_object":
                response_mime_type = "application/json"
            elif response_format == "json_object":
                response_mime_type = "application/json"

        # 默认值
        final_mime_type = response_mime_type if response_mime_type is not None else "text/plain"
        # -------------------------------------------

        generation_config = {
            "temperature": self.args.temperature + isrepeated if isrepeated > 0.0 else self.args.temperature,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": final_mime_type,
        }

        # --- [修复 BUG]：使用 self.model_name 而不是 self.args.reasoning_model_path ---
        model = genai.GenerativeModel(
            model_name=self.model_name, # <-- 修复点
            safety_settings=self.safety_settings,
            generation_config=generation_config,
            system_instruction=system_instruction,
        )

        error = 3
        while error > 0:
            try:
                output = model.generate_content(prompt)
                result = output.text
                break
            except ValueError as v:
                # 可能是安全拦截或其他值错误
                print('ValueError (可能被安全设置拦截):', v.__str__())
                raise UserWarning('unsafe input ' + v.__str__())
            except Exception as e:
                gemini_key_index = self.key_index
                print(GEMINI_KEYS[gemini_key_index], '报错了', e.__str__())

                # 轮换 Key
                gemini_key_index = random.randint(0, len(GEMINI_KEYS) - 1)
                genai.configure(api_key=GEMINI_KEYS[gemini_key_index])

                # 重新初始化模型 (修复点同样应用在这里)
                model = genai.GenerativeModel(
                    model_name=self.model_name, # <-- 修复点
                    safety_settings=self.safety_settings,
                    generation_config=generation_config,
                    system_instruction=system_instruction,
                )
                self.key_index = gemini_key_index
                print('更换Gemini key为', GEMINI_KEYS[gemini_key_index])
                time.sleep(2.0)
                error -= 1

        if error <= 0:
            raise UserWarning('gemini 报错')

        return result