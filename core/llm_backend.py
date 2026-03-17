import openai
from typing import Tuple
from config import LLM_GLOBAL_CONFIG

def llm_generate_code(prompt: str) -> Tuple[str, str]:
    """
    ✅ 统一调用第三方LLM（自动读取全局配置）
    :return: (建模思路, 生成代码)
    """
    client = openai.OpenAI(
        api_key=LLM_GLOBAL_CONFIG["API_KEY"],
        base_url=LLM_GLOBAL_CONFIG["BASE_URL"]
    )

    try:
        # 不同模型支持的参数可能不同
        model_name = LLM_GLOBAL_CONFIG["MODEL"]
        
        # 构建基本参数
        params = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048
        }
        
        # 添加 temperature 参数增加多样性
        # 较高的 temperature 值会增加随机性，避免重复结果
        params["temperature"] = 0.8
        
        response = client.chat.completions.create(**params)
        content = response.choices[0].message.content
        
        # 检查 content 是否为 None
        if content is None:
            print("LLM 返回内容为 None")
            return "", ""
        
        content = content.strip()

        # 拆分思路 + Python代码
        if "```python" in content:
            plan, code_part = content.split("```python", 1)
            code = code_part.split("```")[0].strip()
            return plan.strip(), code
        return "", content

    except Exception as e:
        print(f"LLM调用失败：{str(e)}")
        return "", ""
