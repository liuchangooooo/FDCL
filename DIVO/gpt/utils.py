"""
LLM 调用工具函数。

参考 CurricuLLM/gpt/utils.py，适配 DIVO 项目：
- 支持 DeepSeek / OpenAI 双后端
- 从环境变量读取 API key（不依赖 yaml 文件）
- 包含代码提取逻辑（从 LLM 响应中提取 Python 代码）
"""

import os
import importlib
import logging
from typing import Optional

LOGGER = logging.getLogger(__name__)

# 延迟导入 openai
_openai = None
_openai_import_error = None
try:
    _openai = importlib.import_module("openai")
except Exception as exc:
    _openai_import_error = exc


def file_to_string(filename: str) -> str:
    """读取文件内容为字符串。"""
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


def save_string_to_file(save_path: str, content: str):
    """将字符串写入文件（自动创建目录）。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)


def get_client(api_type: str = "deepseek",
               api_key: Optional[str] = None,
               base_url: Optional[str] = None):
    """
    创建 OpenAI 兼容客户端。

    Args:
        api_type: "deepseek" 或 "openai"
        api_key: API 密钥，None 时从环境变量读取
        base_url: API 基础 URL，None 时使用默认值
    """
    if _openai is None:
        raise ImportError(
            "Missing dependency 'openai'. Install with `pip install openai`."
        ) from _openai_import_error

    if api_key is None:
        if api_type == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("No API key found. Set DEEPSEEK_API_KEY or OPENAI_API_KEY.")

    if api_type == "deepseek":
        return _openai.OpenAI(
            api_key=api_key,
            base_url=base_url or "https://api.deepseek.com",
        )
    else:
        return _openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )


def llm_interaction(client,
                    model: str,
                    system_string: str,
                    user_string: str,
                    temperature: float = 0.7,
                    max_tokens: int = 2000,
                    max_retries: int = 3) -> Optional[str]:
    """
    调用 LLM 并返回响应文本。带重试。

    Args:
        client: OpenAI 兼容客户端
        model: 模型名称
        system_string: system prompt
        user_string: user prompt
        temperature: 温度参数
        max_tokens: 最大 token 数
        max_retries: 最大重试次数

    Returns:
        LLM 响应文本，失败返回 None
    """
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_string},
                    {"role": "user", "content": user_string},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = completion.choices[0].message.content
            if content:
                return content
        except Exception as e:
            LOGGER.warning(f"LLM call attempt {attempt + 1}/{max_retries} failed: {e}")
    LOGGER.error(f"LLM call failed after {max_retries} attempts")
    return None


def extract_code(response: str) -> Optional[str]:
    """
    从 LLM 响应中提取 Python 代码。

    按优先级尝试：
    1. ```python ... ``` 代码块
    2. ``` ... ``` 代码块
    3. def generate_obstacles 开头的代码
    4. 包含 import numpy 的整段响应

    Args:
        response: LLM 的原始响应

    Returns:
        提取的 Python 代码，失败返回 None
    """
    if response is None:
        return None

    # 方法 1：```python 代码块
    if "```python" in response:
        start = response.find("```python") + 9
        end = response.find("```", start)
        if end != -1:
            return response[start:end].strip()

    # 方法 2：``` 代码块
    if "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end != -1:
            code = response[start:end].strip()
            if code.startswith("python"):
                code = code[6:].strip()
            return code

    # 方法 3：def generate_obstacles
    if "def generate_obstacles" in response:
        start = response.find("def generate_obstacles")
        return response[start:].strip()

    # 方法 4：整段响应
    if "import numpy" in response or "import np" in response:
        return response.strip()

    return None
