"""
LLM 调用工具函数。

参考 CurricuLLM/gpt/utils.py，适配 DIVO 项目：
- 支持 DeepSeek / OpenAI 双后端
- 从环境变量读取 API key（不依赖 yaml 文件）
- 包含代码提取逻辑（从 LLM 响应中提取 Python 代码）
"""

import os
import importlib
import json
import logging
from typing import Any, Optional

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


def _get_env_api_key(api_type: str) -> Optional[str]:
    """按后端类型读取默认 API key。"""
    if api_type == "deepseek":
        return os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_type == "openai":
        return os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    raise ValueError(f"Unsupported api_type: {api_type}")


def _get_env_base_url(api_type: str) -> Optional[str]:
    """按后端类型读取默认 base_url。"""
    if api_type == "deepseek":
        return os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    if api_type == "openai":
        return os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    raise ValueError(f"Unsupported api_type: {api_type}")


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
        api_key = _get_env_api_key(api_type)

    if base_url is None:
        base_url = _get_env_base_url(api_type)

    if not api_key:
        raise ValueError("No API key found. Set DEEPSEEK_API_KEY or OPENAI_API_KEY.")

    if api_type not in {"deepseek", "openai"}:
        raise ValueError(f"Unsupported api_type: {api_type}")

    return _openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def _content_to_text(content: Any) -> Optional[str]:
    """Normalize OpenAI-compatible message content into plain text."""
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(text, dict) and isinstance(text.get("value"), str):
                    parts.append(text["value"])
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
                elif hasattr(text, "value") and isinstance(text.value, str):
                    parts.append(text.value)
        return "\n".join(parts) if parts else None
    return str(content)


def _extract_completion_text(completion: Any) -> Optional[str]:
    """
    Extract text from OpenAI SDK objects, OpenAI-compatible dicts, or proxies
    that directly return a response string.
    """
    if completion is None:
        return None
    if isinstance(completion, str):
        return completion

    output_text = getattr(completion, "output_text", None)
    if isinstance(output_text, str):
        return output_text

    if isinstance(completion, dict):
        direct = completion.get("output_text") or completion.get("content")
        text = _content_to_text(direct)
        if text:
            return text
        choices = completion.get("choices") or []
        if choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message") or {}
                if isinstance(message, dict):
                    return _content_to_text(message.get("content"))
                return _content_to_text(getattr(message, "content", None))
            message = getattr(choice, "message", None)
            if message is not None:
                return _content_to_text(getattr(message, "content", None))
            return _content_to_text(getattr(choice, "text", None))
        return None

    choices = getattr(completion, "choices", None)
    if choices:
        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is not None:
            return _content_to_text(getattr(message, "content", None))
        return _content_to_text(getattr(choice, "text", None))

    return None


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
            content = _extract_completion_text(completion)
            if content:
                return content
            LOGGER.warning(
                "LLM call attempt %s/%s returned empty/unrecognized response type: %s",
                attempt + 1,
                max_retries,
                type(completion).__name__,
            )
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
    3. def generate_obstacles / def generate_maze_map 开头的代码
    4. 包含 import numpy 的整段响应

    Args:
        response: LLM 的原始响应

    Returns:
        提取的 Python 代码，失败返回 None
    """
    if response is None:
        return None

    if not isinstance(response, str):
        response = _extract_completion_text(response)
        if response is None:
            return None

    response = response.strip()
    if not response:
        return None

    if response.startswith("{") or response.startswith("["):
        try:
            decoded = json.loads(response)
        except json.JSONDecodeError:
            decoded = None
        if decoded is not None:
            decoded_text = _extract_completion_text(decoded)
            if decoded_text and decoded_text != response:
                code = extract_code(decoded_text)
                if code:
                    return code

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

    if "def generate_maze_map" in response:
        start = response.find("def generate_maze_map")
        return response[start:].strip()

    # 方法 4：整段响应
    if "import numpy" in response or "import np" in response:
        return response.strip()

    return None
