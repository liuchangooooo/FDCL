"""LLM 调用封装(生成/重写 generate_pillars 代码)。

API key 优先来自 Nav YAML，未配置时回退环境变量 OPENAI_API_KEY。无 key 时可用 provider='mock'
走 MockGenerator(不产代码,直接给可调用生成器),供 verifier/evolve 端到端测试。

真实 LLM 路径 provider='openai' 需联网 + key;此处封装接口,实际大规模跑由用户提供 key。
"""
import os
import re

from nav.curriculum.generator_source import SandboxNavExecutor, MockGenerator


def _extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    code = m.group(1) if m else text
    return code.strip()


class NavACGS:
    def __init__(self, provider="mock", model="gpt-5.5", temperature=0.7,
                 max_tokens=1500, timeout_sec=5, base_url=None, api_key=None):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url          # 对齐 Push-T exp_llm_curriculum(自定义端点)
        self.api_key = str(api_key or "").strip().strip("\"'“”")
        self.executor = SandboxNavExecutor(timeout_sec=timeout_sec)
        self._mock_diff = 0.5  # mock 难度状态(evolve 时按方向调)

    # ---------------- 真实 LLM ----------------
    def _call_llm(self, system, user):
        if self.provider != "openai":
            raise RuntimeError("LLM provider not configured; use provider='mock' or set up openai")
        from openai import OpenAI  # 延迟导入
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "Missing API key: set curriculum.api_key or OPENAI_API_KEY"
            )
        kwargs = {"api_key": api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = OpenAI(**kwargs)
        resp = client.chat.completions.create(
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        return resp.choices[0].message.content

    def generate_code(self, system, user):
        """返回 (executor_with_loaded_code, code_str) 或 (None, reason)。每次新建独立 executor。"""
        try:
            raw = self._call_llm(system, user)
        except Exception as exc:
            # SDK 已完成其内部重试；这里只隔离可恢复的上游/传输故障，避免单个
            # candidate 的偶发 429/5xx/断连终止整次 RL 训练。认证和配置错误仍抛出。
            from openai import (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            )
            recoverable = (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            )
            if not isinstance(exc, recoverable):
                raise
            return None, f"llm_request_failed:{type(exc).__name__}:{exc}"
        code = _extract_code(raw)
        ex = SandboxNavExecutor(timeout_sec=self.executor.timeout_sec)
        ok, reason = ex.load_code(code)
        return (ex, code) if ok else (None, reason)

    # ---------------- mock 路径 ----------------
    def mock_candidates(self, direction, R=3, seed=0):
        """按方向产 R 个 MockGenerator 候选(供 verifier 选择)。

        HARDEN -> 提高难度;RELAX -> 降低;PRESERVE/其它 -> 同难度换 seed 结构。
        """
        cands = []
        for r in range(R):
            d = self._mock_diff
            if direction == "HARDEN":
                d = min(1.0, self._mock_diff + 0.15 * (r + 1))
            elif direction == "RELAX":
                d = max(0.0, self._mock_diff - 0.15 * (r + 1))
            else:
                d = self._mock_diff
            cands.append(MockGenerator(difficulty=d, seed=seed * 10 + r))
        return cands

    def set_mock_difficulty(self, d):
        self._mock_diff = float(max(0.0, min(1.0, d)))
