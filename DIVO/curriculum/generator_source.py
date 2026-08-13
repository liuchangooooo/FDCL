from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _cfg_get(cfg: Optional[Any], *paths: str, default=None):
    for path in paths:
        cur = cfg
        ok = True
        for part in path.split('.'):
            if cur is None:
                ok = False
                break
            if isinstance(cur, dict):
                if part not in cur:
                    ok = False
                    break
                cur = cur[part]
            else:
                if not hasattr(cur, part):
                    ok = False
                    break
                cur = getattr(cur, part)
        if ok and cur is not None:
            return cur
    return default


class GeneratorSource(ABC):
    @abstractmethod
    def get_initial_code(
        self,
        sample_tblock_pose: np.ndarray,
        num_obstacles: int,
        acgs_api: Any,
    ) -> Optional[str]:
        raise NotImplementedError


class LLMGeneratorSource(GeneratorSource):
    def get_initial_code(
        self,
        sample_tblock_pose: np.ndarray,
        num_obstacles: int,
        acgs_api: Any,
    ) -> Optional[str]:
        if acgs_api is None:
            raise RuntimeError("ACGS_API is required for LLMGeneratorSource")
        return acgs_api.init_generator(sample_tblock_pose, num_obstacles)


class FileGeneratorSource(GeneratorSource):
    def __init__(self, path: str):
        self.path = str(Path(path).expanduser().resolve())

    def get_initial_code(
        self,
        sample_tblock_pose: np.ndarray,
        num_obstacles: int,
        acgs_api: Any,
    ) -> Optional[str]:
        return Path(self.path).read_text(encoding="utf-8")


def build_generator_source(curriculum_cfg: Optional[Any]) -> GeneratorSource:
    init_mode = str(_cfg_get(
        curriculum_cfg,
        "generator.init_mode",
        default="llm",
    )).lower()
    init_path = _cfg_get(
        curriculum_cfg,
        "generator.init_path",
        "initial_generator_path",
        default=None,
    )

    if init_mode == "file" and init_path:
        return FileGeneratorSource(str(init_path))

    if init_mode == "file" and not init_path:
        # 保持运行兼容：未提供文件路径时回退到 LLM 初始化。
        return LLMGeneratorSource()

    if init_path:
        return FileGeneratorSource(str(init_path))
    return LLMGeneratorSource()
