"""
ACGS API — 封装 LLM 障碍物生成器的完整生命周期。

职责：
- init_generator: 冷启动生成初始障碍物生成器
- evolve: 根据失败诊断生成新的生成器（含重试 + sanity check）
- generate_obstacles: 调用当前生成器生成障碍物

参考 CurricuLLM/gpt/curriculum_api_chain_fetch.py 的封装模式。
"""

import logging
import numpy as np
from typing import Dict, List, Optional

from DIVO.gpt.utils import get_client, llm_interaction, extract_code
from DIVO.gpt.prompt_builder import PromptBuilder
from DIVO.env.pusht.llm_topology_generator import StrategyExecutor

LOGGER = logging.getLogger(__name__)


class ACGS_API:
    """
    ACGS 闭环 API。

    封装了 LLM 客户端、PromptBuilder、StrategyExecutor，
    对外暴露三个方法：init_generator / evolve / generate_obstacles。

    用法：
        api = ACGS_API(task_name="PushT", prompt_dir="DIVO/gpt/prompt", ...)
        api.init_generator(tblock_pose, num_obstacles)
        obstacles = api.generate_obstacles(tblock_pose, num_obstacles)
        new_code = api.evolve(batch_stats, fv_result, diagnosis, reason, ...)
    """

    def __init__(
        self,
        task_name: str,
        prompt_dir: str,
        api_type: str = "deepseek",
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        obstacle_size: float = 0.01,
        target_pose: Optional[List[float]] = None,
        max_evolve_retries: int = 3,
        sanity_check_count: int = 5,
    ):
        """
        Args:
            task_name: 任务名称，对应 prompt_dir 下的子目录
            prompt_dir: prompt 模板根目录
            api_type: "deepseek" 或 "openai"
            api_key: API 密钥，None 时从环境变量读取
            model: LLM 模型名称
            base_url: API 基础 URL
            temperature: LLM 温度参数
            max_tokens: LLM 最大 token 数
            obstacle_size: 障碍物半边长
            target_pose: 目标位姿 [x, y, θ]
            max_evolve_retries: evolve 最大重试次数
            sanity_check_count: sanity check 测试次数
        """
        # LLM 客户端
        self.client = get_client(api_type=api_type, api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Prompt 构建器
        self.prompt_builder = PromptBuilder(task_name=task_name, prompt_dir=prompt_dir)

        # 代码执行器
        if target_pose is None:
            target_pose = [0, 0, -np.pi / 4]
        self.executor = StrategyExecutor(
            obstacle_size=obstacle_size,
            target_pose=target_pose,
        )

        # 当前生成器代码
        self.topology_generator_code: Optional[str] = None

        # 配置
        self.max_evolve_retries = max_evolve_retries
        self.sanity_check_count = sanity_check_count

        LOGGER.info(f"[ACGS_API] initialized: task={task_name}, model={model}, "
                     f"retries={max_evolve_retries}, sanity_checks={sanity_check_count}")

    # ================================================================
    # 冷启动
    # ================================================================

    def init_generator(self, tblock_pose: np.ndarray, num_obstacles: int) -> Optional[str]:
        """
        冷启动：调用 LLM 生成初始障碍物生成器代码。

        Args:
            tblock_pose: 示例 T-block 位姿 [x, y, θ]
            num_obstacles: 障碍物数量

        Returns:
            生成的代码字符串，失败返回 None
        """
        system = self.prompt_builder.load_initial_system()
        user = self.prompt_builder.build_initial_user(tblock_pose)

        LOGGER.info("[ACGS_API] Generating initial topology generator...")

        raw = llm_interaction(
            self.client, self.model, system, user,
            temperature=self.temperature, max_tokens=self.max_tokens,
        )
        code = extract_code(raw)

        if code is None:
            LOGGER.error("[ACGS_API] Initial code generation failed")
            return None

        if not self.executor.load_topology_generator(code):
            LOGGER.error("[ACGS_API] Initial code loading failed")
            return None

        self.topology_generator_code = code
        LOGGER.info(f"[ACGS_API] Initial generator loaded ({len(code)} chars)")
        return code

    # ================================================================
    # Evolve
    # ================================================================

    def evolve(
        self,
        batch_stats: Dict[str, int],
        fv_result: Optional[Dict],
        diagnosis: Optional[Dict],
        reason: str,
        failure_replays_text: str,
        success_replays_text: str,
        current_generator_code: Optional[str] = None,
        num_obstacles: int = 2,
    ) -> Optional[str]:
        """
        根据失败诊断生成新的障碍物生成器代码。含重试 + sanity check。

        Args:
            batch_stats: 粗粒度批次统计
            fv_result: Level 1 failure vector 结果
            diagnosis: Level 2 failure diagnosis 结果
            reason: evolve 触发原因
            failure_replays_text: 格式化的失败回放文本
            success_replays_text: 格式化的成功回放文本
            current_generator_code: 当前生成器代码（None 时用 self.topology_generator_code）
            num_obstacles: 障碍物数量（sanity check 用）

        Returns:
            新的代码字符串，全部重试失败返回 None
        """
        if current_generator_code is None:
            current_generator_code = self.topology_generator_code

        system = self.prompt_builder.load_evolve_system()
        user = self.prompt_builder.build_evolve_user(
            batch_stats=batch_stats,
            fv_result=fv_result,
            diagnosis=diagnosis,
            reason=reason,
            failure_replays_text=failure_replays_text,
            success_replays_text=success_replays_text,
            current_generator_code=current_generator_code,
        )

        old_code = self.topology_generator_code

        for attempt in range(self.max_evolve_retries):
            LOGGER.info(f"[ACGS_API] Evolve attempt {attempt + 1}/{self.max_evolve_retries}...")

            raw = llm_interaction(
                self.client, self.model, system, user,
                temperature=self.temperature, max_tokens=self.max_tokens,
            )
            code = extract_code(raw)

            if code is None:
                LOGGER.warning(f"[ACGS_API] Attempt {attempt + 1}: LLM returned no code")
                continue

            if not self.executor.load_topology_generator(code):
                LOGGER.warning(f"[ACGS_API] Attempt {attempt + 1}: code loading failed")
                continue

            # Sanity check
            if not self.executor.sanity_check(
                num_tests=self.sanity_check_count,
                num_obstacles=num_obstacles,
            ):
                LOGGER.warning(f"[ACGS_API] Attempt {attempt + 1}: sanity check failed")
                # 恢复旧生成器
                if old_code:
                    self.executor.load_topology_generator(old_code)
                continue

            # 全部通过
            self.topology_generator_code = code
            LOGGER.info(f"[ACGS_API] Evolve succeeded (attempt {attempt + 1}), "
                         f"new generator loaded ({len(code)} chars)")
            return code

        # 全部重试失败
        LOGGER.error(f"[ACGS_API] Evolve failed after {self.max_evolve_retries} attempts")
        if old_code:
            self.executor.load_topology_generator(old_code)
        return None

    # ================================================================
    # 生成障碍物
    # ================================================================

    def generate_obstacles(self, tblock_pose: np.ndarray, num_obstacles: int) -> List[Dict]:
        """
        调用当前生成器生成障碍物。

        Args:
            tblock_pose: T-block 位姿 [x, y, θ]
            num_obstacles: 障碍物数量

        Returns:
            障碍物列表 [{'x': float, 'y': float, 'purpose': str}, ...]
        """
        return self.executor.generate(tblock_pose, num_obstacles)

    # ================================================================
    # 状态查询
    # ================================================================

    @property
    def has_generator(self) -> bool:
        """当前是否有可用的生成器。"""
        return self.topology_generator_code is not None and self.executor.generate_obstacles is not None

    def get_prompt_text(
        self,
        batch_stats: Dict[str, int],
        fv_result: Optional[Dict],
        diagnosis: Optional[Dict],
        reason: str,
        failure_replays_text: str,
        success_replays_text: str,
        current_generator_code: Optional[str] = None,
    ) -> tuple:
        """
        返回 (system_prompt, user_prompt) 文本，用于日志记录。
        不调用 LLM。
        """
        system = self.prompt_builder.load_evolve_system()
        user = self.prompt_builder.build_evolve_user(
            batch_stats=batch_stats,
            fv_result=fv_result,
            diagnosis=diagnosis,
            reason=reason,
            failure_replays_text=failure_replays_text,
            success_replays_text=success_replays_text,
            current_generator_code=current_generator_code or self.topology_generator_code,
        )
        return system, user
