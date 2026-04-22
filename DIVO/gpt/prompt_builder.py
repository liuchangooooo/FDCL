"""
PromptBuilder — 加载 prompt 模板并填充占位符。

设计原则（参考 CurricuLLM）：
- prompt 模板是纯文本文件，放在 gpt/prompt/{task_name}/ 下
- 占位符用 Python str.format() 风格：{variable_name}
- PromptBuilder 只负责"加载模板 + 填充占位符"，不负责 LLM 调用
- 换任务只需要新建 prompt 文件夹，不需要改 Python 代码
"""

import os
import numpy as np
from typing import Dict, List, Optional, Any


def _load_template(path: str) -> str:
    """读取模板文件，返回原始字符串。"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# 9 维 failure key 定义（与 td3_curriculum_workspace 保持一致）
_FAILURE_KEYS = [
    "success",
    "collision_rod_early", "collision_rod_mid", "collision_rod_late",
    "collision_tblock_early", "collision_tblock_mid", "collision_tblock_late",
    "timeout", "fall",
]


class PromptBuilder:
    """
    加载 prompt 模板并根据运行时数据填充占位符。

    用法：
        builder = PromptBuilder(task_name="PushT", prompt_dir="DIVO/gpt/prompt")
        system = builder.load_evolve_system()
        user = builder.build_evolve_user(
            batch_stats=..., fv_result=..., diagnosis=..., reason=...,
            failure_replays_text=..., success_replays_text=...,
            current_generator_code=...,
        )
    """

    def __init__(self, task_name: str, prompt_dir: str):
        self.task_name = task_name
        self.task_prompt_dir = os.path.join(prompt_dir, task_name)
        if not os.path.isdir(self.task_prompt_dir):
            raise FileNotFoundError(
                f"Prompt directory not found: {self.task_prompt_dir}"
            )

    # ================================================================
    # 模板加载
    # ================================================================

    def load_initial_system(self) -> str:
        return _load_template(os.path.join(self.task_prompt_dir, "initial_system.txt"))

    def load_initial_user(self) -> str:
        return _load_template(os.path.join(self.task_prompt_dir, "initial_user.txt"))

    def load_evolve_system(self) -> str:
        return _load_template(os.path.join(self.task_prompt_dir, "evolve_system.txt"))

    def _load_evolve_user_template(self) -> str:
        return _load_template(os.path.join(self.task_prompt_dir, "evolve_user.txt"))

    # ================================================================
    # 初始生成器 prompt
    # ================================================================

    def build_initial_user(self, tblock_pose: np.ndarray) -> str:
        """填充 initial_user.txt 的占位符。"""
        template = self.load_initial_user()
        tx, ty, ttheta = tblock_pose
        return template.format(
            tx=tx, ty=ty, theta_deg=np.degrees(ttheta),
        )

    # ================================================================
    # Evolve prompt — 各 block 的格式化
    # ================================================================

    def _format_batch_stats(self, batch_stats: Dict[str, int]) -> str:
        total = max(sum(batch_stats.values()), 1)
        lines = []
        for key in ("success", "collision", "timeout", "fall"):
            cnt = batch_stats.get(key, 0)
            lines.append(f"- {key}: {cnt} ({cnt / total:.1%})")
        return "\n".join(lines)

    def _format_failure_distribution(self, fv_result: Optional[Dict]) -> str:
        if fv_result is None:
            return "(not available)"
        lines = []
        distribution = fv_result.get("distribution", {})
        counts = fv_result.get("counts", {})
        for key in _FAILURE_KEYS:
            ratio = float(distribution.get(key, 0.0))
            cnt = int(counts.get(key, 0))
            lines.append(f"- {key}: {ratio:.1%} ({cnt})")
        dominant = fv_result.get(
            "dominant_failure_type", fv_result.get("dominant_type", "unknown")
        )
        lines.append(f"- dominant_failure_type: {dominant}")
        return "\n".join(lines)

    def _format_failure_diagnosis(self, diagnosis: Optional[Dict]) -> str:
        if diagnosis is None:
            return "(not available)"
        reliability = diagnosis.get("reliability", "weak")
        lines = [f"- reliability: {reliability}"]

        if reliability in ("strong", "medium"):
            region = diagnosis.get("failure_region", {})
            bias = diagnosis.get("behavior_bias", {})
            source = bias.get("source", diagnosis.get("behavior_source", "none"))
            lines.append(
                f"- failure_region: {region.get('label', 'none')} "
                f"(confidence={float(region.get('confidence', 0.0)):.2f})"
            )
            lines.append(
                f"- behavior_bias: {bias.get('label', 'none')} "
                f"(confidence={float(bias.get('confidence', 0.0)):.2f}, source={source})"
            )
            lines.append(f"- sample_count: {int(diagnosis.get('sample_count', 0))}")
        else:
            lines.append(
                "- The current failure pattern is not spatially concentrated "
                "enough for precise targeting."
            )
            lines.append(
                "- Focus on dominant failure type and overall failure "
                "distribution conservatively."
            )
            lines.append(f"- sample_count: {int(diagnosis.get('sample_count', 0))}")
        return "\n".join(lines)

    def _format_revision_instruction(
        self,
        reason: str,
        dominant_type: str,
        diagnosis: Optional[Dict],
        fall_rate: float,
        timeout_rate: float,
    ) -> str:
        """根据 reason + diagnosis 生成三档调整指令。"""
        diag_reliable = False
        if diagnosis is not None:
            diag_reliable = bool(diagnosis.get("is_reliable", False))

        lines = []
        lines.append(f"- trigger_reason: {reason if reason else 'unspecified'}")
        lines.append(f"- dominant_failure_type: {dominant_type}")
        lines.append("- Global: preserve solvability, avoid trivially impossible layouts.")
        lines.append("- Global: do not uniformly increase difficulty; target the dominant failure risk.")

        reason_norm = (reason or "").lower()
        is_too_easy = "too_easy" in reason_norm or "too easy" in reason_norm
        is_too_hard = any(k in reason_norm for k in ("too_hard", "too hard", "impossible", "warmup_failed"))
        is_plateau = "plateau" in reason_norm

        if is_too_easy:
            lines.append("- Template: Challenge-Up")
            lines.append(f"- Objective: increase challenge targeting {dominant_type}.")
            lines.append("- Preserve at least one feasible alternative route.")
            if diag_reliable:
                region = diagnosis.get("failure_region", {})
                bias = diagnosis.get("behavior_bias", {})
                lines.append(
                    f"- Region hint: increase pressure near {region.get('label', 'unknown')} "
                    f"(conf={float(region.get('confidence', 0.0)):.2f})."
                )
                lines.append(
                    f"- Behavior hint: challenge the dominant pattern "
                    f"{bias.get('label', 'unknown')} "
                    f"(conf={float(bias.get('confidence', 0.0)):.2f})."
                )

        elif is_too_hard:
            lines.append("- Template: Recovery")
            lines.append(f"- Objective: reduce {dominant_type} risk, restore learnability.")
            if dominant_type.startswith("collision_") and dominant_type.endswith("_early"):
                lines.append("- Reduce early collision risk; avoid strong frontal obstruction near start.")
            if fall_rate > 0.3:
                lines.append(
                    f"- WARNING: fall rate is {fall_rate:.1%}. "
                    "Reduce out-of-bounds risk; improve early controllability."
                )
            if timeout_rate > 0.3:
                lines.append("- Ensure at least one clear passable route to goal.")
            if diag_reliable:
                region = diagnosis.get("failure_region", {})
                bias = diagnosis.get("behavior_bias", {})
                lines.append(
                    f"- Region hint: reduce obstacle density near "
                    f"{region.get('label', 'unknown')} "
                    f"(conf={float(region.get('confidence', 0.0)):.2f})."
                )
                lines.append(
                    f"- Behavior hint: reduce frontal blocking along "
                    f"{bias.get('label', 'unknown')}."
                )

        elif is_plateau:
            lines.append("- Template: Layout-Shift")
            lines.append("- Objective: change obstacle topology structure, not just coordinates.")
            lines.append("- Avoid repeating the current dominant interference pattern.")
            if diag_reliable:
                region = diagnosis.get("failure_region", {})
                bias = diagnosis.get("behavior_bias", {})
                lines.append(
                    f"- Region hint: shift interference away from "
                    f"{region.get('label', 'unknown')}; introduce new spatial distribution."
                )
                lines.append(
                    f"- Behavior hint: discourage the habitual path "
                    f"{bias.get('label', 'unknown')}; encourage alternative strategies."
                )

        else:
            lines.append("- Template: Balanced")
            lines.append("- Objective: make small learnable adjustments based on failure distribution.")

        return "\n".join(lines)

    # ================================================================
    # Evolve prompt — 组装
    # ================================================================

    def build_evolve_user(
        self,
        batch_stats: Dict[str, int],
        fv_result: Optional[Dict],
        diagnosis: Optional[Dict],
        reason: str,
        failure_replays_text: str,
        success_replays_text: str,
        current_generator_code: Optional[str],
    ) -> str:
        """加载 evolve_user.txt 模板并填充所有占位符。"""
        template = self._load_evolve_user_template()

        # 计算 fall_rate / timeout_rate 供 revision instruction 使用
        total = max(sum(batch_stats.values()), 1)
        fall_rate = batch_stats.get("fall", 0) / total
        timeout_rate = batch_stats.get("timeout", 0) / total

        dominant_type = "unknown"
        if fv_result is not None:
            dominant_type = fv_result.get(
                "dominant_failure_type", fv_result.get("dominant_type", "unknown")
            )

        # 当前生成器代码 block
        if current_generator_code:
            gen_block = f"```python\n{current_generator_code}\n```"
        else:
            gen_block = "(no current generator available)"

        filled = template.format(
            batch_stats_block=self._format_batch_stats(batch_stats),
            failure_distribution_block=self._format_failure_distribution(fv_result),
            failure_diagnosis_block=self._format_failure_diagnosis(diagnosis),
            diagnosis_history_block="(not yet implemented)",
            revision_instruction_block=self._format_revision_instruction(
                reason=reason,
                dominant_type=dominant_type,
                diagnosis=diagnosis,
                fall_rate=fall_rate,
                timeout_rate=timeout_rate,
            ),
            failure_replays_block=failure_replays_text if failure_replays_text else "(no failure replays)",
            success_replays_block=success_replays_text if success_replays_text else "(no success replays)",
            current_generator_block=gen_block,
        )
        return filled
