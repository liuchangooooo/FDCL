"""
ACGS 验证 2：批次汇总验证（不调用真实 LLM）

目标：
1. 设 evaluation_interval=10，收集 10 个 episode 后触发一次汇总。
2. 打印完整 _build_acgs_prompt() 输出。
3. 检查 prompt 结构和关键内容是否完整。

说明：
- 该脚本不会修改训练主代码。
- 通过 monkey patch workspace._trigger_acgs_evolve()，只构建/打印/校验 prompt，不请求 LLM。
"""

import copy
import pathlib
import re
import sys

import numpy as np
from omegaconf import OmegaConf

ROOT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from DIVO.workspace.rl_workspace.td3_curriculum_workspace import TD3CurriculumWorkspace


def _configure_cfg(config_path: pathlib.Path):
    cfg = OmegaConf.load(config_path)

    # 仅做验证，不跑训练循环，不写 wandb
    cfg.log = False
    cfg.training.device = "cpu"
    cfg.device = "cpu"
    cfg.evaluator.output_dir = str(ROOT_DIR / "outputs" / "acgs_prompt_validation" / "eval")

    # ACGS 批次触发参数
    cfg.curriculum.enable_acgs_loop = True
    cfg.curriculum.evaluation_interval = 10
    cfg.curriculum.frame_buffer_size = 5
    cfg.curriculum.max_failure_replays = 20
    cfg.curriculum.max_success_replays = 5

    # 关闭 LLM，确保不发起外部请求
    cfg.curriculum.use_llm_obstacles = False
    cfg.curriculum.api_key = None

    return cfg


def _decode_tblock_pose(obs, desk_size=0.25):
    x = obs[0, 0] * desk_size
    y = obs[0, 1] * desk_size
    theta = np.arctan2(obs[0, 3], obs[0, 2])
    return [x, y, theta]


def _validate_replay_payload(replay, frame_buffer_size):
    required_keys = [
        "episode_id",
        "start_pose",
        "obstacle_config",
        "termination",
        "collision_detail",
        "reward",
        "steps",
        "last_k_frames",
    ]
    for key in required_keys:
        assert key in replay, f"回放缺少字段: {key}"

    frames = replay["last_k_frames"]
    assert len(frames) <= frame_buffer_size, "last_k_frames 超出 frame_buffer_size"

    termination = replay.get("termination", "timeout")
    strict_bound = 0.25
    fall_bound = 0.40

    for frame in frames:
        assert "tblock" in frame and "rod" in frame and "q_value" in frame, "帧字段不完整"
        tx, ty, _ = frame["tblock"]
        rx, ry = frame["rod"]

        # 非 fall 严格检查，fall 允许越界但不能离谱
        if termination == "fall":
            assert -fall_bound <= tx <= fall_bound, f"fall帧 tblock x 异常: {tx}"
            assert -fall_bound <= ty <= fall_bound, f"fall帧 tblock y 异常: {ty}"
        else:
            assert -strict_bound <= tx <= strict_bound, f"tblock x 超范围: {tx}"
            assert -strict_bound <= ty <= strict_bound, f"tblock y 超范围: {ty}"

        # rod 位置允许略超桌面边缘
        assert -0.35 <= rx <= 0.35, f"rod x 超范围: {rx}"
        assert -0.35 <= ry <= 0.35, f"rod y 超范围: {ry}"

    if termination == "collision":
        detail = replay.get("collision_detail")
        assert detail is not None, "collision 回放缺少 collision_detail"
        assert detail.get("type") in ("rod", "tblock"), f"非法碰撞类型: {detail}"
        assert detail.get("obstacle_id") in (0, 1), f"非法 obstacle_id: {detail}"


def _validate_prompt_text(prompt, ws, batch_stats_snapshot, failure_replays_snapshot, success_replays_snapshot):
    # 1) 结构检查
    assert "=== 批次原始数据" in prompt, "缺少批次头"
    assert "【批次统计】" in prompt, "缺少批次统计段"
    assert "【当前障碍物生成策略】" in prompt, "缺少策略段"
    assert "请基于以上数据：" in prompt, "缺少指令段"

    # 2) 统计检查
    total = sum(batch_stats_snapshot.values())
    assert total == ws.evaluation_interval, f"统计总数应为 {ws.evaluation_interval}，实际 {total}"
    for key, value in batch_stats_snapshot.items():
        assert value >= 0, f"统计项 {key} 不能为负数"

    # 3) 失败回放字段检查
    for replay in failure_replays_snapshot:
        _validate_replay_payload(replay, ws.frame_buffer_size)

    # 4) 成功回放段检查
    has_success = len(success_replays_snapshot) > 0
    if has_success:
        assert "【成功回放（对照）】" in prompt, "有成功回放但 prompt 缺少成功段"
    else:
        assert "【成功回放（对照）】" not in prompt, "无成功回放但 prompt 包含成功段"

    # 5) 当前策略段检查
    if ws.topology_generator_code:
        assert "```python" in prompt, "有策略代码但 prompt 未包含代码块"
    else:
        assert "使用随机生成（无 LLM 策略）" in prompt, "无策略代码时未显示随机生成说明"

    # 6) 帧格式文本检查
    frame_line_pattern = re.compile(r"step\d+: T\([^\)]+\)\s+rod\([^\)]+\)(\s+Q=.*)?")
    frame_lines = [line.strip() for line in prompt.splitlines() if line.strip().startswith("step")]
    for line in frame_lines:
        assert frame_line_pattern.match(line), f"帧文本格式异常: {line}"


def main():
    config_path = ROOT_DIR / "config" / "pusht" / "train_td3_curriculum_mujoco_obstacle.yaml"
    cfg = _configure_cfg(config_path)

    print("=" * 70)
    print("ACGS 验证 2：批次汇总验证（仅构建与打印 prompt）")
    print("=" * 70)
    print(f"config: {config_path}")

    ws = TD3CurriculumWorkspace(cfg=cfg, output_dir=str(ROOT_DIR / "outputs" / "acgs_prompt_validation"))

    captured = {
        "trigger_count": 0,
        "prompt": None,
        "batch_stats": None,
        "failure_replays": None,
        "success_replays": None,
    }

    def trigger_without_llm(self):
        print("\n[ACGS-MOCK] evaluation_interval reached, build prompt without LLM...")

        # 在 reset 前抓取快照用于校验
        batch_stats_snapshot = copy.deepcopy(self._batch_stats)
        failure_replays_snapshot = copy.deepcopy(self._batch_failure_replays)
        success_replays_snapshot = copy.deepcopy(self._batch_success_replays)

        prompt = self._build_acgs_prompt()

        captured["trigger_count"] += 1
        captured["prompt"] = prompt
        captured["batch_stats"] = batch_stats_snapshot
        captured["failure_replays"] = failure_replays_snapshot
        captured["success_replays"] = success_replays_snapshot

        print("\n" + "-" * 70)
        print("[ACGS-MOCK] FULL PROMPT START")
        print("-" * 70)
        print(prompt)
        print("-" * 70)
        print("[ACGS-MOCK] FULL PROMPT END")
        print("-" * 70)

        prompt_len = len(prompt)
        print(f"[ACGS-MOCK] Prompt length: {prompt_len} chars")
        if 500 <= prompt_len <= 2000:
            print("[ACGS-MOCK] Prompt length check: PASS (500-2000)")
        else:
            print("[ACGS-MOCK] Prompt length check: WARN (not in 500-2000)")

        self._reset_batch()

    # monkey patch：仅脚本内生效
    ws._trigger_acgs_evolve = trigger_without_llm.__get__(ws, TD3CurriculumWorkspace)

    num_episodes = 10
    max_steps = 10

    print(f"\n[RUN] Collecting {num_episodes} episodes...")

    for ep in range(1, num_episodes + 1):
        obs = ws.env.reset()
        ws._on_episode_start(obs, tblock_pose=None)

        done = False
        step_count = 0
        episode_reward = 0.0
        info = {"termination": "timeout", "collision_detail": None}

        while not done:
            action = np.random.uniform(-1, 1, size=ws.env.action_dim)
            next_obs, reward, done, info = ws.env.step(action)

            # 本验证只验证 prompt 打包流程，Q 可用随机数或 None。
            q_value = float(np.random.uniform(-5.0, 0.0))
            ws._record_frame(next_obs, q_value=q_value)

            episode_reward += float(reward)
            step_count += 1
            obs = next_obs

            if step_count >= max_steps - 1:
                done = True

        ws._on_episode_end(episode_reward=episode_reward, episode_length=step_count, info=info)

        print(
            f"  episode {ep:02d}: termination={info.get('termination')}, "
            f"steps={step_count}, reward={episode_reward:.2f}"
        )

    assert captured["trigger_count"] == 1, f"应触发 1 次汇总，实际 {captured['trigger_count']}"
    assert captured["prompt"] is not None, "未捕获到 prompt"

    _validate_prompt_text(
        prompt=captured["prompt"],
        ws=ws,
        batch_stats_snapshot=captured["batch_stats"],
        failure_replays_snapshot=captured["failure_replays"],
        success_replays_snapshot=captured["success_replays"],
    )

    print("\n[CHECK] 批次汇总验证通过 ✓")
    print(f"[CHECK] termination stats snapshot: {captured['batch_stats']}")
    print(f"[CHECK] failure replays: {len(captured['failure_replays'])}")
    print(f"[CHECK] success replays: {len(captured['success_replays'])}")


if __name__ == "__main__":
    main()
