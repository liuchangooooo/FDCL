if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import ast
import logging
import torch
from torch.nn import functional as F
from omegaconf import OmegaConf
import pathlib
import random
import numpy as np
from collections import deque
from DIVO.workspace.base_workspace import BaseWorkspace
import hydra
import wandb
import tqdm
import time
import copy

from DIVO.RL.component import StateDictReplayBuffer, OrnsteinUhlenbeckProcess, hard_update, soft_update

from DIVO.common.pytorch_util import optimizer_to, dict_to_torch
from DIVO.common.checkpoint_util import TopKCheckpointManager

from DIVO.env import get_env_class
from DIVO.policy import get_policy
from DIVO.critic import get_critic
from DIVO.evaluator import get_evaluator

from DIVO.env.pusht.llm_topology_generator import (
    LLMTopologyGenerator,
    StrategyExecutor,
    build_phase0_prompt_stage_a
)


LOGGER = logging.getLogger(__name__)


# ============================================================================
# 通用工具函数
# ============================================================================

_DIRECTION_LABELS = [
    "right", "upper-right", "up", "upper-left",
    "left", "lower-left", "down", "lower-right",
]
_DIRECTION_LABELS_WITH_STATIONARY = _DIRECTION_LABELS + ["stationary"]

_FAILURE_KEYS = [
    "success",
    "collision_rod_early",
    "collision_rod_mid",    
    "collision_rod_late",
    "collision_tblock_early",
    "collision_tblock_mid",
    "collision_tblock_late",
    "timeout",
    "fall",
]

def quantize_direction(vec_x, vec_y, epsilon=1e-4):
    """
    将 2D 向量量化到 8 方向标签 + stationary。

    Task-agnostic：任何有 2D 工作空间的任务都能用。

    Args:
        vec_x, vec_y: 2D 向量分量
        epsilon: 零向量判定阈值

    Returns:
        str: 9 种标签之一
             "right", "upper-right", "up", "upper-left",
             "left", "lower-left", "down", "lower-right",
             "stationary"
    """
    if np.sqrt(vec_x ** 2 + vec_y ** 2) < epsilon:
        return "stationary"
    angle = np.arctan2(vec_y, vec_x)  # [-pi, pi]
    idx = int(np.round(angle / (np.pi / 4))) % 8
    return _DIRECTION_LABELS[idx]


class TD3CurriculumWorkspace(BaseWorkspace):
    """
    TD3 + ACGS 闭环 Workspace

    闭环流程：
    1. 每步采集帧数据 (T-block位姿, rod位置, Q值) 到滑动缓冲区
    2. Episode 结束时记录终止原因、存储失败回放片段
    3. 每 evaluation_interval 个 episode 打包四样数据传给 LLM
    4. LLM 自己分析失败模式并生成新的 generate_obstacles 函数
    """

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        self.device = torch.device(cfg.training.device)

        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # set env (使用 LLM 版本的环境)
        self.env = get_env_class(**cfg.env)
        self.no_obs_env = get_env_class(**cfg.no_obs_env)
        self.unseen_env = get_env_class(**cfg.unseen_env)
        self.action_dim, self.obs_dim = self.env.get_info()

        print("\n [1] Env is set:")

        # 初始化 LLM 障碍物生成器
        self._init_llm_generator(cfg)

        # 初始化 ACGS 数据采集
        self._init_acgs_collection(cfg)

        # configure model
        self.model = get_policy(
            self.env,
            **cfg.policy
        ).to(self.device)
        self.model_target = get_policy(
            self.env,
            **cfg.policy
        ).to(self.device)
        hard_update(self.model_target, self.model)
        
        print("\n [2] Policy is set:")
        print(self.model)
        
        # configure RL
        self.critic = get_critic(**cfg.critic).to(self.device)
        self.critic_target = get_critic(**cfg.critic).to(self.device)
        hard_update(self.critic_target, self.critic)
        
        print("\n [3] Critic is set:")
        print(self.critic)
        
        # set evaluator
        self.evaluator = get_evaluator(**cfg.evaluator)
        
        self.optimizer = hydra.utils.get_class(
            cfg.optimizer._target_)(
            self.model.parameters(), 
            lr=cfg.optimizer.lr)
        self.critic_optimizer = hydra.utils.get_class(
            cfg.critic_optimizer._target_)(
            self.critic.parameters(), 
            lr=cfg.critic_optimizer.lr)
        
        self.critic_gradient_clip = cfg.rl.critic_gradient_clip
        self.critic_gradient_max_norm = cfg.rl.critic_gradient_max_norm
        self.policy_gradient_clip = cfg.rl.policy_gradient_clip
        self.policy_gradient_max_norm = cfg.rl.policy_gradient_max_norm

        if cfg.rl.add_noise:
            self.random_process = OrnsteinUhlenbeckProcess(
                size=cfg.action_size, 
                theta=0.15, 
                mu=0, 
                sigma=cfg.rl.noise_sigma)

        replay_buffer_args = {
            "obs_dim": (self.obs_dim),
            "action_dim": (self.action_dim),
        }
        
        self.replay_buffer = StateDictReplayBuffer(
            cfg.rl.replay_buffer_size, 
            **replay_buffer_args
        )
        self.global_step = 0
        self.epoch = 0
        self.num_timesteps = 0
        self.gamma = cfg.rl.gamma
        self.save_dir = os.path.join(
            self.output_dir, 
            'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        self.evolve_prompt_log_path = os.path.join(self.output_dir, 'evolve_prompt.log')
    
    def _init_llm_generator(self, cfg):
        """初始化 LLM 拓扑生成器（Stage A）"""
        self.llm_generator = None
        self.executor = None
        self.topology_generator_code = None
        self.obstacle_num = getattr(cfg.env, 'obstacle_num', 2)
        
        curriculum_cfg = getattr(cfg, 'curriculum', None)
        
        # Episode 计数（用于日志）
        self.episode_count = 0
        
        # 对比实验：是否使用 LLM 生成障碍物
        self.use_llm_obstacles = getattr(curriculum_cfg, 'use_llm_obstacles', True) if curriculum_cfg else False
        
        if not self.use_llm_obstacles:
            print("\n [1.1] LLM obstacle generation DISABLED (using random obstacles)")
            print(f" [1.1] This is a baseline experiment")
            return
        
        curriculum_cfg = getattr(cfg, 'curriculum', None)
        if curriculum_cfg is None:
            print("\n [1.1] No curriculum config, LLM generator disabled")
            return
        
        api_type = getattr(curriculum_cfg, 'api_type', 'deepseek')
        api_key = getattr(curriculum_cfg, 'api_key', None)
        model = getattr(curriculum_cfg, 'model', 'deepseek-chat')
        
        if api_key is None:
            api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if api_key:
            try:
                # 初始化 LLM 生成器和执行器
                self.llm_generator = LLMTopologyGenerator(
                    api_type=api_type,
                    api_key=api_key,
                    model=model,
                    temperature=0.7,
                    verbose=False
                )
                self.executor = StrategyExecutor(
                    obstacle_size=0.01,  # 障碍物半边长 0.01m
                    target_pose=[0, 0, -np.pi/4]  # 目标位姿
                )
                print(f"\n [1.1] LLM Topology Generator initialized (Stage A, model: {model})")
                print(f" [1.1] Obstacle generation mode: LLM topology generator (every reset)")
                print(f" [1.1] T-block position: sampled by environment, obstacles generated based on T-block")
            except Exception as e:
                print(f"\n [1.1] LLM init failed: {e}")
                self.use_llm_obstacles = False
        else:
            print("\n [1.1] No API key found, LLM generator disabled")
            self.use_llm_obstacles = False
    
    def _init_topology_generator(self):
        """初始化拓扑生成器（只在训练开始时调用一次）"""
        if not self.use_llm_obstacles:
            return False
        
        if self.llm_generator is None or self.executor is None:
            return False
        
        if self.topology_generator_code is not None:
            return True  # 已经初始化过了
        
        try:
            print(f"\n[Stage A] Generating topology generator code (first time)...")
            
            # 使用一个示例起点生成 prompt
            sample_tblock_pose = self._random_tblock_pose()
            prompt = build_phase0_prompt_stage_a(
                sample_tblock_pose,
                num_obstacles=self.obstacle_num
            )
            
            # 调用 LLM 生成代码
            code = self.llm_generator._call_llm(prompt)
            code = self.llm_generator._extract_code(code)
            
            if code is None:
                print("[Stage A] ❌ Code generation failed, disabling LLM obstacles")
                self.use_llm_obstacles = False
                return False
            
            # 加载到执行器
            if not self.executor.load_topology_generator(code):
                print("[Stage A] ❌ Code loading failed, disabling LLM obstacles")
                self.use_llm_obstacles = False
                return False
            
            self.topology_generator_code = code
            print(f"[Stage A] ✓ Topology generator loaded successfully")
            print(f"[Stage A] Code length: {len(code)} characters")
            return True
            
        except Exception as e:
            print(f"[Stage A] ❌ Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.use_llm_obstacles = False
            return False
    
    def _generate_obstacles_for_reset(self):
        """
        每次 reset 前调用：生成 T-block 位置和障碍物配置

        A（起点）和为 A 生成的障碍物是绑定的整体配置。
        如果物理引擎检测到初始接触（ncon > 0），整体重新生成，不单独换起点。

        流程：
        1. 第一次 reset：让环境随机采样起点 A
        2. 基于 A 生成障碍物
        3. 第二次 reset：强制用 A + 障碍物
        4. 检查 ncon，不通过则从 1 重来

        Returns:
            (obs, tblock_pose): reset 后的观测和 T-block 位置
                               如果不使用 LLM 则返回 (None, None)
        """
        if not self.use_llm_obstacles:
            return None, None

        if self.executor is None or not hasattr(self.env, 'set_obstacle_config'):
            return None, None

        attempt = 0
        while True:
            attempt += 1
            try:
                # Step 1: 第一次 reset，让环境随机采样起点 A
                self.env.clear_obstacle_config()
                obs_temp = self.env.reset()
                tblock_pose_A = self._get_tblock_pose_from_obs(obs_temp)

                # Step 2: 基于 A 生成障碍物（A 和障碍物绑定）
                obstacles = self.executor.generate(tblock_pose_A, self.obstacle_num)

                if not obstacles or len(obstacles) == 0:
                    # 生成为空，无障碍物直接返回
                    self.env.clear_obstacle_config()
                    return obs_temp, tblock_pose_A

                self.env.set_obstacle_config(obstacles)

                # Step 3: 第二次 reset，强制用 A + 障碍物
                obs = self.env.reset(tblock_pos=tblock_pose_A, force_tblock_pos=True)

                # Step 4: 检查 ncon，不通过则整对重新生成
                ncon = self.env.get_ncon()
                if ncon == 0:
                    # 成功：A 和障碍物匹配且无初始接触
                    # 只在关键时刻写日志：第一个 episode、每 1000 episodes、或 evolve 后
                    if self.episode_count == 0 or \
                       self.episode_count % 1000 == 0 or \
                       (hasattr(self, '_just_evolved') and self._just_evolved):
                        log_lines = [
                            f"[LLM] Episode {self.episode_count}: Generated {len(obstacles)} obstacles",
                            f"  T-block: ({tblock_pose_A[0]:.3f}, {tblock_pose_A[1]:.3f}, {np.degrees(tblock_pose_A[2]):.0f}°)",
                        ]
                        for i, ob in enumerate(obstacles):
                            log_lines.append(f"  Obs{i+1}: ({ob['x']:.3f}, {ob['y']:.3f}) - {ob.get('purpose', 'N/A')}")
                        LOGGER.info("\n".join(log_lines))
                        if hasattr(self, '_just_evolved'):
                            self._just_evolved = False
                    return obs, tblock_pose_A

                if attempt % 10 == 0:
                    print(f"[LLM] ⚠ {attempt} attempts, ncon={ncon}, retrying A + obstacles...")

            except Exception as e:
                if attempt % 10 == 0:
                    print(f"[LLM] ⚠ {attempt} attempts, exception: {e}")
    
    # ===== ACGS 数据采集与闭环 =====

    def _log_acgs(self, message: str):
        """同时输出到控制台与 train.log（Hydra logging）。"""
        print(message)
        LOGGER.info(message)

    def _init_acgs_collection(self, cfg):
        """初始化 ACGS 数据采集基础设施"""
        curriculum_cfg = getattr(cfg, 'curriculum', None)

        # 配置参数
        self.evaluation_interval = getattr(curriculum_cfg, 'evaluation_interval', 5000) if curriculum_cfg else 5000
        self.frame_buffer_size = getattr(curriculum_cfg, 'frame_buffer_size', 5) if curriculum_cfg else 5
        self.enable_acgs_loop = getattr(curriculum_cfg, 'enable_acgs_loop', True) if curriculum_cfg else False
        self.max_failure_replays = getattr(curriculum_cfg, 'max_failure_replays', 20) if curriculum_cfg else 20
        self.max_success_replays = getattr(curriculum_cfg, 'max_success_replays', 5) if curriculum_cfg else 5

        # 条件触发参数
        self.warmup_success_threshold = getattr(curriculum_cfg, 'warmup_success_threshold', 0.15) if curriculum_cfg else 0.15
        self.warmup_min_episodes = getattr(curriculum_cfg, 'warmup_min_episodes', 5000) if curriculum_cfg else 5000
        self.first_evolve_episode = getattr(curriculum_cfg, 'first_evolve_episode', self.warmup_min_episodes) if curriculum_cfg else self.warmup_min_episodes
        self.first_evolve_episode = max(1, int(self.first_evolve_episode))
        self.success_rate_high = getattr(curriculum_cfg, 'success_rate_high', 0.75) if curriculum_cfg else 0.75
        self.success_rate_low = getattr(curriculum_cfg, 'success_rate_low', 0.30) if curriculum_cfg else 0.30
        self.plateau_threshold = getattr(curriculum_cfg, 'plateau_threshold', 0.03) if curriculum_cfg else 0.03
        self.plateau_window = getattr(curriculum_cfg, 'plateau_window', 3) if curriculum_cfg else 3
        self.evolve_cooldown = getattr(curriculum_cfg, 'evolve_cooldown', 10000) if curriculum_cfg else 10000

        # Level 2 诊断参数
        self.max_typed_replays = getattr(curriculum_cfg, 'max_typed_replays', 50) if curriculum_cfg else 50
        self.direction_confidence_threshold = getattr(curriculum_cfg, 'direction_confidence_threshold', 0.3) if curriculum_cfg else 0.3
        self.max_steps = max(1, int(getattr(cfg, 'max_steps', 10)))
        self.primary_task_body = getattr(curriculum_cfg, 'primary_task_body', 'tblock') if curriculum_cfg else 'tblock'

        # task_reference_point 优先级: env 接口 > cfg > 默认 (0,0)
        ref_from_env = None
        if hasattr(self.env, 'get_task_reference_point'):
            try:
                ref_from_env = self.env.get_task_reference_point()
            except Exception:
                ref_from_env = None
        _ref = ref_from_env
        if _ref is None:
            _ref = getattr(curriculum_cfg, 'task_reference_point', None) if curriculum_cfg else None
        if _ref is not None:
            self.task_reference_point = np.array(_ref, dtype=np.float64)
        else:
            self.task_reference_point = np.array([0.0, 0.0])

        # 每 episode 的帧缓冲区（滑动窗口，只保留最后 K 帧）
        self._frame_buffer = deque(maxlen=self.frame_buffer_size)

        # 每 episode 的状态
        self._episode_start_pose = None
        self._episode_obstacle_config = None
        self._episode_termination = 'timeout'
        self._episode_collision_detail = None

        # 批次统计
        self._batch_stats = {'success': 0, 'collision': 0, 'timeout': 0, 'fall': 0}
        self._batch_failure_counts = {k: 0 for k in _FAILURE_KEYS}
        self._batch_episode_count = 0

        # 批次回放存储
        self._batch_failure_replays = []
        self._batch_success_replays = []

        # 按 failure_key 细分的回放存储（Level 2 诊断用）
        self._batch_typed_failure_replays = {}

        # evolve 历史
        self._evolve_count = 0
        self._total_episode_count = 0  # 全局 episode 计数（不重置）
        self._success_rate_history = []  # 每个批次的成功率历史
        self._just_evolved = False  # 标志：刚刚 evolve 过，下一个 episode 打印障碍物
        self._warmup_completed = False  # warmup 是否已结束（固定首轮触发）
        self._first_evolve_triggered = False  # 首次 evolve 是否已经触发（无论成功失败）
        self._last_evolve_episode = -self.evolve_cooldown  # 上次 evolve 时的 episode（初始化为可立即触发）

        # 全局历史记录（论文分析用，不随 batch 重置）
        self._failure_vector_history = []   # 每个 batch 一条：failure distribution + diagnosis
        self._acgs_evolve_history = []      # 每次 evolve 一条：预留 accepted/rejected

        if self.enable_acgs_loop:
            print(f"\n [1.2] ACGS loop enabled: eval every {self.evaluation_interval} episodes, "
                  f"frame buffer K={self.frame_buffer_size}")
            print(f"        first evolve: fixed at >= {self.first_evolve_episode} episodes, "
                  f"high={self.success_rate_high}, low={self.success_rate_low}, "
                  f"plateau_δ={self.plateau_threshold}, plateau_window={self.plateau_window}, "
                  f"cooldown={self.evolve_cooldown}")
            print(f"        Level 2: primary_body={self.primary_task_body}, "
                  f"ref_point={self.task_reference_point.tolist()}, "
                f"phase_split=(1/3,2/3), "
                f"direction_conf_th={self.direction_confidence_threshold}, "
                  f"max_typed_replays={self.max_typed_replays}")
        else:
            print(f"\n [1.2] ACGS loop disabled")

    def _get_tblock_pose_from_obs(self, obs):
        """从归一化 obs 解码 T-block 位姿 [x, y, theta]"""
        desk_size = self.env.task._desk_size
        x = obs[0, 0] * desk_size
        y = obs[0, 1] * desk_size
        theta = np.arctan2(obs[0, 3], obs[0, 2])
        return [x, y, theta]

    def _get_rod_position(self):
        """从物理引擎获取 rod 当前位置 [x, y]"""
        try:
            raw_state = self.env._observation_updater.get_observation()
            rod_pos = raw_state['unnamed_model/joint_positions'][0]
            return [float(rod_pos[0]), float(rod_pos[1])]
        except Exception:
            return [0.0, 0.0]

    def _record_frame(self, obs, q_value=None):
        """每步调用：记录一帧到滑动缓冲区"""
        tblock_pose = self._get_tblock_pose_from_obs(obs)
        rod_pos = self._get_rod_position()
        self._frame_buffer.append({
            'tblock': tblock_pose,
            'rod': rod_pos,
            'q_value': q_value,
        })

    def _on_episode_start(self, obs, tblock_pose=None):
        """Episode 开始时调用：记录起点，清空缓冲区"""
        self._frame_buffer.clear()
        self._episode_termination = 'timeout'
        self._episode_collision_detail = None

        if tblock_pose is not None:
            self._episode_start_pose = list(tblock_pose)
        else:
            self._episode_start_pose = self._get_tblock_pose_from_obs(obs)

        # 记录当前障碍物配置
        if hasattr(self.env, 'get_obstacle_positions'):
            self._episode_obstacle_config = self.env.get_obstacle_positions()
        elif hasattr(self.env.task, 'llm_obstacle_config') and self.env.task.llm_obstacle_config is not None:
            self._episode_obstacle_config = list(self.env.task.llm_obstacle_config)
        else:
            self._episode_obstacle_config = None

    def _on_episode_end(self, episode_reward, episode_length, info):
        """Episode 结束时调用：打包回放片段，更新批次统计"""
        termination = info.get('termination', self._episode_termination)
        collision_detail = info.get('collision_detail', self._episode_collision_detail)

        # 更新批次统计
        self._batch_stats[termination] = self._batch_stats.get(termination, 0) + 1
        self._batch_episode_count += 1
        self._total_episode_count += 1

        # 构造 failure_key（Level 2 细分标签）
        failure_key = self._build_failure_key(termination, collision_detail, episode_length)
        self._batch_failure_counts[failure_key] = self._batch_failure_counts.get(failure_key, 0) + 1

        # 打包回放片段
        replay = {
            'episode_id': self.episode_count,
            'start_pose': self._episode_start_pose,
            'obstacle_config': self._episode_obstacle_config,
            'termination': termination,
            'collision_detail': collision_detail,
            'failure_key': failure_key,
            'reward': episode_reward,
            'steps': episode_length,
            'last_k_frames': list(self._frame_buffer),
        }

        if termination == 'success':
            self._batch_success_replays.append(replay)
            if len(self._batch_success_replays) > self.max_success_replays:
                self._batch_success_replays.pop(0)
        else:
            self._batch_failure_replays.append(replay)
            if len(self._batch_failure_replays) > self.max_failure_replays:
                self._batch_failure_replays.pop(0)

            # 按 failure_key 存入 typed replay buffer（Level 2 诊断用）
            bucket = self._batch_typed_failure_replays.setdefault(failure_key, [])
            bucket.append(replay)
            if len(bucket) > self.max_typed_replays:
                bucket.pop(0)

        # 检查是否到达 evaluation_interval
        if self.enable_acgs_loop and self._batch_episode_count >= self.evaluation_interval:
            self._trigger_acgs_evolve()

    def _get_collision_phase(self, episode_length):
        """按 episode 进度将碰撞划分为 early/mid/late。"""
        progress = float(episode_length) / float(max(self.max_steps, 1))
        if progress < (1.0 / 3.0):
            return 'early'
        if progress < (2.0 / 3.0):
            return 'mid'
        return 'late'

    def _build_failure_key(self, termination, collision_detail, episode_length):
        """生成细粒度 failure key。"""
        if termination == 'success':
            return 'success'
        if termination == 'collision':
            default_body = self.primary_task_body if self.primary_task_body in ('rod', 'tblock') else 'tblock'
            body = default_body
            if collision_detail and collision_detail.get('type') in ('rod', 'tblock'):
                body = collision_detail['type']
            phase = self._get_collision_phase(episode_length)
            return f'collision_{body}_{phase}'
        if termination in ('timeout', 'fall'):
            return termination
        return termination

    def _empty_direction_counts(self):
        return {label: 0 for label in _DIRECTION_LABELS_WITH_STATIONARY}

    def _extract_body_xy_from_frame(self, frame, body):
        """从单帧中提取指定 body 的二维坐标。"""
        if frame is None:
            return None

        data = frame.get('rod') if body == 'rod' else frame.get('tblock')
        if data is None or len(data) < 2:
            return None

        try:
            return np.array([float(data[0]), float(data[1])], dtype=np.float64)
        except Exception:
            return None

    def _resolve_analysis_body(self, dominant_type):
        """Level 2 分析默认用 primary_task_body；rod 碰撞时强制用 rod。"""
        body = self.primary_task_body if self.primary_task_body in ('rod', 'tblock') else 'tblock'
        if isinstance(dominant_type, str) and dominant_type.startswith('collision_rod_'):
            return 'rod'
        return body

    def _compute_failure_vector(self):
        """Level 1：将 batch 失败计数归一化为分布并给出主导失败类型。"""
        counts = {k: int(self._batch_failure_counts.get(k, 0)) for k in _FAILURE_KEYS}

        # 保留潜在扩展键，避免数据丢失
        for key, value in self._batch_failure_counts.items():
            if key not in counts:
                counts[key] = int(value)

        total = sum(counts.values())
        if total > 0:
            distribution = {k: v / total for k, v in counts.items()}
        else:
            distribution = {k: 0.0 for k in counts}

        failure_only = {k: v for k, v in counts.items() if k != 'success'}
        if sum(failure_only.values()) > 0:
            dominant_type = max(failure_only, key=failure_only.get)
        else:
            dominant_type = 'success'

        return {
            'counts': counts,
            'distribution': distribution,
            'total': total,
            'dominant_type': dominant_type,
            'dominant_failure_type': dominant_type,
        }

    def _compute_failure_diagnosis(self, failure_vector_result):
        """Level 2：根据主导失败类型计算 failure_region 与 behavior_bias。"""
        dominant_type = failure_vector_result.get('dominant_type', 'success')
        analysis_body = self._resolve_analysis_body(dominant_type)
        reference_point = np.array(self.task_reference_point, dtype=np.float64)

        replays = list(self._batch_typed_failure_replays.get(dominant_type, []))
        if not replays:
            replays = [
                r for r in self._batch_failure_replays
                if r.get('failure_key') == dominant_type
            ]

        sample_count = len(replays)

        # Step B: failure_region
        region_counts = self._empty_direction_counts()
        valid_region_samples = 0
        for replay in replays:
            frames = replay.get('last_k_frames') or []
            pos = None
            if frames:
                pos = self._extract_body_xy_from_frame(frames[-1], analysis_body)
            if pos is None:
                start_pose = replay.get('start_pose')
                if start_pose is not None and len(start_pose) >= 2:
                    pos = np.array([float(start_pose[0]), float(start_pose[1])], dtype=np.float64)

            if pos is None:
                continue

            vec = pos - reference_point
            label = quantize_direction(vec[0], vec[1])
            region_counts[label] += 1
            valid_region_samples += 1

        if valid_region_samples > 0:
            region_label = max(region_counts, key=region_counts.get)
            region_conf = region_counts[region_label] / valid_region_samples
        else:
            region_label = 'none'
            region_conf = 0.0

        # Step C1: terminal_motion_direction
        motion_counts = self._empty_direction_counts()
        valid_motion_samples = 0
        for replay in replays:
            frames = replay.get('last_k_frames') or []
            if len(frames) < 2:
                continue

            start_pos = self._extract_body_xy_from_frame(frames[0], analysis_body)
            end_pos = self._extract_body_xy_from_frame(frames[-1], analysis_body)
            if start_pos is None or end_pos is None:
                continue

            delta = end_pos - start_pos
            label = quantize_direction(delta[0], delta[1])
            motion_counts[label] += 1
            valid_motion_samples += 1

        if valid_motion_samples > 0:
            motion_peak_label = max(motion_counts, key=motion_counts.get)
            motion_peak_conf = motion_counts[motion_peak_label] / valid_motion_samples
        else:
            motion_peak_label = 'none'
            motion_peak_conf = 0.0

        # Step C2: initial_relative_direction
        start_counts = self._empty_direction_counts()
        valid_start_samples = 0
        for replay in replays:
            start_pose = replay.get('start_pose')
            if start_pose is None or len(start_pose) < 2:
                continue

            delta = np.array([float(start_pose[0]), float(start_pose[1])], dtype=np.float64) - reference_point
            label = quantize_direction(delta[0], delta[1])
            start_counts[label] += 1
            valid_start_samples += 1

        if valid_start_samples > 0:
            start_peak_label = max(start_counts, key=start_counts.get)
            start_peak_conf = start_counts[start_peak_label] / valid_start_samples
        else:
            start_peak_label = 'none'
            start_peak_conf = 0.0

        # 优先终端运动方向，不明显时回退到起点方向
        if motion_peak_conf > self.direction_confidence_threshold:
            behavior_source = 'terminal_motion'
            behavior_label = motion_peak_label
            behavior_conf = motion_peak_conf
        elif start_peak_conf > self.direction_confidence_threshold:
            behavior_source = 'initial_direction'
            behavior_label = start_peak_label
            behavior_conf = start_peak_conf
        else:
            behavior_source = 'none'
            behavior_label = 'no clear directional bias'
            behavior_conf = max(motion_peak_conf, start_peak_conf)

        # 诊断可靠性分级：用于 prompt 门控，避免低样本噪声误导 LLM
        reliability = 'weak'
        if sample_count >= 12 and (region_conf >= 0.45 or behavior_conf >= 0.45):
            reliability = 'strong'
        elif sample_count >= 6 and (region_conf >= 0.30 or behavior_source != 'none'):
            reliability = 'medium'
        is_reliable = reliability in ('strong', 'medium')

        self._log_acgs(
            f"[ACGS][Diag] dominant={dominant_type}, samples={sample_count}, "
            f"motion_peak_confidence={motion_peak_conf:.3f}, "
            f"start_peak_confidence={start_peak_conf:.3f}, "
            f"chosen_source={behavior_source}"
        )

        return {
            'dominant_failure_type': dominant_type,
            'reliability': reliability,
            'is_reliable': is_reliable,
            'diagnosis_reliability': {
                'label': reliability,
                'is_reliable': is_reliable,
            },
            'reference_point': reference_point.tolist(),
            'failure_region': {
                'label': region_label,
                'confidence': region_conf,
                'counts': region_counts,
            },
            'behavior_bias': {
                'label': behavior_label,
                'confidence': behavior_conf,
                'source': behavior_source,
                'motion_counts': motion_counts,
                'start_counts': start_counts,
                'motion_peak_confidence': motion_peak_conf,
                'start_peak_confidence': start_peak_conf,
            },
            'behavior_source': behavior_source,
            'sample_count': sample_count,
        }

    def _append_failure_vector_history(self, failure_vector_result, diagnosis, should_evolve, reason):
        """每个 batch 都记录一条 failure history（无论是否触发 evolve）。"""
        history_item = {
            'time': time.time(),
            'total_episode_count': self._total_episode_count,
            'batch_episode_count': self._batch_episode_count,
            'batch_stats': copy.deepcopy(self._batch_stats),
            'failure_counts': copy.deepcopy(failure_vector_result.get('counts', {})),
            'failure_distribution': copy.deepcopy(failure_vector_result.get('distribution', {})),
            'dominant_failure_type': failure_vector_result.get('dominant_type', 'success'),
            'diagnosis': copy.deepcopy(diagnosis),
            'sample_count': diagnosis.get('sample_count', 0),
            'behavior_source': diagnosis.get('behavior_source', 'none'),
            'should_evolve': bool(should_evolve),
            'reason': reason,
        }
        self._failure_vector_history.append(history_item)

    def _append_evolve_prompt_log(self, prompt, reason, fv_result=None, diagnosis=None):
        """将每次 evolve 传给 LLM 的完整 prompt 追加写入日志文件。"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            dominant = 'unknown'
            if fv_result is not None:
                dominant = fv_result.get('dominant_failure_type', fv_result.get('dominant_type', 'unknown'))
            reliability = 'none'
            if diagnosis is not None:
                reliability = diagnosis.get('reliability', 'unknown')

            header_lines = [
                "\n" + "=" * 100,
                f"[EVOLVE PROMPT] time={timestamp}",
                f"episode_total={self._total_episode_count}, batch_episodes={self._batch_episode_count}, evolve_count={self._evolve_count}",
                f"reason={reason}",
                f"dominant_failure_type={dominant}, diagnosis_reliability={reliability}",
                "-" * 100,
            ]

            with open(self.evolve_prompt_log_path, 'a', encoding='utf-8') as f:
                f.write("\n".join(header_lines))
                f.write("\n")
                f.write(prompt)
                if not prompt.endswith("\n"):
                    f.write("\n")
                f.write("=" * 100)
                f.write("\n")
        except Exception as e:
            LOGGER.warning(f"[ACGS] Failed to write evolve prompt log: {e}")

    def _format_replay_for_llm(self, replay):
        """将单条回放片段格式化为 LLM 可读的文本"""
        lines = []
        term = replay['termination']
        detail = replay.get('collision_detail')

        header = f"episode #{replay['episode_id']}, {term}"
        if term == 'collision' and detail:
            header += f", {detail['type']}撞障碍物#{detail['obstacle_id']}"
        header += f", step={replay['steps']}, reward={replay['reward']:.1f}"
        lines.append(f"[{header}]")

        sp = replay['start_pose']
        if sp:
            lines.append(f"  起点: T-block({sp[0]:.3f}, {sp[1]:.3f}, {np.degrees(sp[2]):.0f}°)")

        obs_cfg = replay['obstacle_config']
        if obs_cfg:
            obs_str = ", ".join([f"({{x:{o['x']:.3f}, y:{o['y']:.3f}}})" for o in obs_cfg])
            lines.append(f"  障碍物: [{obs_str}]")

        frames = replay['last_k_frames']
        if frames:
            lines.append(f"  最后 {len(frames)} 帧:")
            for i, f in enumerate(frames):
                t = f['tblock']
                r = f['rod']
                q_str = f" Q={f['q_value']:.2f}" if f['q_value'] is not None else ""
                step_idx = replay['steps'] - len(frames) + i + 1
                lines.append(
                    f"    step{step_idx}: T({t[0]:.3f},{t[1]:.3f},{np.degrees(t[2]):.0f}°) "
                    f"rod({r[0]:.3f},{r[1]:.3f}){q_str}"
                )

        return "\n".join(lines)

    def _extract_clean_generate_obstacles(self):
        """
        提取并清洗当前 generate_obstacles 函数：
        - 只保留 generate_obstacles 完整函数本体
        - 去掉注释（AST 反解天然去注释）
        - 去掉 docstring
        """
        if not self.topology_generator_code:
            return None

        def _strip_comments_and_docstring_from_function_source(function_source):
            """在不依赖 ast.unparse 的情况下，对函数源码做最小清洗。"""
            if not function_source:
                return None

            try:
                parsed = ast.parse(function_source)
                if not parsed.body or not isinstance(parsed.body[0], ast.FunctionDef):
                    return function_source.strip()

                fn = parsed.body[0]
                doc_start = None
                doc_end = None
                if fn.body and isinstance(fn.body[0], ast.Expr):
                    value = fn.body[0].value
                    is_docstring = (
                        isinstance(value, ast.Str) or
                        (isinstance(value, ast.Constant) and isinstance(value.value, str))
                    )
                    if is_docstring:
                        doc_start = fn.body[0].lineno
                        doc_end = getattr(fn.body[0], 'end_lineno', doc_start)

                function_source_wo_doc = function_source
                if doc_start is not None and doc_end is not None:
                    src_lines = function_source.splitlines()
                    function_source_wo_doc = "\n".join(src_lines[:doc_start - 1] + src_lines[doc_end:])

                import io
                import tokenize

                output_tokens = []
                token_stream = tokenize.generate_tokens(io.StringIO(function_source_wo_doc).readline)
                for token in token_stream:
                    token_type = token.type

                    if token_type == tokenize.COMMENT:
                        continue

                    output_tokens.append(token)

                cleaned = tokenize.untokenize(output_tokens)
                cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines()).strip()
                return cleaned if cleaned else function_source_wo_doc.strip()

            except Exception:
                return function_source.strip()

        try:
            module = ast.parse(self.topology_generator_code)
            raw_func_node = None

            for node in module.body:
                if isinstance(node, ast.FunctionDef) and node.name == 'generate_obstacles':
                    raw_func_node = node
                    break

            if raw_func_node is None:
                return None

            func_node = copy.deepcopy(raw_func_node)

            # 移除函数级 docstring，避免把历史语义带入下一轮 prompt。
            if func_node.body and isinstance(func_node.body[0], ast.Expr):
                doc_node = func_node.body[0].value
                is_docstring = (
                    isinstance(doc_node, ast.Str) or
                    (isinstance(doc_node, ast.Constant) and isinstance(doc_node.value, str))
                )
                if is_docstring:
                    func_node.body = func_node.body[1:]

            clean_module = ast.Module(body=[func_node], type_ignores=[])
            ast.fix_missing_locations(clean_module)

            if hasattr(ast, 'unparse'):
                return ast.unparse(clean_module).strip()

            # 兼容回退：部分运行环境可能缺少 ast.unparse。
            source_segment = ast.get_source_segment(self.topology_generator_code, raw_func_node)
            if source_segment is None and hasattr(raw_func_node, 'lineno') and hasattr(raw_func_node, 'end_lineno'):
                lines = self.topology_generator_code.splitlines()
                source_segment = "\n".join(lines[raw_func_node.lineno - 1:raw_func_node.end_lineno])

            cleaned_segment = _strip_comments_and_docstring_from_function_source(source_segment)
            if cleaned_segment:
                LOGGER.warning("[ACGS] ast.unparse unavailable, using source fallback for generate_obstacles.")
                return cleaned_segment

            return None

        except Exception as e:
            LOGGER.warning(f"[ACGS] Failed to extract/clean generate_obstacles: {e}")
            return None

    def _build_acgs_prompt(self, reason="", fv_result=None, diagnosis=None):
        """将批次数据与失败诊断打包成 A-I 结构化 prompt。"""
        lines = []

        # B. 粗粒度 batch 统计（保留 4 维）
        total = sum(self._batch_stats.values())
        success_rate = self._batch_stats['success'] / max(total, 1)
        collision_rate = self._batch_stats['collision'] / max(total, 1)
        fall_rate = self._batch_stats['fall'] / max(total, 1)
        timeout_rate = self._batch_stats['timeout'] / max(total, 1)

        lines.append(f"=== 批次原始数据 (最近 {total} episodes) ===\n")
        lines.append("【批次统计（粗粒度）】")
        lines.append(
            f"成功 {self._batch_stats['success']} ({success_rate:.1%}), "
            f"碰撞 {self._batch_stats['collision']} ({collision_rate:.1%}), "
            f"超时 {self._batch_stats['timeout']} ({timeout_rate:.1%}), "
            f"掉落 {self._batch_stats['fall']} ({fall_rate:.1%})\n"
        )

        # C. 细粒度失败分布（新增 9 维）
        dominant_type = 'unknown'
        if fv_result is not None:
            lines.append("【细粒度失败分布】")
            distribution = fv_result.get('distribution', {})
            counts = fv_result.get('counts', {})
            for key in _FAILURE_KEYS:
                ratio = float(distribution.get(key, 0.0))
                cnt = int(counts.get(key, 0))
                lines.append(f"- {key}: {ratio:.1%} ({cnt})")
            dominant_type = fv_result.get('dominant_failure_type', fv_result.get('dominant_type', 'unknown'))
            lines.append(f"- dominant_failure_type: {dominant_type}")
            lines.append("")

        # D. 失败诊断（新增，带门控）
        diag_reliable = False
        if diagnosis is not None:
            diag_rel = diagnosis.get('diagnosis_reliability', {})
            reliability = diagnosis.get('reliability', diag_rel.get('label', 'weak'))
            diag_reliable = bool(diagnosis.get('is_reliable', diag_rel.get('is_reliable', False)))
            lines.append("【失败诊断（Level 2）】")
            lines.append(f"- reliability: {reliability}")
            if reliability in ('strong', 'medium'):
                region = diagnosis.get('failure_region', {})
                bias = diagnosis.get('behavior_bias', {})
                bias_source = bias.get('source', diagnosis.get('behavior_source', 'none'))
                lines.append(
                    f"- failure_region: {region.get('label', 'none')} "
                    f"(confidence={float(region.get('confidence', 0.0)):.2f})"
                )
                lines.append(
                    f"- behavior_bias: {bias.get('label', 'none')} "
                    f"(confidence={float(bias.get('confidence', 0.0)):.2f}, source={bias_source})"
                )
                lines.append(f"- sample_count: {int(diagnosis.get('sample_count', 0))}")
            else:
                lines.append("- The current failure pattern is not spatially concentrated enough for precise targeting.")
                lines.append("- Focus on dominant failure type and overall failure distribution conservatively.")
                lines.append(f"- sample_count: {int(diagnosis.get('sample_count', 0))}")
            lines.append("")

        # E. 调整指令（基于 reason + diagnosis 的三档模板）
        lines.append("【调整指令（失败驱动三档模板）】")
        lines.append(f"- trigger_reason: {reason if reason else 'unspecified'}")
        lines.append(f"- dominant_failure_type_for_adjustment: {dominant_type}")
        lines.append("- 全局约束: 每个生成出来的布局都必须保持可解，不能把通往目标的可行路径完全堵死。")
        lines.append("- 全局约束: 不要生成明显无法完成的布局。")
        lines.append("- 全局约束: 不要无差别地整体加难；这轮调整应有针对性，只围绕当前主要失败风险做定向修改。")

        reason_norm = (reason or "").lower()
        is_too_easy = ('too easy' in reason_norm) or ('too_easy' in reason_norm)
        is_too_hard = (
            ('too hard' in reason_norm)
            or ('too_hard' in reason_norm)
            or ('impossible' in reason_norm)
            or ('warmup_failed' in reason_norm)
        )
        is_plateau = 'plateau' in reason_norm

        if is_too_easy:
            lines.append("- 模板: Challenge-Up（加压）")
            lines.append(f"- 目标: 针对主导失败类型 {dominant_type} 增加挑战，打破当前惯用策略。")
            lines.append("- 原则: 必须保留可替代通路，避免形成完全封锁。")
            if diag_reliable and diagnosis is not None:
                region = diagnosis.get('failure_region', {})
                bias = diagnosis.get('behavior_bias', {})
                lines.append(
                    f"- 区域提示: 可在失败更集中的 {region.get('label', 'unknown')} 区域附近适度增加压力 "
                    f"(region_conf={float(region.get('confidence', 0.0)):.2f})。"
                )
                lines.append(
                     f"- 行为提示: 可针对当前行为偏置方向 {bias.get('label', 'unknown')} 设计额外挑战 "
                    f"(bias_conf={float(bias.get('confidence', 0.0)):.2f})。"
                )

        elif is_too_hard:
            lines.append("- 模板: Recovery（简化）")
            lines.append(f"- 主目标: 优先降低主导失败类型 {dominant_type}相关的风险，恢复可学习性。")
            lines.append("- 约束: 优先生成更可恢复、更可学习的布局。")
            
            if dominant_type.startswith('collision_') and dominant_type.endswith('_early'):
                
                lines.append("- 约束: 优先降低 early collision risk。")
                lines.append("- 约束: 检查起步阶段近场操作空间是否过于拥挤。")
                lines.append("- 约束: 避免在初始推动阶段形成强正面阻挡。")

            if fall_rate > 0.3:
                lines.append("- 约束: 优先降低 early fall risk。")
                lines.append("- 约束: 优先提升前几步的可控性。")
                lines.append("- 约束: 避免采用会放大立即出界风险的布局方式。")

            if timeout_rate > 0.3:
                lines.append("- 约束: 避免明显闭塞或过度拥挤的布局。")
                lines.append("- 约束: 保留一条清晰可达的目标路径。")

            if diag_reliable and diagnosis is not None:
                region = diagnosis.get('failure_region', {})
                bias = diagnosis.get('behavior_bias', {})
                lines.append(
                    f"- 区域提示: 优先减轻 {region.get('label', 'unknown')} 区域障碍密度 "
                    f"(region_conf={float(region.get('confidence', 0.0)):.2f})。"
                )
                lines.append(
                    f"- 行为提示: 对偏置方向 {bias.get('label', 'unknown')} 减少正面阻断，"
                    f"先保证策略能通过。"
                )

        elif is_plateau:
            lines.append("- 模板: Layout-Shift（换布局）")
            lines.append("- 主目标: 在不显著改变整体难度的前提下，改变障碍物拓扑结构。")
            lines.append("- 约束: 优先进行结构性布局变化，而不是只做小幅坐标微调。")
            lines.append("- 约束: 避免重复当前的主要干扰模式。")

            if diag_reliable and diagnosis is not None:
                region = diagnosis.get('failure_region', {})
                bias = diagnosis.get('behavior_bias', {})
                lines.append(
                    f"- 区域提示：避免重复，不要再次把主干干扰集中在 {region.get('label', 'unknown')}，"
                    f"应引入新的空间干扰分布。"
                )
                lines.append(
                    f"- 行为提示: 针对 {bias.get('label', 'unknown')} 的习惯动作路径，"
                    f"采用不同结构的绕行挑战。"
                )

        else:
            lines.append("- 模板: Balanced（平衡微调）")
            lines.append("- 目标: 依据失败分布做小幅可学习调整，不进行激进改动。")

        lines.append("")

        # F. 失败回放（保留现有）
        if self._batch_failure_replays:
            by_type = {}
            for r in self._batch_failure_replays:
                t = r['termination']
                if t not in by_type:
                    by_type[t] = []
                by_type[t].append(r)

            for term_type, replays in by_type.items():
                lines.append(f"【失败回放 — {term_type}】")
                for r in replays[-2:]:
                    lines.append(self._format_replay_for_llm(r))
                    lines.append("")

        # G. 成功回放（保留现有）
        if self._batch_success_replays:
            lines.append("【成功回放（对照）】")
            for r in self._batch_success_replays[-2:]:
                lines.append(self._format_replay_for_llm(r))
                lines.append("")

        # H. 当前环境生成策略（保留现有）
        lines.append("【当前障碍物生成策略】")
        clean_strategy = self._extract_clean_generate_obstacles()
        if clean_strategy:
            lines.append(f"```python\n{clean_strategy}\n```\n")
        elif self.topology_generator_code:
            lines.append("无法提取 generate_obstacles 函数，已跳过策略展示以避免语义污染。\n")
        else:
            lines.append("使用随机生成（无 LLM 策略）\n")
        return "\n".join(lines)

    def _trigger_acgs_evolve(self):
        """批次结束时检查条件，决定是否触发 LLM evolve"""
        total = sum(self._batch_stats.values())
        success_rate = self._batch_stats['success'] / max(total, 1)
        collision_rate = self._batch_stats['collision'] / max(total, 1)

        self._log_acgs(f"\n[ACGS] Batch complete ({self._batch_episode_count} episodes, "
                   f"total {self._total_episode_count})")
        self._log_acgs(f"[ACGS] Stats: {self._batch_stats}")
        self._log_acgs(f"[ACGS] Success rate: {success_rate:.3f}, Collision rate: {collision_rate:.3f}")
        self._log_acgs(f"[ACGS] Failure replays: {len(self._batch_failure_replays)}, "
                   f"Success replays: {len(self._batch_success_replays)}")

        failure_vector_result = self._compute_failure_vector()
        diagnosis = self._compute_failure_diagnosis(failure_vector_result)

        # 记录成功率历史
        self._success_rate_history.append(success_rate)

        # ---- 条件判断：是否需要 evolve ----
        should_evolve = False
        reason = "normal_range"

        # 条件 0：首次 evolve 固定在 first_evolve_episode（不依赖 success rate）
        if not self._first_evolve_triggered:
            if self._total_episode_count < self.first_evolve_episode:
                reason = (
                    f"fixed_warmup(eps={self._total_episode_count}, "
                    f"need_eps={self.first_evolve_episode})"
                )
                self._log_acgs(f"[ACGS] Skipping: fixed warmup (episodes={self._total_episode_count}, "
                               f"need >={self.first_evolve_episode} eps for first evolve)")
                self._append_failure_vector_history(failure_vector_result, diagnosis, should_evolve=False, reason=reason)
                self._reset_batch()
                return

            self._warmup_completed = True
            should_evolve = True
            reason = (
                f"first_fixed(eps={self._total_episode_count}>="
                f"{self.first_evolve_episode})"
            )
            self._log_acgs(f"[ACGS] Warmup completed by fixed schedule: "
                           f"episodes={self._total_episode_count} >= {self.first_evolve_episode}")

        # 第二次以后：恢复自适应触发（too_easy / too_hard / plateau）
        else:
            # 条件 0.5：冷却期 — evolve 后等待足够的 episode 让策略适应
            episodes_since_evolve = self._total_episode_count - self._last_evolve_episode
            if episodes_since_evolve < self.evolve_cooldown:
                reason = f"cooldown({episodes_since_evolve}/{self.evolve_cooldown})"
                self._log_acgs(f"[ACGS] Skipping: cooldown ({episodes_since_evolve}/{self.evolve_cooldown} episodes since last evolve)")
                self._append_failure_vector_history(failure_vector_result, diagnosis, should_evolve=False, reason=reason)
                self._reset_batch()
                return

            # 条件 1：成功率过高 → 环境太简单
            if success_rate > self.success_rate_high:
                should_evolve = True
                reason = f"too_easy(sr={success_rate:.3f}>{self.success_rate_high})"

            # 条件 2：成功率过低 → 环境太难
            elif success_rate < self.success_rate_low:
                should_evolve = True
                reason = f"too_hard(sr={success_rate:.3f}<{self.success_rate_low})"

            # 条件 3：性能瓶颈 → 连续 K 个批次变化 < δ
            elif len(self._success_rate_history) >= self.plateau_window:
                recent = self._success_rate_history[-self.plateau_window:]
                max_change = max(recent) - min(recent)
                if max_change < self.plateau_threshold:
                    should_evolve = True
                    reason = (
                        f"plateau(range={max_change:.4f}<"
                        f"{self.plateau_threshold}, window={self.plateau_window})"
                    )
                    self._log_acgs(f"[ACGS] {reason}")

            if not should_evolve:
                reason = f"{reason}|no_evolve"
                self._log_acgs(f"[ACGS] No evolve needed: success_rate={success_rate:.3f} in normal range, "
                               f"no plateau (history len={len(self._success_rate_history)})")
                self._append_failure_vector_history(failure_vector_result, diagnosis, should_evolve=False, reason=reason)
                self._reset_batch()
                return

        self._append_failure_vector_history(failure_vector_result, diagnosis, should_evolve=True, reason=reason)

        if not self._first_evolve_triggered:
            self._first_evolve_triggered = True

        # ---- 触发 evolve ----
        self._log_acgs(f"[ACGS] >>> Triggering evolve: {reason}")

        evolve_record = {
            'time': time.time(),
            'total_episode_count': self._total_episode_count,
            'batch_episode_count': self._batch_episode_count,
            'reason': reason,
            'dominant_failure_type': failure_vector_result.get('dominant_type', 'success'),
            'llm_generation_success': False,
            'load_success': False,
            'accepted': False,
            'rejected_reason': None,
            'error': None,
        }

        if self.llm_generator is None or self.executor is None:
            self._log_acgs("[ACGS] No LLM generator, skipping evolve")
            evolve_record['rejected_reason'] = 'no_llm_generator'
            self._acgs_evolve_history.append(evolve_record)
            self._reset_batch()
            return

        try:
            prompt = self._build_acgs_prompt(
                reason=reason,
                fv_result=failure_vector_result,
                diagnosis=diagnosis,
            )
            self._log_acgs(f"[ACGS] Prompt length: {len(prompt)} chars")

            # 每次 evolve 前，将完整 prompt 追加写入实验目录下的 evolve_prompt.log
            self._append_evolve_prompt_log(
                prompt=prompt,
                reason=reason,
                fv_result=failure_vector_result,
                diagnosis=diagnosis,
            )

            new_code = self.llm_generator.evolve(prompt)
            evolve_record['llm_generation_success'] = bool(new_code)

            if new_code:
                load_success = bool(self.executor.load_topology_generator(new_code))
                evolve_record['load_success'] = load_success

                if load_success:
                    self.topology_generator_code = new_code
                    self._evolve_count += 1
                    self._just_evolved = True  # 设置标志，下一个 episode 会打印
                    self._last_evolve_episode = self._total_episode_count  # 记录 evolve 时间点
                    self._log_acgs(f"[ACGS] Evolve #{self._evolve_count} succeeded, new generator loaded "
                                   f"(cooldown {self.evolve_cooldown} eps)")
                    # evolve 成功后清空成功率历史，重新开始观察
                    self._success_rate_history.clear()
                    evolve_record['accepted'] = True
                else:
                    evolve_record['rejected_reason'] = 'load_failed'
                    self._log_acgs("[ACGS] Evolve failed: generated code could not be loaded")
            else:
                evolve_record['rejected_reason'] = 'llm_generation_failed'
                self._log_acgs("[ACGS] Evolve failed: LLM did not return valid code")

        except Exception as e:
            evolve_record['error'] = str(e)
            evolve_record['rejected_reason'] = 'exception'
            self._log_acgs(f"[ACGS] Evolve error: {e}")
            import traceback
            traceback.print_exc()

        self._acgs_evolve_history.append(evolve_record)

        self._reset_batch()

    def _reset_batch(self):
        """重置批次数据"""
        self._batch_stats = {'success': 0, 'collision': 0, 'timeout': 0, 'fall': 0}
        self._batch_failure_counts = {k: 0 for k in _FAILURE_KEYS}
        self._batch_episode_count = 0
        self._batch_failure_replays = []
        self._batch_success_replays = []
        self._batch_typed_failure_replays = {}

    def _random_tblock_pose(self):
        """随机生成有效的 T-block 位置"""
        while True:
            x = np.random.uniform(-0.18, 0.18)
            y = np.random.uniform(-0.18, 0.18)
            theta = np.random.uniform(0, 2*np.pi)
            if abs(x) > 0.1 or abs(y) > 0.1:
                return [x, y, theta]
    
    def learn(self):
        start_time = time.time()
        episode_reward_logger = []
        episode_reward = 0
        episode_length = 0
        num_episode = 0
        max_test_score = -100
        val_reward = -100
        
        step_log = dict()
        self.updates = 0
        self.critic_update = 0
        
        cfg = copy.deepcopy(self.cfg)
        if cfg.log:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
        
        optimizer_to(self.optimizer, self.device)
        optimizer_to(self.critic_optimizer, self.device)
        
        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # 初始化拓扑生成器（只调用一次 LLM）
        self._init_topology_generator()
        
        # 第一次 reset：生成 T-block 位置和障碍物
        obs, tblock_pose = self._generate_obstacles_for_reset()
        if obs is None:
            # 不使用 LLM，正常 reset
            obs = self.env.reset()
            tblock_pose = None

        # ACGS: 记录第一个 episode 的起点
        self._on_episode_start(obs, tblock_pose)

        self.model.reset()
        epsilon = cfg.rl.noise_epsilon
        self.depsilon = 1.0 / epsilon
        self.epsilon = 1.0
        
        with tqdm.tqdm(total=cfg.training.num_epochs, ncols=50, desc=f"Train epochs") as pbar:
            with tqdm.tqdm(total=cfg.rl.warmup, ncols=50, desc=f"Warm Up") as pbar2:
                while self.updates < cfg.training.num_epochs:
                    self.model.eval()

                    if isinstance(obs, dict):
                        obs_th = dict_to_torch(obs, device=self.device)
                    else:
                        if len(obs.shape) < 3:
                            obs = np.concatenate((obs,np.random.uniform(-1,1,self.env.obs_dim[0]-obs.shape[-1]).reshape(1,-1)),axis=1)

                        obs_th = torch.tensor(obs, dtype=torch.float32).to(device=self.device)

                    # sample action
                    with torch.no_grad():
                        if self.num_timesteps <= cfg.rl.warmup:
                            action = np.random.uniform(
                                self.env.action_space.low.flat[0], 
                                self.env.action_space.high.flat[0], 
                                self.action_dim
                            )
                            action = action.reshape(
                                1, *self.action_dim
                            )
                        else:
                            action = self.model.predict_action(obs_th)
                            action = action.detach().cpu().numpy()
                            if cfg.rl.add_noise:
                                random_noise = self.random_process.sample()
                                random_noise = random_noise.reshape(*action.shape)
                                action += random_noise*max(self.epsilon, 0)
                                self.epsilon -= self.depsilon
                                action = np.clip(action, self.env.action_space.low.flat[0], self.env.action_space.high.flat[0])
                    
                    # env step
                    next_obs, reward, done, info = self.env.step(action[0])

                    if isinstance(obs, dict):
                        obs_th = dict_to_torch(obs, device=self.device)
                    else:
                        if len(next_obs.shape) < 3:
                            next_obs = np.concatenate((next_obs,np.random.uniform(-1,1,self.env.obs_dim[0]-next_obs.shape[-1]).reshape(1,-1)),axis=1)

                    # ACGS: 采集帧数据 (T-block位姿, rod位置, Q值)
                    q_value = None
                    if self.num_timesteps > cfg.rl.warmup:
                        try:
                            with torch.no_grad():
                                obs_for_q = torch.tensor(obs, dtype=torch.float32).to(self.device)
                                act_for_q = torch.tensor(action, dtype=torch.float32).to(self.device)
                                q_vals = self.critic(obs_for_q, act_for_q)
                                q_value = torch.cat(q_vals, dim=1).min(dim=1)[0].item()
                        except Exception:
                            q_value = None
                    self._record_frame(next_obs, q_value)

                    # Replay buffer
                    self.replay_buffer.add(obs, next_obs, action, reward, done)
 
                    obs = next_obs
                    episode_reward += reward
                    episode_length += 1
                    self.num_timesteps += 1

                    if episode_length >= cfg.max_steps-1:
                        done = True

                    pbar2.update(1)

                    if done:
                        episode_reward_logger.append(episode_reward)

                        # ACGS: 记录 episode 结果
                        self._on_episode_end(episode_reward, episode_length, info)

                        self.env.seed()
                        self.model.reset()

                        # 每次 reset 前：生成新的 T-block 位置和障碍物配置
                        self.episode_count += 1
                        obs, tblock_pose = self._generate_obstacles_for_reset()
                        if obs is None:
                            # 不使用 LLM，正常 reset
                            obs = self.env.reset()
                            tblock_pose = None

                        # ACGS: 记录新 episode 的起点
                        self._on_episode_start(obs, tblock_pose)

                        episode_length, episode_reward = 0, 0
                        num_episode += 1
                    
                    # update model
                    if self.num_timesteps > cfg.rl.warmup:
                        if self.num_timesteps == cfg.rl.warmup+1:
                            pbar2.close()
                        self.model.train()
                            
                        training_info = self.update(batch_size=cfg.rl.batch_size)
                            
                        pbar.update(1)
                        pbar.set_postfix(episode_reward=np.mean(episode_reward_logger[-100:]))
                        
                        # validate
                        policy = self.model
                        policy.eval()

                        if cfg.training.validate and self.updates % cfg.training.validate_steps == 0:
                            # 验证前清除 LLM 配置，使用原版 between 策略
                            if hasattr(self.env, 'clear_obstacle_config'):
                                self.env.clear_obstacle_config()
                                if self.updates % (cfg.training.validate_steps * 5) == 0:  # 每 5 次验证打印一次
                                    print(f"\n[Validation] Cleared LLM config, using original 'between' strategy")
                            
                            val_reward, val_log = self.evaluator(self.env, policy, self.no_obs_env, self.unseen_env)
                            _ = self.env.reset()
                            wandb_run.log({
                                    'validate_reward': val_reward,
                                },step = self.updates)
                            step_log.update(val_log)
                            wandb_run.log(step_log, step=self.updates)

                        # checkpoint
                        if (self.updates % cfg.training.checkpoint_every) == 0:
                            # checkpointing
                            if cfg.checkpoint.save_last_ckpt:
                                self.save_checkpoint()
                                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_latest.pt'))
                            if cfg.checkpoint.save_last_snapshot:
                                self.save_snapshot()
                            
                            if (max_test_score < val_reward):
                                max_test_score = val_reward
                                # sanitize metric names
                                metric_dict = dict()
                                metric_dict['test_mean_score'] = val_reward
                                metric_dict['epoch'] = self.updates
                                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                                if topk_ckpt_path is not None:
                                    self.save_checkpoint(path=topk_ckpt_path)
                                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'model_epoch={self.updates}-test_mean_score={val_reward:.3f}.pt'))

                        # log
                        if cfg.log and self.updates % cfg.log_interval == 0:
                            with torch.no_grad():
                                # ACGS metrics
                                acgs_log = {
                                    'acgs/batch_episode_count': self._batch_episode_count,
                                    'acgs/evolve_count': self._evolve_count,
                                }
                                batch_total = sum(self._batch_stats.values())
                                if batch_total > 0:
                                    acgs_log['acgs/batch_success_rate'] = self._batch_stats['success'] / batch_total
                                    acgs_log['acgs/batch_collision_rate'] = self._batch_stats['collision'] / batch_total

                                wandb_run.log({
                                    **training_info,
                                    **acgs_log,
                                    'episode_mean_reward': np.mean(episode_reward_logger[-100:]),
                                    'num_timesteps': self.num_timesteps,
                                    "num_episode": num_episode,
                                    'step_ps': self.num_timesteps / (time.time() - start_time),
                                },step = self.updates)
                                
                        self.updates += 1
                        
    def update(self, batch_size: int):
        cfg = copy.deepcopy(self.cfg)
        experience_replay = self.replay_buffer.sample(batch_size=batch_size)

        # Compute target q values
        with torch.no_grad():
            next_q_value = self.compute_next_q_value(experience_replay)
            rewards = torch.from_numpy(experience_replay.rewards).to(self.device)
            target_q_value = rewards + self.gamma * (1 - torch.from_numpy(experience_replay.dones).to(self.device)) * next_q_value
            target_q_value = target_q_value.detach()
        
        self.critic_optimizer.zero_grad()
        
        # Compute critic loss
        critic_loss, critic_loss_info = self.compute_critic_loss(experience_replay, target_q_value)
        critic_loss.backward()
        
        if self.critic_gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_gradient_max_norm, norm_type=2)

        parameters = [p for p in self.critic.parameters() if p.grad is not None and p.requires_grad]
        critic_gradient_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(self.device) for p in parameters]), 2.0).item()
        self.critic_optimizer.step()
        critic_loss_info['[critic]gradient_norm'] = critic_gradient_norm

        # Compute actor loss
        if self.critic_update % cfg.rl.policy_update == 0:
            self.optimizer.zero_grad()
            policy_loss, policy_loss_info = self.compute_policy_loss(experience_replay)
            policy_loss.backward()

            if self.policy_gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.policy_gradient_max_norm, norm_type=2)

            parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
            policy_gradient_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(self.device) for p in parameters]), 2.0).item()

            self.optimizer.step()
            policy_loss_info['[policy]replay_rewards'] = experience_replay.rewards.mean()
            policy_loss_info['[policy]gradient_norm'] = policy_gradient_norm
        else:
            policy_loss_info = {}

        # Update the target model
        soft_update(self.critic_target, self.critic, cfg.rl.soft_update_tau)
        if self.critic_update % cfg.rl.policy_update == 0:
            soft_update(self.model_target, self.model, cfg.rl.soft_update_tau)
        
        self.critic_update += 1

        info = {
            **critic_loss_info,
            **policy_loss_info,
        }
        return info
    
    def compute_next_q_value(self, experience_replay):
        next_obs = experience_replay.next_observations
        if isinstance(next_obs, dict):
            next_obs_th = dict_to_torch(next_obs, device=self.device)
        else:
            next_obs_th = torch.tensor(next_obs, dtype=torch.float32).to(device=self.device)
        next_action = self.model_target.predict_action(next_obs_th)

        next_q_values = self.critic_target(next_obs_th, next_action)
        next_q_values = torch.cat(next_q_values, dim=1)
        next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
        return next_q_values
    
    def compute_critic_loss(self, experience_replay, target_q_value):
        obs = experience_replay.observations
        if isinstance(obs, dict):
            obs_th = dict_to_torch(obs, device=self.device)
        else:
            obs_th = torch.tensor(obs, dtype=torch.float32).to(device=self.device)
        action = torch.from_numpy(experience_replay.actions).to(self.device)
        current_q_values = self.critic(obs_th, action)
        critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_value) for current_q in current_q_values])
        with torch.no_grad():
            critic_loss_info = {}
            critic_loss_info['[critic]loss'] = critic_loss.item()
        return critic_loss.to(self.device), critic_loss_info
   
    def compute_policy_loss(self, experience_replay):
        obs = experience_replay.observations
        if isinstance(obs, dict):
            obs_th = dict_to_torch(obs, device=self.device)
        else:
            obs_th = torch.tensor(obs, dtype=torch.float32).to(device=self.device)
        if ('regularize_z' in self.cfg.training):
                z = self.model.encoder(obs_th)
                state = self.model.obs2state(obs_th)
                action = self.model.decoder(torch.cat([state, z], dim=1))
                z_mean = z.mean(0)
                z_var = z.var(0)
        else:   
            action = self.model.predict_action(obs_th)

        current_q_values = self.critic(obs_th, action)
        current_q_values = torch.cat(current_q_values, dim=1)
        current_q_values, _ = torch.min(current_q_values, dim=1, keepdim=True)
        policy_loss = -current_q_values

        policy_loss = policy_loss.mean()
        if ('regularize_z' in self.cfg.training):
            if self.cfg.training.regularize_z == 'norm':
                policy_loss += self.cfg.training.reg_coeff*(torch.norm(z, dim=1)**2).mean()
            elif self.cfg.training.regularize_z == 'gaussian':
                feature_loss = F.mse_loss(z_mean, torch.full_like(z_mean, 0)) + \
                            F.mse_loss(z_var, torch.full_like(z_var, 1))

                policy_loss += self.cfg.training.reg_coeff * feature_loss
            elif self.cfg.training.regularize_z == False:
                pass
            else:
                NotImplementedError

        with torch.no_grad():
            policy_loss_info = {}
            policy_loss_info['[policy]policy_loss'] = policy_loss.item()
            policy_loss_info['[policy]action_norm'] = (torch.norm(action.mean(dim=0))/torch.norm(torch.ones_like(action[0]))).item()
            policy_loss_info['[policy]current_q_values'] = current_q_values.mean().item()
            policy_loss_info['[policy]current_q_values max'] = current_q_values.max().item()
            policy_loss_info['[policy]current_q_values min'] = current_q_values.min().item()
            if 'z' in locals():
                policy_loss_info['[policy]z_norm'] = (torch.norm(z, dim=1)**2).mean().item()
                policy_loss_info['[policy]z_mean'] = (z_mean).mean().item()
                policy_loss_info['[policy]z_var'] = (z_var).mean().item()

        return policy_loss.to(self.device), policy_loss_info
