if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import ast
import csv
import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, fields
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
import traceback

from DIVO.RL.component import StateDictReplayBuffer, OrnsteinUhlenbeckProcess, hard_update, soft_update, reward_for_critic

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
from DIVO.curriculum import (
    CurriculumRuntimeState,
    GeneratorArtifactStore,
    RLLoopState,
    build_generator_source,
)
from DIVO.curriculum.adapters.pusht_adapter import (
    PUSHT_TARGET_XY,
    pusht_build_scene_graph,
    pusht_encode_layout,
)
from DIVO.curriculum.attribution import (
    AttributionConfig,
    append_attribution_history,
    compute_attribution,
    write_attribution_outputs,
)
from DIVO.curriculum.candidate_verifier import (
    CandidateVerifierConfig,
    SkillCandidateVerifierSession,
    SkillVerifierConfig,
    save_verification_artifacts,
    verify_pusht_candidate,
)
from DIVO.curriculum.phase4_monitor import build_phase4_record
from DIVO.curriculum.skill_signal import build_pusht_env_for_probe
from DIVO.curriculum.skill_diversity import (
    DiversityConfig,
    run_paired_diversity_rollout,
    compute_diversity_rewards,
    write_diversity_transitions,
    k_eff_from_rollouts,
)
from DIVO.curriculum.layout_features import extract_layout_z
from DIVO.curriculum.pattern_discovery import (
    PatternDiscoveryConfig,
    discover_failure_graph_patterns,
)
from DIVO.gpt.acgs_api import ACGS_API


LOGGER = logging.getLogger(__name__)


def _cfg_get(cfg, *paths, default=None):
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


def _cfg_to_plain_dict(cfg):
    if cfg is None:
        return {}
    try:
        value = OmegaConf.to_container(cfg, resolve=True)
        return value if isinstance(value, dict) else {}
    except Exception:
        pass
    if isinstance(cfg, dict):
        return dict(cfg)
    return {}


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


class TD3CurriculumWorkspace(BaseWorkspace):

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        self.curriculum_state = CurriculumRuntimeState()
        self.rl_loop_state = RLLoopState()
        self._resume_applied = False

        # set seed
        seed = cfg.training.seed
        self.device = torch.device(cfg.training.device)

        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # set env (使用 LLM 版本的环境)
        self.env = get_env_class(**cfg.env)
        self.llm_eval_env = get_env_class(**cfg.env)
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

        # Task 6: per-episode skill sampling config for skill-conditioned rollouts.
        self._init_skill_rollout(cfg)

        replay_buffer_args = {
            "obs_dim": (self.obs_dim),
            "action_dim": (self.action_dim),
        }
        
        self.replay_buffer = StateDictReplayBuffer(
            cfg.rl.replay_buffer_size, 
            track_skill=self.skill_enabled,
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
        self.obstacle_rollout_log_path = os.path.join(self.output_dir, 'obstacle_rollouts.jsonl')
        self.generator_artifacts = GeneratorArtifactStore(self.output_dir)

    def get_extra_state(self):
        self._sync_curriculum_state_from_workspace()
        extra_state = {
            'curriculum_state': asdict(self.curriculum_state),
            'rl_loop_state': asdict(self.rl_loop_state),
            'rng_state': self._capture_rng_state(),
        }
        return extra_state

    def load_extra_state(self, extra_state):
        curriculum_payload = extra_state.get('curriculum_state')
        if curriculum_payload is not None:
            self.curriculum_state = self._runtime_state_from_payload(curriculum_payload)
            self._restore_curriculum_state_to_workspace()

        rl_payload = extra_state.get('rl_loop_state')
        if rl_payload is not None:
            self.rl_loop_state = RLLoopState(**rl_payload)

        rng_state = extra_state.get('rng_state')
        if rng_state is not None:
            self._restore_rng_state(rng_state)

    def _runtime_state_from_payload(self, payload):
        """Load checkpoints across old/new curriculum runtime-state schemas."""
        valid_keys = {field.name for field in fields(CurriculumRuntimeState)}
        filtered = {
            key: value
            for key, value in dict(payload).items()
            if key in valid_keys
        }
        return CurriculumRuntimeState(**filtered)

    def _capture_rng_state(self):
        rng_state = {
            'python_random': random.getstate(),
            'numpy_random': np.random.get_state(),
            'torch_random': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            try:
                rng_state['torch_cuda_random'] = torch.cuda.get_rng_state_all()
            except Exception:
                rng_state['torch_cuda_random'] = None
        return rng_state

    def _restore_rng_state(self, rng_state):
        random_state = rng_state.get('python_random')
        if random_state is not None:
            random.setstate(random_state)

        numpy_state = rng_state.get('numpy_random')
        if numpy_state is not None:
            np.random.set_state(numpy_state)

        torch_state = rng_state.get('torch_random')
        if torch_state is not None:
            torch.set_rng_state(torch_state)

        cuda_state = rng_state.get('torch_cuda_random')
        if cuda_state is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(cuda_state)
            except Exception:
                pass

    def _sync_curriculum_state_from_workspace(self):
        self.curriculum_state.initial_generator_code = (
            self.curriculum_state.initial_generator_code or self.topology_generator_code
        )
        self.curriculum_state.current_generator_code = self.topology_generator_code
        self.curriculum_state.evolve_count = self._evolve_count
        self.curriculum_state.total_episode_count = self._total_episode_count
        self.curriculum_state.success_rate_history = list(self._success_rate_history)
        self.curriculum_state.warmup_completed = self._warmup_completed
        self.curriculum_state.first_evolve_triggered = self._first_evolve_triggered
        self.curriculum_state.last_evolve_episode = self._last_evolve_episode
        self.curriculum_state.pre_evolve_checkpoint_saved = getattr(
            self, '_pre_evolve_checkpoint_saved', False
        )
        self.curriculum_state.batch_stats = copy.deepcopy(self._batch_stats)
        self.curriculum_state.batch_failure_counts = copy.deepcopy(self._batch_failure_counts)
        self.curriculum_state.batch_episode_count = self._batch_episode_count
        self.curriculum_state.batch_obstacle_rollout_records = copy.deepcopy(
            getattr(self, '_batch_obstacle_rollout_records', [])
        )
        self.curriculum_state.failure_vector_history = copy.deepcopy(self._failure_vector_history)
        self.curriculum_state.evolve_history = copy.deepcopy(self._acgs_evolve_history)
        self.curriculum_state.attribution_history = copy.deepcopy(
            getattr(self, '_attribution_history', [])
        )

    def _restore_curriculum_state_to_workspace(self):
        self.topology_generator_code = self.curriculum_state.current_generator_code
        self._evolve_count = self.curriculum_state.evolve_count
        self._total_episode_count = self.curriculum_state.total_episode_count
        self._success_rate_history = list(self.curriculum_state.success_rate_history)
        self._warmup_completed = self.curriculum_state.warmup_completed
        self._first_evolve_triggered = self.curriculum_state.first_evolve_triggered
        self._last_evolve_episode = self.curriculum_state.last_evolve_episode
        self._pre_evolve_checkpoint_saved = self.curriculum_state.pre_evolve_checkpoint_saved
        self._batch_stats = copy.deepcopy(self.curriculum_state.batch_stats)
        self._batch_failure_counts = copy.deepcopy(self.curriculum_state.batch_failure_counts)
        self._batch_episode_count = self.curriculum_state.batch_episode_count
        self._batch_obstacle_rollout_records = copy.deepcopy(
            getattr(self.curriculum_state, 'batch_obstacle_rollout_records', [])
        )
        self._failure_vector_history = copy.deepcopy(self.curriculum_state.failure_vector_history)
        self._acgs_evolve_history = copy.deepcopy(self.curriculum_state.evolve_history)
        self._attribution_history = copy.deepcopy(
            getattr(self.curriculum_state, 'attribution_history', [])
        )

    def _sync_rl_loop_state(
        self,
        max_test_score=None,
        val_reward=None,
        episode_reward_logger=None,
        num_episode=None,
    ):
        if hasattr(self, 'updates'):
            self.rl_loop_state.updates = int(self.updates)
        if hasattr(self, 'critic_update'):
            self.rl_loop_state.critic_update = int(self.critic_update)
        self.rl_loop_state.global_step = int(getattr(self, 'global_step', 0))
        self.rl_loop_state.epoch = int(getattr(self, 'epoch', 0))
        self.rl_loop_state.num_timesteps = int(getattr(self, 'num_timesteps', 0))
        self.rl_loop_state.episode_count = int(getattr(self, 'episode_count', 0))
        if num_episode is not None:
            self.rl_loop_state.num_episode = int(num_episode)
        if hasattr(self, 'epsilon'):
            self.rl_loop_state.epsilon = float(self.epsilon)
        if max_test_score is not None:
            self.rl_loop_state.max_test_score = float(max_test_score)
        if val_reward is not None:
            self.rl_loop_state.last_validate_reward = float(val_reward)
        if episode_reward_logger is not None:
            self.rl_loop_state.recent_episode_rewards = list(episode_reward_logger[-100:])

    def _restore_generator_runtime_from_state(self):
        code = self.curriculum_state.current_generator_code
        if not code:
            return

        if self.acgs_api is not None:
            if not self.acgs_api.load_generator_code(code):
                raise RuntimeError("Failed to restore generator code into ACGS_API")
            self.topology_generator_code = self.acgs_api.export_generator_code()
            self.executor = self.acgs_api.executor
            return

        if self.executor is not None:
            if not self.executor.load_topology_generator(code):
                raise RuntimeError("Failed to restore generator code into local executor")
            self.topology_generator_code = code

    def _clear_pending_evolve_state(self):
        self.curriculum_state.pending_evolve_reason = None

    def _save_pre_evolve_branch_checkpoint(self, reason):
        if getattr(self, '_pre_evolve_checkpoint_saved', False):
            return

        self.curriculum_state.pending_evolve_reason = reason
        self._pre_evolve_checkpoint_saved = True
        self.curriculum_state.pre_evolve_checkpoint_saved = True
        self._sync_curriculum_state_from_workspace()
        self._sync_rl_loop_state()

        ckpt_path = self.save_checkpoint(
            tag=self.pre_evolve_checkpoint_tag,
            include_keys=(),
            use_thread=False,
        )
        self._log_acgs(f"[ACGS] Saved pre-evolve branch checkpoint: {ckpt_path}")

    def _resume_from_checkpoint_if_requested(self):
        resume_path = getattr(self, 'resume_checkpoint_path', None)
        if not resume_path or self._resume_applied:
            return False

        payload = self.load_checkpoint(path=resume_path, include_keys=())
        self._restore_generator_runtime_from_state()
        self._resume_applied = True
        LOGGER.info(f"[Resume] Loaded checkpoint from {resume_path}")
        return bool(payload)
    
    def _init_llm_generator(self, cfg):
        """初始化 LLM 拓扑生成器（Stage A）"""
        self.acgs_api = None
        self.llm_generator = None   # deprecated，保留兼容
        self.executor = None        # deprecated，保留兼容
        self.topology_generator_code = None
        self.obstacle_num = getattr(cfg.env, 'obstacle_num', 2)
        
        curriculum_cfg = getattr(cfg, 'curriculum', None)
        self.generator_source = build_generator_source(curriculum_cfg)
        self.save_generator_artifacts = bool(_cfg_get(
            curriculum_cfg,
            "generator.save_artifacts",
            "save_generator_artifacts",
            default=True,
        ))
        self.save_pre_evolve_checkpoint = bool(_cfg_get(
            curriculum_cfg,
            "branch.save_pre_evolve_checkpoint",
            "save_pre_evolve_checkpoint",
            default=False,
        ))
        self.pre_evolve_checkpoint_tag = str(_cfg_get(
            curriculum_cfg,
            "branch.pre_evolve_checkpoint_tag",
            "pre_evolve_checkpoint_tag",
            default="pre_first_evolve",
        ))
        self.resume_checkpoint_path = _cfg_get(
            curriculum_cfg,
            "branch.resume_checkpoint",
            default=None,
        )
        if self.resume_checkpoint_path is None:
            self.resume_checkpoint_path = _cfg_get(
                getattr(cfg, 'training', None),
                "resume_checkpoint",
                default=None,
            )
        self._pre_evolve_checkpoint_saved = False
        
        # Episode 计数（用于日志）
        self.episode_count = 0
        
        # 对比实验：是否使用 LLM 生成障碍物
        self.use_llm_obstacles = bool(_cfg_get(
            curriculum_cfg,
            "use_llm_obstacles",
            "generator.use_llm_obstacles",
            default=True,
        )) if curriculum_cfg else False
        
        if not self.use_llm_obstacles:
            print("\n [1.1] LLM obstacle generation DISABLED (using random obstacles)")
            print(f" [1.1] This is a baseline experiment")
            return
        
        if curriculum_cfg is None:
            print("\n [1.1] No curriculum config, LLM generator disabled")
            return
        
        api_type = _cfg_get(curriculum_cfg, 'api_type', 'generator.api_type', default='deepseek')
        api_key = _cfg_get(curriculum_cfg, 'api_key', 'generator.api_key', default=None)
        model = _cfg_get(curriculum_cfg, 'model', 'generator.model', default=None) or (
            'deepseek-chat' if api_type == 'deepseek' else 'gpt-5.4'
        )
        base_url = _cfg_get(curriculum_cfg, 'base_url', 'generator.base_url', default=None)
        
        if api_key is None:
            if api_type == 'deepseek':
                api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
            else:
                api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        
        if api_key:
            try:
                prompt_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "gpt", "prompt"
                )
                task_name = _cfg_get(curriculum_cfg, 'task_name', 'generator.task_name', default='PushT')

                self.acgs_api = ACGS_API(
                    task_name=task_name,
                    prompt_dir=prompt_dir,
                    api_type=api_type,
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                    temperature=0.7,
                    obstacle_size=float(getattr(cfg.env, 'obstacle_size', 0.01)),
                    target_pose=[0, 0, -np.pi / 4],
                    max_evolve_retries=3,
                    sanity_check_count=5,
                )
                # 保留旧引用以兼容尚未迁移的代码
                self.llm_generator = self.acgs_api  # deprecated
                self.executor = self.acgs_api.executor  # deprecated
                print(f"\n [1.1] ACGS_API initialized (task={task_name}, model={model})")
                print(f" [1.1] Obstacle generation mode: LLM topology generator (every reset)")
            except Exception as e:
                print(f"\n [1.1] ACGS_API init failed: {e}")
                traceback.print_exc()
                self.use_llm_obstacles = False
        else:
            init_path = _cfg_get(curriculum_cfg, "generator.init_path", "initial_generator_path", default=None)
            if init_path:
                self.executor = StrategyExecutor(
                    obstacle_size=float(getattr(cfg.env, 'obstacle_size', 0.01)),
                    target_pose=[0, 0, -np.pi / 4],
                )
                self.llm_generator = self.executor
                print("\n [1.1] No API key found, using local file-backed generator runtime")
                print(f" [1.1] Initial generator source: {init_path}")
            else:
                print("\n [1.1] No API key found, LLM generator disabled")
                self.use_llm_obstacles = False
    
    def _init_topology_generator(self):
        """初始化拓扑生成器（只在训练开始时调用一次）"""
        if not self.use_llm_obstacles:
            return False

        if self.acgs_api is None and self.executor is None:
            return False
        
        if self.topology_generator_code is not None:
            return True  # 已经初始化过了
        
        try:
            print(f"\n[Stage A] Generating topology generator code (first time)...")

            if hasattr(self.env, 'sample_valid_tblock_pose'):
                sample_tblock_pose = self.env.sample_valid_tblock_pose()
            else:
                sample_tblock_pose = self._random_tblock_pose()
            code = self.generator_source.get_initial_code(
                np.array(sample_tblock_pose),
                num_obstacles=self.obstacle_num,
                acgs_api=self.acgs_api,
            )
            
            if code is None:
                print("[Stage A] ❌ Code generation failed, disabling LLM obstacles")
                self.use_llm_obstacles = False
                return False

            load_ok = False
            if self.acgs_api is not None:
                load_ok = self.acgs_api.load_generator_code(code)
                if load_ok:
                    self.topology_generator_code = self.acgs_api.export_generator_code()
                    self.executor = self.acgs_api.executor
            else:
                load_ok = self.executor.load_topology_generator(code)
                if load_ok:
                    self.topology_generator_code = code

            if not load_ok:
                print("[Stage A] ❌ Code loading failed, disabling LLM obstacles")
                self.use_llm_obstacles = False
                return False

            self.curriculum_state.initial_generator_code = self.topology_generator_code
            self.curriculum_state.current_generator_code = self.topology_generator_code

            if self.save_generator_artifacts and self.topology_generator_code:
                self.generator_artifacts.save_initial(self.topology_generator_code)
                self.generator_artifacts.save_current(self.topology_generator_code)
                self.generator_artifacts.save_generator_version(self.topology_generator_code, generator_id=0)

            print(f"[Stage A] ✓ Topology generator loaded successfully")
            print(f"[Stage A] Code length: {len(code)} characters")
            return True
            
        except Exception as e:
            print(f"[Stage A] ❌ Initialization failed: {e}")
            traceback.print_exc()
            self.use_llm_obstacles = False
            return False
    
    def _has_runtime_generator(self):
        if self.acgs_api is not None and self.acgs_api.has_generator:
            return True
        if self.executor is not None and self.topology_generator_code:
            return True
        return False

    def _normalize_obstacle_config(self, obstacles):
        if obstacles is None:
            return []

        normalized = []
        for cfg in obstacles:
            if not isinstance(cfg, dict):
                continue
            if 'x' not in cfg or 'y' not in cfg:
                continue
            item = {
                'x': float(cfg['x']),
                'y': float(cfg['y']),
            }
            if 'purpose' in cfg:
                item['purpose'] = str(cfg.get('purpose', ''))
            normalized.append(item)
        return normalized

    def _log_llm_scene(self, tblock_pose, obstacles, attempt, env_name="train"):
        if env_name != "train":
            return
        if not (
            self.episode_count == 0
            or self.episode_count % 1000 == 0
            or (hasattr(self, '_just_evolved') and self._just_evolved)
        ):
            return

        log_lines = [
            f"[LLM] Episode {self.episode_count}: Generated {len(obstacles)} obstacles",
            f"  T-block: ({tblock_pose[0]:.3f}, {tblock_pose[1]:.3f}, {np.degrees(tblock_pose[2]):.0f}deg)",
        ]
        for idx, obs_cfg in enumerate(obstacles):
            log_lines.append(f"  Obs{idx+1}: ({obs_cfg['x']:.3f}, {obs_cfg['y']:.3f})")
        if attempt > 1:
            log_lines.append(f"  Valid after {attempt} attempts")
        LOGGER.info("\n".join(log_lines))
        if hasattr(self, '_just_evolved'):
            self._just_evolved = False

    def _make_llm_scene(self, env=None, env_name="train"):
        """
        基于当前 generator 构造一个绑定场景：先采样起点 A，再基于 A 生成障碍物。

        A（起点）和为 A 生成的障碍物是绑定的整体配置。
        如果物理引擎检测到初始接触（ncon > 0），整体重新生成，不单独换起点。

        Returns:
            (obs, tblock_pose): reset 后的观测和 T-block 位置
                               如果不使用 LLM 则返回 (None, None)
        """
        if env is None:
            env = self.env

        if not self.use_llm_obstacles:
            return None, None

        if not self._has_runtime_generator() or not hasattr(env, 'set_obstacle_config'):
            return None, None

        attempt = 0
        while True:
            attempt += 1
            try:
                if hasattr(env, 'sample_valid_tblock_pose'):
                    tblock_pose_A = np.array(env.sample_valid_tblock_pose(), dtype=np.float64)
                else:
                    tblock_pose_A = np.array(self._random_tblock_pose(), dtype=np.float64)

                if self.acgs_api is not None and self.acgs_api.has_generator:
                    obstacles = self.acgs_api.generate_obstacles(tblock_pose_A, self.obstacle_num)
                else:
                    obstacles = self.executor.generate(tblock_pose_A, self.obstacle_num)

                obstacles = self._normalize_obstacle_config(obstacles)
                if len(obstacles) == 0:
                    if attempt % 10 == 0:
                        print(f"[LLM] ⚠ {attempt} attempts, empty obstacle set on {env_name}, resampling...")
                    if attempt >= 200:
                        print("[LLM] Too many empty obstacle generations, fallback to normal reset")
                        if hasattr(env, 'clear_obstacle_config'):
                            env.clear_obstacle_config()
                        return None, None
                    continue

                if hasattr(env, 'is_obstacle_config_valid'):
                    if not env.is_obstacle_config_valid(obstacles, tblock_pose_A):
                        if attempt % 10 == 0:
                            print(f"[LLM] ⚠ {attempt} attempts, invalid scene pair on {env_name}, retrying...")
                        if attempt >= 200:
                            print("[LLM] Too many invalid scenes, fallback to normal reset")
                            if hasattr(env, 'clear_obstacle_config'):
                                env.clear_obstacle_config()
                            return None, None
                        continue

                env.set_obstacle_config(obstacles)
                obs = env.reset(tblock_pos=tblock_pose_A, force_tblock_pos=True)

                ncon = env.get_ncon()
                if ncon == 0:
                    self._log_llm_scene(tblock_pose_A, obstacles, attempt, env_name=env_name)
                    return obs, tblock_pose_A

                if attempt % 10 == 0:
                    print(f"[LLM] ⚠ {attempt} attempts, ncon={ncon}, retrying A + obstacles...")
                if attempt >= 200:
                    print("[LLM] Too many contact retries, fallback to normal reset")
                    if hasattr(env, 'clear_obstacle_config'):
                        env.clear_obstacle_config()
                    return None, None

            except Exception as e:
                if attempt % 10 == 0:
                    print(f"[LLM] ⚠ {attempt} attempts on {env_name}, exception: {e}")
                if attempt >= 20:
                    print("[LLM] Too many failures, fallback to normal reset")
                    if hasattr(env, 'clear_obstacle_config'):
                        env.clear_obstacle_config()
                    return None, None

    def _generate_obstacles_for_reset(self, env=None, env_name="train"):
        return self._make_llm_scene(env=env, env_name=env_name)

    def _reset_seen_eval_episode(self, env, episode):
        obs, _ = self._make_llm_scene(env=env, env_name="eval")
        if obs is None:
            return env.reset()
        return obs

    # ===== 技能库回合采集(Task 6) =====

    def _init_skill_rollout(self, cfg):
        """Configure per-episode skill sampling (Task 6 / Requirements 6.1, 6.4, 6.5).

        Disabled by default (skill_enabled=False) so strict-DIVO training is
        unchanged. When enabled, each training episode executes a single skill
        w drawn with P(w_0) >= w0_rollout_ratio (>= 0.5) and the remaining mass
        spread uniformly over the K probe skills w_1..w_K.
        """
        policy_cfg = getattr(cfg, 'policy', None)
        self.skill_enabled = bool(_cfg_get(policy_cfg, 'skill_enabled', default=False))
        self.skill_K = int(_cfg_get(policy_cfg, 'K', default=0))
        self.w0_rollout_ratio = float(_cfg_get(
            cfg, 'skill.w0_rollout_ratio', 'curriculum.skill.w0_rollout_ratio',
            default=0.5,
        ))
        self.probe_rollout_ratio = float(_cfg_get(
            cfg, 'skill.probe_rollout_ratio', 'curriculum.skill.probe_rollout_ratio',
            default=0.5,
        ))
        self.paired_sample_ratio = float(_cfg_get(
            cfg, 'skill.paired_sample_ratio', 'curriculum.skill.paired_sample_ratio',
            default=0.3,
        ))
        self.lambda_cov_all = float(_cfg_get(
            cfg, 'skill.lambda_cov_all', 'curriculum.skill.lambda_cov_all',
            default=0.0,
        ))
        # Task 9: periodic paired diversity batch (default OFF: diversity_every=0).
        self.diversity_every = int(_cfg_get(
            cfg, 'skill.diversity_every', 'curriculum.skill.diversity_every',
            default=0,
        ))
        self.diversity_num_layouts = int(_cfg_get(
            cfg, 'skill.diversity_num_layouts', 'curriculum.skill.diversity_num_layouts',
            default=8,
        ))
        self.use_dual_critic = bool(_cfg_get(
            cfg, 'skill.use_dual_critic', 'curriculum.skill.use_dual_critic',
            default=False,
        ))
        skill_cfg = getattr(cfg, 'skill', None)
        self._diversity_cfg = DiversityConfig.from_cfg(
            skill_cfg if skill_cfg is not None else None
        )
        # Task 10: dynamic beta_div schedule (constraint-balanced, DOMiNO-style).
        self.task_warmup_steps = int(_cfg_get(
            cfg, 'skill.task_warmup_steps', 'curriculum.skill.task_warmup_steps',
            default=0,
        ))
        self.beta_div = float(self._diversity_cfg.beta_div)
        self.beta_div_min = float(_cfg_get(
            cfg, 'skill.beta_div_min', 'curriculum.skill.beta_div_min', default=1e-4,
        ))
        self.beta_div_max = float(_cfg_get(
            cfg, 'skill.beta_div_max', 'curriculum.skill.beta_div_max', default=0.1,
        ))
        self.success_floor = float(_cfg_get(
            cfg, 'skill.success_floor', 'curriculum.skill.success_floor', default=0.5,
        ))
        _eff_floor = _cfg_get(cfg, 'skill.eff_floor', 'curriculum.skill.eff_floor', default=None)
        self.eff_floor = float(_eff_floor) if _eff_floor is not None else 0.6 * max(int(self.skill_K), 1)
        self.keff_cluster_threshold = float(_cfg_get(
            cfg, 'skill.keff_cluster_threshold', 'curriculum.skill.keff_cluster_threshold',
            default=0.5,
        ))
        self._last_k_eff = 0.0
        self._recent_success = deque(maxlen=100)
        if self.skill_enabled:
            assert self.skill_K >= 1, "skill_enabled requires policy.K >= 1"
            assert self.w0_rollout_ratio >= 0.5, (
                "w0_rollout_ratio must be >= 0.5 to protect the deployment skill "
                "(Requirement 6.5)"
            )
            if self.use_dual_critic:
                # Task 9.5: dual-critic (Q_task / Q_total) is a designed fallback,
                # not implemented in Stage 1. Fail loudly rather than silently
                # single-critic.
                raise NotImplementedError(
                    "use_dual_critic is a Stage 1 fallback skeleton and is not "
                    "implemented yet; keep it False (single critic + reward routing)."
                )
        self._current_skill_id = 0

    def _sample_episode_skill_id(self):
        """Draw the skill id for one episode (0 == deployment skill w_0)."""
        if not self.skill_enabled or self.skill_K < 1:
            return 0
        if np.random.rand() < self.w0_rollout_ratio:
            return 0
        return int(np.random.randint(1, self.skill_K + 1))

    def _skill_w_from_batch(self, experience_replay):
        """Map a batch's per-transition skill_id to skill codes w (Task 7).

        Returns a [B, d_w] tensor from the (constant) policy codebook, or None
        when the skill library is disabled.
        """
        if not self.skill_enabled:
            return None
        skill_id = torch.as_tensor(
            experience_replay.skill_id, dtype=torch.long, device=self.device
        ).reshape(-1)
        return self.model.codebook.to(self.device)[skill_id]

    def _sample_diversity_layouts(self, n):
        """Sample n (start, obstacles) layouts for the paired diversity batch.

        Uses the current generator (LLM/executor) like ``_make_llm_scene``. Stage
        1's fixed difficulty ladder (Task 12) can supply layouts directly instead.
        """
        layouts = []
        tries = 0
        while len(layouts) < int(n) and tries < int(n) * 20 + 10:
            tries += 1
            if hasattr(self.env, 'sample_valid_tblock_pose'):
                start = np.array(self.env.sample_valid_tblock_pose(), dtype=np.float64)
            else:
                start = np.array(self._random_tblock_pose(), dtype=np.float64)
            if self.acgs_api is not None and self.acgs_api.has_generator:
                obstacles = self.acgs_api.generate_obstacles(start, self.obstacle_num)
            elif self.executor is not None:
                obstacles = self.executor.generate(start, self.obstacle_num)
            else:
                obstacles = []
            obstacles = self._normalize_obstacle_config(obstacles)
            if len(obstacles) != self.obstacle_num:
                continue
            if hasattr(self.env, 'is_obstacle_config_valid') and not self.env.is_obstacle_config_valid(obstacles, start):
                continue
            layouts.append({"start": start.tolist(), "obstacles": obstacles})

        if not layouts:
            # Fallback (no LLM generator): sample layouts from the SAME training
            # distribution (env random obstacles), NOT an easier fixed ladder.
            # Using the easy ladder here gave probe skills an implicit easy-scene
            # curriculum that w_0 never received, making w_0 underperform.
            layouts = self._sample_training_distribution_layouts(int(n))
        return layouts

    def _sample_training_distribution_layouts(self, n):
        """Sample (start, obstacles) from the env's own random-obstacle
        distribution (matches training when use_llm_obstacles=false)."""
        layouts = []
        desk = float(getattr(self.env.task, '_desk_size', 1.0))
        if hasattr(self.env, 'clear_obstacle_config'):
            self.env.clear_obstacle_config()
        for _ in range(int(n)):
            try:
                obs = self.env.reset()
                if not hasattr(self.env, 'get_obstacle_positions'):
                    break
                positions = self.env.get_obstacle_positions()
                obstacles = [{"x": float(p["x"]), "y": float(p["y"])} for p in positions]
                if len(obstacles) != self.obstacle_num:
                    continue
                arr = np.asarray(obs, dtype=np.float64).reshape(-1)
                start = [float(arr[0] * desk), float(arr[1] * desk),
                         float(np.arctan2(arr[3], arr[2]))]
                layouts.append({"start": start, "obstacles": obstacles})
            except Exception:
                continue
        return layouts

    def _run_diversity_batch(self, layouts=None):
        """Task 9: roll out probe skills on shared layouts, compute Form B r_div,
        and write paired transitions (reward_total) into the replay buffer.

        Returns aggregate metrics; a no-op unless the skill library is enabled.
        """
        if not self.skill_enabled or self.skill_K < 1:
            return {}
        if layouts is None:
            layouts = self._sample_diversity_layouts(self.diversity_num_layouts)
        if not layouts:
            return {}
        probe_ids = list(range(1, self.skill_K + 1))
        # Use the current (scheduled) beta_div for this batch's reward_total.
        self._diversity_cfg.beta_div = float(self.beta_div)
        was_training = bool(getattr(self.model, "training", False))
        self.model.eval()
        try:
            rollouts = run_paired_diversity_rollout(
                env=self.env,
                policy=self.model,
                layouts=layouts,
                probe_skill_ids=probe_ids,
                device=self.device,
                max_steps=int(self.cfg.max_steps),
            )
        finally:
            if was_training:
                self.model.train()
        result = compute_diversity_rewards(rollouts, self._diversity_cfg)
        written = write_diversity_transitions(self.replay_buffer, result)
        # Task 11: K_eff via the fixed cluster-entropy protocol (go/no-go metric).
        keff = k_eff_from_rollouts(
            rollouts, threshold=self.keff_cluster_threshold, max_steps=int(self.cfg.max_steps)
        )
        self._last_k_eff = float(keff.get("k_eff", 0.0))
        metrics = dict(result.get("metrics", {}))
        metrics["written"] = int(written)
        metrics["k_eff"] = self._last_k_eff
        metrics["occupied_clusters"] = float(keff.get("occupied_clusters", 0.0))
        metrics["beta_div"] = float(self.beta_div)
        return metrics

    def _update_beta_div(self):
        """Task 10: constraint-balanced beta_div schedule (per eval period).

        Based ONLY on reward_task success and K_eff (never reward_total):
          - success below floor -> multiplicative decrease (diversity hurting task)
          - success ok but K_eff below floor -> increase (skills collapsing)
        Clipped to [beta_div_min, beta_div_max]. No-op during task warm-up.
        """
        if not self.skill_enabled:
            return
        if (self.num_timesteps - self.cfg.rl.warmup) < self.task_warmup_steps:
            self.beta_div = 0.0
            return
        avg_success = float(np.mean(self._recent_success)) if len(self._recent_success) else 0.0
        beta = self.beta_div if self.beta_div > 0.0 else self.beta_div_min
        if avg_success < self.success_floor:
            beta *= 0.5
        elif self._last_k_eff < self.eff_floor:
            beta *= 1.2
        self.beta_div = float(np.clip(beta, self.beta_div_min, self.beta_div_max))
    
    # ===== ACGS 数据采集与闭环 =====

    def _log_acgs(self, message: str):
        """同时输出到控制台与 train.log（Hydra logging）。"""
        print(message)
        LOGGER.info(message)

    def _init_acgs_collection(self, cfg):
        """初始化 ACGS 数据采集基础设施"""
        curriculum_cfg = getattr(cfg, 'curriculum', None)

        # 配置参数
        self.evaluation_interval = int(_cfg_get(
            curriculum_cfg,
            'evolve.evaluation_interval',
            'evaluation_interval',
            default=5000,
        )) if curriculum_cfg else 5000
        self.enable_acgs_loop = bool(_cfg_get(
            curriculum_cfg,
            'evolve.enabled',
            'evolve.enable_acgs_loop',
            'enable_acgs_loop',
            default=True,
        )) if curriculum_cfg else False
        self.feedback_mode = str(_cfg_get(
            curriculum_cfg,
            'evolve.feedback_mode',
            'feedback_mode',
            default='coarse',
        )).strip().lower() if curriculum_cfg else 'coarse'
        if self.feedback_mode not in (
            'static',
            'coarse',
            'attribution',
            'cfa',
            'graph_pattern',
            'skill_signal',
            'boundary_skill',
            'ours',
            'skill_library',
        ):
            LOGGER.warning(
                "[ACGS] Unknown feedback_mode=%s; falling back to coarse",
                self.feedback_mode,
            )
            self.feedback_mode = 'coarse'
        self.trigger_mode = str(_cfg_get(
            curriculum_cfg,
            'evolve.trigger_mode',
            'trigger_mode',
            default='fixed',
        )).strip().lower() if curriculum_cfg else 'fixed'
        if self.trigger_mode != 'fixed':
            LOGGER.warning(
                "[ACGS] trigger_mode=%s is deprecated for main experiments; using fixed",
                self.trigger_mode,
            )
            self.trigger_mode = 'fixed'
        attribution_cfg = _cfg_get(
            curriculum_cfg,
            'attribution',
            'evolve.attribution',
            default=None,
        ) if curriculum_cfg else None
        try:
            self.attribution_config = AttributionConfig.from_dict(
                _cfg_to_plain_dict(attribution_cfg)
            )
        except Exception as exc:
            LOGGER.warning(
                "[ACGS] Invalid attribution config (%s); using defaults",
                exc,
            )
            self.attribution_config = AttributionConfig()

        graph_pattern_cfg = _cfg_get(
            curriculum_cfg,
            'graph_pattern',
            'evolve.graph_pattern',
            default=None,
        ) if curriculum_cfg else None
        graph_pattern_dict = _cfg_to_plain_dict(graph_pattern_cfg)
        try:
            self.graph_pattern_config = PatternDiscoveryConfig(
                num_bins=int(graph_pattern_dict.get('num_bins', 4)),
                max_conditions=int(graph_pattern_dict.get('max_conditions', 3)),
                min_support=int(graph_pattern_dict.get('min_support', 100)),
                min_failure_count=int(graph_pattern_dict.get('min_failure_count', 20)),
                top_k=int(graph_pattern_dict.get('top_k', 8)),
                beam_width=int(graph_pattern_dict.get('beam_width', 250)),
                max_overlap=float(graph_pattern_dict.get('max_overlap', 0.90)),
                min_lift=float(graph_pattern_dict.get('min_lift', 1.0)),
            )
        except Exception as exc:
            LOGGER.warning(
                "[ACGS] Invalid graph_pattern config (%s); using defaults",
                exc,
            )
            self.graph_pattern_config = PatternDiscoveryConfig(top_k=8)

        # Fixed evolve schedule and prompt-only difficulty signal.
        self.first_evolve_episode = _cfg_get(
            curriculum_cfg,
            'evolve.first_episode',
            'evolve.first_evolve_episode',
            'first_evolve_episode',
            default=self.evaluation_interval,
        ) if curriculum_cfg else self.evaluation_interval
        self.first_evolve_episode = max(1, int(self.first_evolve_episode))
        self.max_evolve_count = int(_cfg_get(
            curriculum_cfg,
            'evolve.max_evolve_count',
            'max_evolve_count',
            default=8,
        )) if curriculum_cfg else 8
        self.max_evolve_count = max(0, self.max_evolve_count)
        self.difficulty_success_high = float(_cfg_get(
            curriculum_cfg,
            'evolve.difficulty_success_high',
            'difficulty_success_high',
            default=0.80,
        )) if curriculum_cfg else 0.80
        self.difficulty_success_low = float(_cfg_get(
            curriculum_cfg,
            'evolve.difficulty_success_low',
            'difficulty_success_low',
            default=0.20,
        )) if curriculum_cfg else 0.20
        self.candidate_verifier_enabled = bool(_cfg_get(
            curriculum_cfg,
            'evolve.candidate_verifier.enabled',
            'attribution.candidate_verifier.enabled',
            default=False,
        )) if curriculum_cfg else False
        self.candidate_verifier_num_pose_samples = int(_cfg_get(
            curriculum_cfg,
            'attribution.candidate_verifier.num_pose_samples',
            'evolve.candidate_verifier.num_pose_samples',
            default=500,
        )) if curriculum_cfg else 500
        self.candidate_verifier_seed = int(_cfg_get(
            curriculum_cfg,
            'attribution.candidate_verifier.seed',
            'evolve.candidate_verifier.seed',
            default=int(getattr(getattr(cfg, 'training', None), 'seed', 42) or 42),
        )) if curriculum_cfg else 42
        self.candidate_verifier_timeout_sec = int(_cfg_get(
            curriculum_cfg,
            'evolve.candidate_verifier.timeout_sec',
            'attribution.candidate_verifier.timeout_sec',
            default=5,
        )) if curriculum_cfg else 5
        self.candidate_verifier_min_valid_generation_rate = float(_cfg_get(
            curriculum_cfg,
            'evolve.candidate_verifier.min_valid_generation_rate',
            'attribution.candidate_verifier.min_valid_generation_rate',
            default=0.95,
        )) if curriculum_cfg else 0.95
        self.candidate_verifier_save_audits = bool(_cfg_get(
            curriculum_cfg,
            'evolve.candidate_verifier.save_audits',
            'attribution.candidate_verifier.save_audits',
            default=True,
        )) if curriculum_cfg else True
        self.skill_verifier_use_verifier = bool(_cfg_get(
            curriculum_cfg,
            'skill_verifier.use_verifier',
            'evolve.skill_verifier.use_verifier',
            'evolve.candidate_verifier.use_verifier',
            default=True,
        )) if curriculum_cfg else True
        self.skill_verifier_probe_M = int(_cfg_get(
            curriculum_cfg,
            'skill_verifier.probe_M',
            'evolve.skill_verifier.probe_M',
            'evolve.candidate_verifier.probe_M',
            default=30,
        )) if curriculum_cfg else 30
        self.skill_verifier_probe_K = int(_cfg_get(
            curriculum_cfg,
            'skill_verifier.probe_K',
            'evolve.skill_verifier.probe_K',
            'evolve.candidate_verifier.probe_K',
            default=8,
        )) if curriculum_cfg else 8
        self.skill_verifier_probe_n = int(_cfg_get(
            curriculum_cfg,
            'skill_verifier.probe_n',
            'evolve.skill_verifier.probe_n',
            'evolve.candidate_verifier.probe_n',
            default=0,
        )) if curriculum_cfg else 0
        self.skill_verifier_infeasible_cap = float(_cfg_get(
            curriculum_cfg,
            'skill_verifier.infeasible_cap',
            'evolve.skill_verifier.infeasible_cap',
            'evolve.candidate_verifier.infeasible_cap',
            default=0.30,
        )) if curriculum_cfg else 0.30
        self.skill_verifier_min_score_delta = float(_cfg_get(
            curriculum_cfg,
            'skill_verifier.min_score_delta',
            'evolve.skill_verifier.min_score_delta',
            'evolve.candidate_verifier.min_score_delta',
            default=0.0,
        )) if curriculum_cfg else 0.0
        self.skill_verifier_validity_num_pose_samples = int(_cfg_get(
            curriculum_cfg,
            'skill_verifier.validity_num_pose_samples',
            'evolve.skill_verifier.validity_num_pose_samples',
            'evolve.candidate_verifier.validity_num_pose_samples',
            default=100,
        )) if curriculum_cfg else 100
        # Route B V0 (plan 1A): number of candidate generators per evolve cycle
        # for the best-of-R argmax selection (skill_library mode only).
        self.skill_library_candidates_per_cycle = int(_cfg_get(
            curriculum_cfg,
            'skill_library.candidates_per_cycle',
            'evolve.skill_library.candidates_per_cycle',
            'skill_verifier.candidates_per_cycle',
            default=3,
        )) if curriculum_cfg else 3
        # Route B V0 boundary verifier thresholds (single source of truth: shared by
        # both the verifier `decide_boundary_selection` and the prompt direction).
        self.skill_library_valid_rate_min = float(_cfg_get(
            curriculum_cfg, 'skill_library.valid_rate_min',
            'evolve.skill_library.valid_rate_min', default=0.8,
        )) if curriculum_cfg else 0.8
        self.skill_library_duplicate_rate_max = float(_cfg_get(
            curriculum_cfg, 'skill_library.duplicate_rate_max',
            'evolve.skill_library.duplicate_rate_max', default=0.25,
        )) if curriculum_cfg else 0.25
        self.skill_library_min_boundary_count_delta = int(_cfg_get(
            curriculum_cfg, 'skill_library.min_boundary_count_delta',
            'evolve.skill_library.min_boundary_count_delta', default=4,
        )) if curriculum_cfg else 4
        self.skill_library_r_easy_max = float(_cfg_get(
            curriculum_cfg, 'skill_library.r_easy_max',
            'evolve.skill_library.r_easy_max', default=0.8,
        )) if curriculum_cfg else 0.8
        self.skill_library_r_hard_max = float(_cfg_get(
            curriculum_cfg, 'skill_library.r_hard_max',
            'evolve.skill_library.r_hard_max', default=0.8,
        )) if curriculum_cfg else 0.8
        self.skill_library_tau_saturation = float(_cfg_get(
            curriculum_cfg, 'skill_library.tau_saturation',
            'evolve.skill_library.tau_saturation', default=0.125,
        )) if curriculum_cfg else 0.125
        self.skill_library_target_delta = float(_cfg_get(
            curriculum_cfg, 'skill_library.target_delta',
            'evolve.skill_library.target_delta', default=0.15,
        )) if curriculum_cfg else 0.15
        # Edit-1 switch (DEFAULT False = legacy): drop invalid scenes from the
        # verifier boundary stats so generator-invalidity is not conflated with
        # library failure. Turn on only after the pure-revert baseline is measured.
        self.skill_library_exclude_invalid_from_boundary = bool(_cfg_get(
            curriculum_cfg, 'skill_library.exclude_invalid_from_boundary',
            'evolve.skill_library.exclude_invalid_from_boundary', default=False,
        )) if curriculum_cfg else False
        # PRESERVE_AND_DIVERSIFY swap (points 2/3, default OFF = legacy hold).
        self.skill_library_diversify_on_hold = bool(_cfg_get(
            curriculum_cfg, 'skill_library.diversify_on_hold',
            'evolve.skill_library.diversify_on_hold', default=False,
        )) if curriculum_cfg else False
        self.skill_library_diversify_bc_tolerance = int(_cfg_get(
            curriculum_cfg, 'skill_library.diversify_bc_tolerance',
            'evolve.skill_library.diversify_bc_tolerance', default=2,
        )) if curriculum_cfg else 2
        self.skill_library_diversify_prefer_lower_easy = bool(_cfg_get(
            curriculum_cfg, 'skill_library.diversify_prefer_lower_easy',
            'evolve.skill_library.diversify_prefer_lower_easy', default=False,
        )) if curriculum_cfg else False
        self.skill_library_diversify_r_easy_soft = float(_cfg_get(
            curriculum_cfg, 'skill_library.diversify_r_easy_soft',
            'evolve.skill_library.diversify_r_easy_soft', default=0.5,
        )) if curriculum_cfg else 0.5
        self.skill_library_diversify_easy_eps = float(_cfg_get(
            curriculum_cfg, 'skill_library.diversify_easy_eps',
            'evolve.skill_library.diversify_easy_eps', default=0.05,
        )) if curriculum_cfg else 0.05
        self.skill_signal_probe_M = int(_cfg_get(
            curriculum_cfg,
            'skill_signal.probe_M',
            'evolve.skill_signal.probe_M',
            default=40,
        )) if curriculum_cfg else 40
        self.skill_signal_probe_K = int(_cfg_get(
            curriculum_cfg,
            'skill_signal.probe_K',
            'evolve.skill_signal.probe_K',
            default=12,
        )) if curriculum_cfg else 12
        self.skill_signal_context_n = int(_cfg_get(
            curriculum_cfg,
            'skill_signal.context_n',
            'evolve.skill_signal.context_n',
            default=4,
        )) if curriculum_cfg else 4
        self.skill_signal_include_behavior = bool(_cfg_get(
            curriculum_cfg,
            'skill_signal.include_behavior',
            'evolve.skill_signal.include_behavior',
            default=True,
        )) if curriculum_cfg else True
        self.skill_signal_route_waypoints = int(_cfg_get(
            curriculum_cfg,
            'skill_signal.route_waypoints',
            'evolve.skill_signal.route_waypoints',
            default=5,
        )) if curriculum_cfg else 5
        self.phase4_monitor_enabled = bool(_cfg_get(
            curriculum_cfg,
            'phase4_monitor.enabled',
            'evolve.phase4_monitor.enabled',
            default=False,
        )) if curriculum_cfg else False
        self.phase4_paired_shift_enabled = bool(_cfg_get(
            curriculum_cfg,
            'phase4_monitor.paired_shift.enabled',
            'evolve.phase4_monitor.paired_shift.enabled',
            default=True,
        )) if curriculum_cfg else True
        self.phase4_paired_shift_M = int(_cfg_get(
            curriculum_cfg,
            'phase4_monitor.paired_shift.M',
            'evolve.phase4_monitor.paired_shift.M',
            default=40,
        )) if curriculum_cfg else 40
        self.phase4_paired_shift_K = int(_cfg_get(
            curriculum_cfg,
            'phase4_monitor.paired_shift.K',
            'evolve.phase4_monitor.paired_shift.K',
            default=12,
        )) if curriculum_cfg else 12
        self.phase4_paired_shift_n = int(_cfg_get(
            curriculum_cfg,
            'phase4_monitor.paired_shift.n',
            'evolve.phase4_monitor.paired_shift.n',
            default=0,
        )) if curriculum_cfg else 0
        self.phase4_paired_shift_seed_offset = int(_cfg_get(
            curriculum_cfg,
            'phase4_monitor.paired_shift.seed_offset',
            'evolve.phase4_monitor.paired_shift.seed_offset',
            default=2000,
        )) if curriculum_cfg else 2000
        self.phase4_fresh_validation_enabled = bool(_cfg_get(
            curriculum_cfg,
            'phase4_monitor.fresh_validation.enabled',
            'evolve.phase4_monitor.fresh_validation.enabled',
            default=True,
        )) if curriculum_cfg else True
        self.phase4_fresh_validation_M = int(_cfg_get(
            curriculum_cfg,
            'phase4_monitor.fresh_validation.M',
            'evolve.phase4_monitor.fresh_validation.M',
            default=40,
        )) if curriculum_cfg else 40
        self.phase4_fresh_validation_K = int(_cfg_get(
            curriculum_cfg,
            'phase4_monitor.fresh_validation.K',
            'evolve.phase4_monitor.fresh_validation.K',
            default=12,
        )) if curriculum_cfg else 12
        self.phase4_fresh_validation_n = int(_cfg_get(
            curriculum_cfg,
            'phase4_monitor.fresh_validation.n',
            'evolve.phase4_monitor.fresh_validation.n',
            default=0,
        )) if curriculum_cfg else 0
        self.phase4_fresh_validation_seed_offset = int(_cfg_get(
            curriculum_cfg,
            'phase4_monitor.fresh_validation.seed_offset',
            'evolve.phase4_monitor.fresh_validation.seed_offset',
            default=3000,
        )) if curriculum_cfg else 3000
        self.phase4_bin_coverage_enabled = bool(_cfg_get(
            curriculum_cfg,
            'phase4_monitor.bin_coverage.enabled',
            'evolve.phase4_monitor.bin_coverage.enabled',
            default=True,
        )) if curriculum_cfg else True
        self.phase4_bin_coverage_grid = int(_cfg_get(
            curriculum_cfg,
            'phase4_monitor.bin_coverage.grid',
            'evolve.phase4_monitor.bin_coverage.grid',
            default=8,
        )) if curriculum_cfg else 8
        self.skill_probe_env = None
        self.skill_probe_env_obstacle_size = None
        self.skill_probe_env_obstacle_num = None

        self.max_steps = max(1, int(getattr(cfg, 'max_steps', 10)))

        # 每 episode 的状态
        self._episode_start_pose = None
        self._episode_goal_pose = None
        self._episode_obstacle_config = None
        self._episode_obstacle_z = None
        self._episode_scene_graph = None
        self._episode_termination = 'timeout'
        self._episode_collision_detail = None

        # 批次统计
        self._batch_stats = {'success': 0, 'collision': 0, 'timeout': 0, 'fall': 0}
        self._batch_failure_counts = {k: 0 for k in _FAILURE_KEYS}
        self._batch_episode_count = 0

        # 完整 batch 的轻量 obstacle rollout 记录，用于 attribution。
        self._batch_obstacle_rollout_records = []

        # evolve 历史
        self._evolve_count = 0
        self._total_episode_count = 0  # 全局 episode 计数（不重置）
        self._success_rate_history = []  # 每个批次的成功率历史
        self._just_evolved = False  # 标志：刚刚 evolve 过，下一个 episode 打印障碍物
        self._warmup_completed = False  # 保留 checkpoint 兼容；fixed schedule 下等价于首次触发已到达
        self._first_evolve_triggered = False  # 首次 evolve 是否已经触发（无论成功失败）
        self._last_evolve_episode = 0

        # 全局历史记录（论文分析用，不随 batch 重置）
        self._failure_vector_history = []   # 每个 batch 一条 aggregate failure distribution
        self._acgs_evolve_history = []      # 每次 evolve 一条：预留 accepted/rejected
        self._attribution_history = []      # 每次 attribution/cfa evolve 使用过的 z-space summary

        if self.enable_acgs_loop:
            print(f"\n [1.2] ACGS loop enabled: eval every {self.evaluation_interval} episodes")
            print(f"        feedback_mode={self.feedback_mode}, trigger_mode={self.trigger_mode}")
            print(f"        fixed schedule: first_episode={self.first_evolve_episode}, "
                  f"interval={self.evaluation_interval}, max_evolve_count={self.max_evolve_count}")
            print(f"        difficulty signal: low={self.difficulty_success_low}, "
                  f"high={self.difficulty_success_high}")
            if self.feedback_mode in ('attribution', 'cfa'):
                print(
                    f"        Attribution: min_support={self.attribution_config.min_support}, "
                    f"top_k={self.attribution_config.top_k}, "
                    f"low_k={self.attribution_config.low_k}"
                )
                print(
                    f"        Candidate verifier: enabled={self.candidate_verifier_enabled}, "
                    f"samples={self.candidate_verifier_num_pose_samples}"
                )
            if self.feedback_mode == 'graph_pattern':
                print(
                    f"        Graph patterns: min_support={self.graph_pattern_config.min_support}, "
                    f"min_failure_count={self.graph_pattern_config.min_failure_count}, "
                    f"top_k={self.graph_pattern_config.top_k}, "
                    f"max_conditions={self.graph_pattern_config.max_conditions}"
                )
        else:
            print(f"\n [1.2] ACGS loop disabled")

    def _get_tblock_pose_from_obs(self, obs):
        """从归一化 obs 解码 T-block 位姿 [x, y, theta]"""
        desk_size = self.env.task._desk_size
        x = obs[0, 0] * desk_size
        y = obs[0, 1] * desk_size
        theta = np.arctan2(obs[0, 3], obs[0, 2])
        return [x, y, theta]

    def _get_goal_pose(self):
        task = getattr(self.env, 'task', None)
        target_pose = getattr(task, '_target_block_pos', None)
        if target_pose is None:
            return [float(PUSHT_TARGET_XY[0]), float(PUSHT_TARGET_XY[1]), -np.pi / 4]
        arr = np.asarray(target_pose, dtype=np.float64).reshape(-1)
        if arr.size >= 3:
            return [float(arr[0]), float(arr[1]), float(arr[2])]
        if arr.size >= 2:
            return [float(arr[0]), float(arr[1]), -np.pi / 4]
        return [float(PUSHT_TARGET_XY[0]), float(PUSHT_TARGET_XY[1]), -np.pi / 4]

    def _get_obstacle_half_size(self):
        task = getattr(self.env, 'task', None)
        obstacle_size = getattr(task, '_obstacle_size', None)
        if obstacle_size is None:
            obstacle_size = getattr(self.cfg.env, 'obstacle_size', 0.01)
        return float(obstacle_size)

    def _current_obstacle_size(self):
        """Obstacle half-size used by the active training environment."""
        return self._get_obstacle_half_size()

    def _encode_episode_obstacle_z(self):
        if not self._episode_obstacle_config or not self._episode_start_pose:
            return []

        try:
            zs = pusht_encode_layout(
                self._episode_obstacle_config,
                tblock_pose=self._episode_start_pose,
                target_xy=tuple(self._episode_goal_pose[:2]),
                obstacle_half_size=self._get_obstacle_half_size(),
            )
            return [z.to_dict() for z in zs]
        except Exception as exc:
            LOGGER.warning(f"[ObstacleZ] Failed to encode obstacle layout: {exc}")
            return []

    def _build_episode_scene_graph(self):
        if not self._episode_obstacle_config or not self._episode_start_pose:
            return None

        try:
            return pusht_build_scene_graph(
                self._episode_obstacle_config,
                tblock_pose=self._episode_start_pose,
                target_pose=self._episode_goal_pose,
                obstacle_z=self._episode_obstacle_z,
                obstacle_half_size=self._get_obstacle_half_size(),
            )
        except Exception as exc:
            LOGGER.warning(f"[SceneGraph] Failed to build episode scene graph: {exc}")
            return None

    def _json_sanitize(self, value):
        if isinstance(value, dict):
            return {str(k): self._json_sanitize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_sanitize(v) for v in value]
        if isinstance(value, np.ndarray):
            return self._json_sanitize(value.tolist())
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        return value

    def _make_obstacle_rollout_record(self, replay):
        """Return the lightweight episode record used by JSONL logs and attribution."""
        record = {
            'episode_id': replay.get('episode_id'),
            'start_pose': replay.get('start_pose'),
            'goal_pose': replay.get('goal_pose'),
            'obstacle_config': replay.get('obstacle_config'),
            'obstacle_z': replay.get('obstacle_z'),
            'scene_graph': replay.get('scene_graph'),
            'termination': replay.get('termination'),
            'failure_key': replay.get('failure_key'),
            'reward': replay.get('reward'),
            'steps': replay.get('steps'),
        }
        return self._json_sanitize(record)

    def _append_obstacle_rollout_log(self, replay):
        record = self._make_obstacle_rollout_record(replay)
        try:
            with open(self.obstacle_rollout_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=True) + '\n')
        except Exception as exc:
            LOGGER.warning(f"[ObstacleZ] Failed to append rollout log: {exc}")

    def _on_episode_start(self, obs, tblock_pose=None):
        """Episode 开始时调用：记录起点，清空缓冲区"""
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

        self._episode_goal_pose = self._get_goal_pose()
        self._episode_obstacle_z = self._encode_episode_obstacle_z()
        self._episode_scene_graph = self._build_episode_scene_graph()

    def _on_episode_end(self, episode_reward, episode_length, info):
        """Episode 结束时调用：打包回放片段，更新批次统计"""
        termination = info.get('termination', self._episode_termination)
        collision_detail = info.get('collision_detail', self._episode_collision_detail)

        # 更新批次统计
        self._batch_stats[termination] = self._batch_stats.get(termination, 0) + 1
        self._batch_episode_count += 1
        self._total_episode_count += 1

        # Keep detailed failure_key in rollout JSONL for offline forensic analysis.
        failure_key = self._build_failure_key(termination, collision_detail, episode_length)
        self._batch_failure_counts[failure_key] = self._batch_failure_counts.get(failure_key, 0) + 1

        # 打包回放片段
        replay = {
            'episode_id': self.episode_count,
            'start_pose': self._episode_start_pose,
            'goal_pose': self._episode_goal_pose,
            'obstacle_config': self._episode_obstacle_config,
            'obstacle_z': self._episode_obstacle_z,
            'scene_graph': self._episode_scene_graph,
            'termination': termination,
            'collision_detail': collision_detail,
            'failure_key': failure_key,
            'reward': episode_reward,
            'steps': episode_length,
        }
        self._append_obstacle_rollout_log(replay)
        self._batch_obstacle_rollout_records.append(
            self._make_obstacle_rollout_record(replay)
        )

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
            body = 'tblock'
            if collision_detail and collision_detail.get('type') in ('rod', 'tblock'):
                body = collision_detail['type']
            phase = self._get_collision_phase(episode_length)
            return f'collision_{body}_{phase}'
        if termination in ('timeout', 'fall'):
            return termination
        return termination

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

    def _difficulty_reason(self, success_rate: float, schedule_reason: str) -> str:
        if success_rate > self.difficulty_success_high:
            signal = (
                f"too_easy(sr={success_rate:.3f}>"
                f"{self.difficulty_success_high:.3f})"
            )
        elif success_rate < self.difficulty_success_low:
            signal = (
                f"too_hard(sr={success_rate:.3f}<"
                f"{self.difficulty_success_low:.3f})"
            )
        else:
            signal = (
                f"balanced(sr={success_rate:.3f}, "
                f"range=[{self.difficulty_success_low:.3f},"
                f"{self.difficulty_success_high:.3f}])"
            )
        return f"{schedule_reason}|difficulty={signal}"

    def _append_failure_vector_history(self, failure_vector_result, should_evolve, reason):
        """Record aggregate failure distribution for every closed batch."""
        history_item = {
            'time': time.time(),
            'total_episode_count': self._total_episode_count,
            'batch_episode_count': self._batch_episode_count,
            'batch_stats': copy.deepcopy(self._batch_stats),
            'failure_counts': copy.deepcopy(failure_vector_result.get('counts', {})),
            'failure_distribution': copy.deepcopy(failure_vector_result.get('distribution', {})),
            'dominant_failure_type': failure_vector_result.get('dominant_type', 'success'),
            'should_evolve': bool(should_evolve),
            'reason': reason,
        }
        self._failure_vector_history.append(history_item)

    def _prompt_artifact_dir(self):
        return pathlib.Path(self.output_dir) / "prompts" / (
            f"evolve_{self._evolve_count + 1:03d}_ep_{self._total_episode_count}"
        )

    def _save_evolve_prompt_artifacts(
        self,
        system_prompt,
        user_prompt,
        reason,
        fv_result=None,
        attribution_output_dir=None,
        graph_pattern_output_dir=None,
    ):
        """Save the exact prompts sent to the LLM for this evolve attempt."""
        try:
            prompt_dir = self._prompt_artifact_dir()
            prompt_dir.mkdir(parents=True, exist_ok=True)

            metadata = {
                'time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                'episode_total': int(self._total_episode_count),
                'batch_episodes': int(self._batch_episode_count),
                'evolve_index': int(self._evolve_count + 1),
                'feedback_mode': getattr(self, 'feedback_mode', 'coarse'),
                'reason': reason,
                'dominant_failure_type': (
                    fv_result.get('dominant_failure_type', fv_result.get('dominant_type', 'unknown'))
                    if fv_result else 'unknown'
                ),
                'attribution_output_dir': attribution_output_dir,
                'graph_pattern_output_dir': graph_pattern_output_dir,
                'system_prompt_chars': len(system_prompt or ""),
                'user_prompt_chars': len(user_prompt or ""),
            }

            (prompt_dir / "system_prompt.txt").write_text(
                system_prompt or "",
                encoding="utf-8",
            )
            (prompt_dir / "user_prompt.txt").write_text(
                user_prompt or "",
                encoding="utf-8",
            )
            (prompt_dir / "metadata.json").write_text(
                json.dumps(self._json_sanitize(metadata), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            self._log_acgs(f"[ACGS] Saved prompt artifacts to {prompt_dir}")
            return str(prompt_dir)
        except Exception as e:
            LOGGER.warning(f"[ACGS] Failed to save evolve prompt artifacts: {e}")
            return None

    def _append_evolve_prompt_log(self, prompt, reason, fv_result=None):
        """将每次 evolve 传给 LLM 的完整 prompt 追加写入日志文件。"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            dominant = 'unknown'
            if fv_result is not None:
                dominant = fv_result.get('dominant_failure_type', fv_result.get('dominant_type', 'unknown'))

            header_lines = [
                "\n" + "=" * 100,
                f"[EVOLVE PROMPT] time={timestamp}",
                f"episode_total={self._total_episode_count}, batch_episodes={self._batch_episode_count}, evolve_count={self._evolve_count}",
                f"feedback_mode={getattr(self, 'feedback_mode', 'coarse')}",
                f"reason={reason}",
                f"dominant_failure_type={dominant}",
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

    def _compute_online_attribution_for_evolve(self, reason):
        """Compute z-space attribution over the full current batch for prompt evidence."""
        if getattr(self, 'feedback_mode', 'coarse') not in ('attribution', 'cfa'):
            return None

        records = list(getattr(self, '_batch_obstacle_rollout_records', []))
        if not records:
            self._log_acgs("[ACGS][Attribution] No batch rollout records available")
            return None

        total = max(sum(self._batch_stats.values()), 1)
        metadata = {
            'source': 'online_evolve',
            'feedback_mode': getattr(self, 'feedback_mode', 'coarse'),
            'evolve_index': int(self._evolve_count + 1),
            'total_episode_count': int(self._total_episode_count),
            'batch_episode_count': int(self._batch_episode_count),
            'success_rate': float(self._batch_stats.get('success', 0) / total),
            'trigger_reason': reason,
            'batch_stats': copy.deepcopy(self._batch_stats),
        }

        try:
            result = compute_attribution(
                records,
                self.attribution_config,
                metadata=metadata,
            )
        except Exception as exc:
            LOGGER.warning("[ACGS][Attribution] Failed to compute online attribution: %s", exc)
            return None

        self._log_acgs(
            "[ACGS][Attribution] Computed on "
            f"{result.global_counts.get('num_obstacle_samples', 0)} obstacle samples "
            f"from {result.global_counts.get('num_episodes', 0)} episodes; "
            f"top_cells={len(result.top_cells)}"
        )
        return result

    def _save_online_attribution_outputs(self, attribution_result):
        if attribution_result is None:
            return None

        output_dir = pathlib.Path(self.output_dir) / "failure_attribution" / (
            f"evolve_{self._evolve_count + 1:03d}_ep_{self._total_episode_count}"
        )
        try:
            write_attribution_outputs(attribution_result, output_dir)
            self._log_acgs(f"[ACGS][Attribution] Saved outputs to {output_dir}")
            return str(output_dir)
        except Exception as exc:
            LOGGER.warning("[ACGS][Attribution] Failed to save outputs: %s", exc)
            return None

    def _build_graph_pattern_records(self, records):
        pattern_records = []
        skipped = 0
        for record in records:
            scene_graph = record.get('scene_graph')
            if not isinstance(scene_graph, Mapping):
                skipped += 1
                continue
            try:
                layout_z = extract_layout_z(scene_graph)
            except Exception as exc:
                LOGGER.warning("[ACGS][GraphPattern] Failed to extract layout_z: %s", exc)
                skipped += 1
                continue
            pattern_records.append(
                {
                    'episode_id': record.get('episode_id'),
                    'layout_z': layout_z,
                    'termination': record.get('termination', 'unknown'),
                    'failure_key': record.get(
                        'failure_key',
                        record.get('termination', 'unknown'),
                    ),
                    'reward': record.get('reward'),
                    'steps': record.get('steps'),
                }
            )
        return pattern_records, skipped

    def _compute_online_graph_patterns_for_evolve(self, reason):
        """Mine failure-conditioned scene-graph patterns over the current batch."""
        if getattr(self, 'feedback_mode', 'coarse') != 'graph_pattern':
            return None

        records = list(getattr(self, '_batch_obstacle_rollout_records', []))
        if not records:
            self._log_acgs("[ACGS][GraphPattern] No batch rollout records available")
            return None

        pattern_records, skipped = self._build_graph_pattern_records(records)
        if not pattern_records:
            self._log_acgs(
                "[ACGS][GraphPattern] No scene_graph records available for pattern mining"
            )
            return None

        total = max(sum(self._batch_stats.values()), 1)
        metadata = {
            'source': 'online_evolve',
            'feedback_mode': getattr(self, 'feedback_mode', 'coarse'),
            'evolve_index': int(self._evolve_count + 1),
            'total_episode_count': int(self._total_episode_count),
            'batch_episode_count': int(self._batch_episode_count),
            'success_rate': float(self._batch_stats.get('success', 0) / total),
            'trigger_reason': reason,
            'batch_stats': copy.deepcopy(self._batch_stats),
            'raw_record_count': len(records),
            'scene_graph_record_count': len(pattern_records),
            'skipped_without_scene_graph': skipped,
        }

        try:
            result = discover_failure_graph_patterns(
                pattern_records,
                config=self.graph_pattern_config,
                metadata=metadata,
            )
        except Exception as exc:
            LOGGER.warning("[ACGS][GraphPattern] Failed to mine graph patterns: %s", exc)
            return None

        stats = result.get('global_stats', {})
        self._log_acgs(
            "[ACGS][GraphPattern] Computed on "
            f"{stats.get('num_episodes', len(pattern_records))} scene-graph episodes; "
            f"top_patterns={len(result.get('top_patterns', []))}, "
            f"skipped={skipped}"
        )
        return self._json_sanitize(result)

    def _save_online_graph_pattern_outputs(self, graph_pattern_result):
        if graph_pattern_result is None:
            return None

        output_dir = pathlib.Path(self.output_dir) / "graph_pattern_attribution" / (
            f"evolve_{self._evolve_count + 1:03d}_ep_{self._total_episode_count}"
        )
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "failure_graph_patterns.json").write_text(
                json.dumps(self._json_sanitize(graph_pattern_result), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            self._write_graph_pattern_csv(graph_pattern_result, output_dir / "failure_graph_patterns.csv")
            (output_dir / "failure_graph_pattern_summary.txt").write_text(
                self._build_graph_pattern_summary_text(graph_pattern_result),
                encoding="utf-8",
            )
            self._log_acgs(f"[ACGS][GraphPattern] Saved outputs to {output_dir}")
            return str(output_dir)
        except Exception as exc:
            LOGGER.warning("[ACGS][GraphPattern] Failed to save outputs: %s", exc)
            return None

    def _write_graph_pattern_csv(self, graph_pattern_result, path):
        fields = [
            "rank",
            "support",
            "failure_count",
            "failure_rate",
            "failure_lift",
            "failure_rate_lcb",
            "failure_lift_lcb",
            "dominant_failure_key",
            "dominant_failure_count",
            "dominant_failure_lift",
            "score",
            "conditions",
            "example_episode_ids",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for rank, pattern in enumerate(graph_pattern_result.get("top_patterns", []), start=1):
                row = dict(pattern)
                row["rank"] = rank
                row["conditions"] = " AND ".join(pattern.get("conditions", []))
                row["example_episode_ids"] = json.dumps(pattern.get("example_episode_ids", []))
                writer.writerow({field: row.get(field, "") for field in fields})

    def _build_graph_pattern_summary_text(self, graph_pattern_result):
        metadata = graph_pattern_result.get("metadata", {})
        stats = graph_pattern_result.get("global_stats", {})
        lines = [
            "Failure-Conditioned Graph Pattern Attribution",
            "================================================",
            f"source: {metadata.get('source')}",
            f"feedback_mode: {metadata.get('feedback_mode')}",
            f"evolve_index: {metadata.get('evolve_index')}",
            f"total_episode_count: {metadata.get('total_episode_count')}",
            f"raw_record_count: {metadata.get('raw_record_count')}",
            f"scene_graph_record_count: {metadata.get('scene_graph_record_count')}",
            f"skipped_without_scene_graph: {metadata.get('skipped_without_scene_graph')}",
            f"num_bins: {metadata.get('num_bins')}",
            f"max_conditions: {metadata.get('max_conditions')}",
            f"min_support: {metadata.get('min_support')}",
            f"min_failure_count: {metadata.get('min_failure_count')}",
            "",
            "Global stats:",
            f"- num_episodes: {stats.get('num_episodes')}",
            f"- failure_rate: {float(stats.get('failure_rate', 0.0)):.3f}",
            f"- failure_count: {stats.get('failure_count')}",
            f"- success_count: {stats.get('success_count')}",
            "",
            f"atomic_predicate_count: {graph_pattern_result.get('atomic_predicate_count')}",
            f"candidate_count: {graph_pattern_result.get('candidate_count')}",
            "",
            "Top patterns:",
        ]
        for idx, pattern in enumerate(graph_pattern_result.get("top_patterns", []), start=1):
            lines.extend(
                [
                    "",
                    f"{idx}. {' AND '.join(pattern.get('conditions', []))}",
                    f"   support={pattern.get('support')}, "
                    f"failure_rate={float(pattern.get('failure_rate', 0.0)):.3f}, "
                    f"lift={float(pattern.get('failure_lift', 0.0)):.3f}, "
                    f"lcb_lift={float(pattern.get('failure_lift_lcb', 0.0)):.3f}",
                    f"   dominant_failure={pattern.get('dominant_failure_key')} "
                    f"({pattern.get('dominant_failure_count')}), "
                    f"dominant_lift={float(pattern.get('dominant_failure_lift', 0.0)):.3f}",
                ]
            )
        return "\n".join(lines) + "\n"

    def _build_candidate_verifier_callback(
        self,
        current_generator_code,
        attribution_result,
        attribution_output_dir,
        evolve_record,
    ):
        if getattr(self, 'feedback_mode', 'coarse') in ('skill_signal', 'boundary_skill', 'ours', 'skill_library'):
            return self._build_skill_verifier_callback(
                current_generator_code=current_generator_code,
                attribution_output_dir=attribution_output_dir,
                evolve_record=evolve_record,
            )

        if (
            not self.candidate_verifier_enabled
            or getattr(self, 'feedback_mode', 'coarse') != 'attribution'
            or current_generator_code is None
            or attribution_result is None
        ):
            return None

        total = max(sum(self._batch_stats.values()), 1)
        success_rate = float(self._batch_stats.get('success', 0) / total)
        audit_root = pathlib.Path(
            attribution_output_dir
            or (
                pathlib.Path(self.output_dir)
                / "failure_attribution"
                / f"evolve_{self._evolve_count + 1:03d}_ep_{self._total_episode_count}"
            )
        ) / "candidate_audits"
        verifier_seed = int(self.candidate_verifier_seed + self._evolve_count)
        verifier_config = CandidateVerifierConfig(
            num_pose_samples=max(1, int(self.candidate_verifier_num_pose_samples)),
            obstacle_num=int(self.obstacle_num),
            seed=verifier_seed,
            success_rate=success_rate,
            success_low=float(self.difficulty_success_low),
            success_high=float(self.difficulty_success_high),
            min_valid_generation_rate=float(self.candidate_verifier_min_valid_generation_rate),
            timeout_sec=max(1, int(self.candidate_verifier_timeout_sec)),
        )
        evolve_record['candidate_verifier_enabled'] = True
        evolve_record['candidate_verifier_audit_dir'] = str(audit_root)
        evolve_record['candidate_verifier_num_pose_samples'] = verifier_config.num_pose_samples
        candidate_audits = []
        evolve_record['candidate_verifier_audits'] = candidate_audits

        def _callback(candidate_code, attempt_idx):
            result = verify_pusht_candidate(
                current_generator_code=current_generator_code,
                candidate_generator_code=candidate_code,
                attribution_result=attribution_result,
                config=verifier_config,
            )
            artifact_paths = {}
            if self.candidate_verifier_save_audits:
                artifact_paths = save_verification_artifacts(
                    output_dir=audit_root,
                    attempt_idx=attempt_idx,
                    candidate_code=candidate_code,
                    result=result,
                )
            record = {
                'attempt': int(attempt_idx),
                'accepted': bool(result.accepted),
                'reason': result.reason,
                'direction': result.direction,
                'current_score': float(result.current.score),
                'candidate_score': float(result.candidate.score),
                'score_delta': float(result.score_delta),
                **artifact_paths,
            }
            candidate_audits.append(record)
            self._log_acgs(
                "[ACGS][Verifier] attempt "
                f"{attempt_idx}: accepted={result.accepted}, "
                f"direction={result.direction}, "
                f"score {result.current.score:.4f}->{result.candidate.score:.4f} "
                f"(delta={result.score_delta:+.4f}), reason={result.reason}"
            )
            return result

        return _callback

    def _get_skill_probe_env(self):
        obstacle_size = float(self._current_obstacle_size())
        obstacle_num = int(self.obstacle_num)
        if (
            self.skill_probe_env is None
            or self.skill_probe_env_obstacle_size != obstacle_size
            or self.skill_probe_env_obstacle_num != obstacle_num
        ):
            self.skill_probe_env = build_pusht_env_for_probe(
                obstacle_num=obstacle_num,
                obstacle_size=obstacle_size,
            )
            self.skill_probe_env_obstacle_size = obstacle_size
            self.skill_probe_env_obstacle_num = obstacle_num
            self._log_acgs(
                "[ACGS][SkillSignal] Created probe env "
                f"(obstacle_num={obstacle_num}, obstacle_size={obstacle_size})"
            )
        return self.skill_probe_env

    def _compute_skill_signal_for_evolve(self, current_generator_code, evolve_record):
        if getattr(self, 'feedback_mode', 'coarse') not in ('skill_signal', 'boundary_skill', 'ours', 'skill_library'):
            return None
        if current_generator_code is None:
            evolve_record['skill_signal_error'] = 'current_generator_code_none'
            self._log_acgs(
                "[ACGS][SkillSignal] feedback_mode="
                f"{getattr(self, 'feedback_mode', 'coarse')} but current generator "
                "code is None; skipping skill-signal probe and falling back to "
                "coarse prompt. Verifier/phase4 will also be skipped."
            )
            return None

        from DIVO.curriculum.candidate_verifier import SandboxPushTExecutor
        from DIVO.curriculum.skill_signal import extract_skill_signals

        obstacle_size = float(self._current_obstacle_size())
        executor = SandboxPushTExecutor(obstacle_size=obstacle_size)
        if not executor.load(current_generator_code):
            evolve_record['skill_signal_error'] = 'current_generator_load_failed'
            self._log_acgs("[ACGS][SkillSignal] Failed to load current generator for context probe")
            return None

        probe_env = self._get_skill_probe_env()
        # Route B V0 (skill_library): probe the current generator with the trained
        # W_probe library (Task 16) to derive the p(1-p) boundary signal for both
        # the LLM prompt context and the evolution direction.
        is_skill_library = getattr(self, 'feedback_mode', 'coarse') == 'skill_library'
        # Deviation-2 REVERTED: the context/direction probe uses random starts +
        # 'resample' (measures library capability on VALID scenes of G_t), same as
        # the legacy path, differing only in probe_source='w_probe'. Forcing this
        # probe onto fixed_starts+'zero' (the earlier double-probe 同源) mis-read a
        # hard G_0 as fully infeasible (realized=0), causing a ~17-round cold-start
        # RELAX collapse and D=0.00 at deployment. The verifier keeps its OWN
        # fixed-start paired probe of G_t + candidates (see
        # _evolve_skill_library_best_of_r); paired starts are only needed there.
        probe_M = max(1, int(self.skill_signal_probe_M))
        probe_K = max(1, int(self.skill_signal_probe_K))
        probe_seed = int(self.candidate_verifier_seed + 1000 + self._evolve_count)
        probe_fixed_starts = None
        probe_invalid_policy = 'resample'
        was_training = bool(getattr(self.model, "training", False))
        try:
            result = extract_skill_signals(
                env=probe_env,
                policy=self.model,
                generator=executor,
                device=self.device,
                latent_dim=int(self.cfg.policy.encoder_net.out_chan),
                K=probe_K,
                M=probe_M,
                n=max(0, int(self.skill_signal_context_n)),
                num_obstacles=int(self.obstacle_num),
                max_steps=max(1, int(self.max_steps)),
                seed=probe_seed,
                generator_timeout_sec=max(1, int(self.candidate_verifier_timeout_sec)),
                route_waypoints=max(1, int(self.skill_signal_route_waypoints)),
                include_behavior=bool(self.skill_signal_include_behavior),
                progress_every=0,
                probe_source=('w_probe' if is_skill_library else 'random_z'),
                fixed_starts=probe_fixed_starts,
                invalid_scene_policy=probe_invalid_policy,
            )
        finally:
            if was_training and hasattr(self.model, "train"):
                self.model.train()

        result['probe_config'] = {
            'probe_M': int(self.skill_signal_probe_M),
            'probe_K': int(self.skill_signal_probe_K),
            'context_n': int(self.skill_signal_context_n),
            'include_behavior': bool(self.skill_signal_include_behavior),
            'route_waypoints': int(self.skill_signal_route_waypoints),
            'obstacle_num': int(self.obstacle_num),
            'obstacle_size': float(obstacle_size),
            'max_steps': int(self.max_steps),
            'seed': int(probe_seed),
        }
        profile = result.get('difficulty_profile', {})
        evolve_record['skill_signal_probe_config'] = result['probe_config']
        evolve_record['skill_signal_profile'] = {
            'mean_realized': float(profile.get('mean_realized', 0.0)),
            'frac_infeasible': float(profile.get('frac_infeasible', 0.0)),
            'mean_feasible': float(profile.get('mean_feasible', 0.0)),
            'mean_deployed': float(profile.get('mean_deployed', 0.0)),
            'mean_lv': float(profile.get('mean_lv', 0.0)),
            'realized_hist': profile.get('realized_hist', {}),
            'sampling': profile.get('sampling', {}),
        }
        self._log_acgs(
            "[ACGS][SkillSignal] context profile: "
            f"mean_realized={float(profile.get('mean_realized', 0.0)):.3f}, "
            f"frac_infeasible={float(profile.get('frac_infeasible', 0.0)):.3f}, "
            f"mean_lv={float(profile.get('mean_lv', 0.0)):.3f}"
        )

        if is_skill_library:
            boundary_signal = dict(result.get('boundary_signal', {}) or {})
            direction = self._skill_library_direction(boundary_signal)
            boundary_signal['evolution_direction'] = direction
            result['boundary_signal'] = boundary_signal
            # Surface the direction + boundary stats to the LLM prompt context.
            result.setdefault('skill_library', {})['boundary_signal'] = boundary_signal
            result['skill_library']['evolution_direction'] = direction
            evolve_record['skill_signal_boundary'] = boundary_signal
            evolve_record['skill_library_direction'] = direction
            self._log_acgs(
                "[ACGS][SkillLibrary] boundary signal: "
                f"boundary_count={boundary_signal.get('boundary_count', 0)} "
                f"boundary_rate={boundary_signal.get('boundary_rate', 0.0):.3f} "
                f"r_hard={boundary_signal.get('r_hard', 0.0):.3f} "
                f"r_easy={boundary_signal.get('r_easy', 0.0):.3f} "
                f"mean_b={boundary_signal.get('mean_b', 0.0):.3f} "
                f"w0_success={boundary_signal.get('w0_success_rate', 0.0):.3f} "
                f"-> direction={direction}"
            )
        return result

    def _skill_library_direction(self, boundary_signal):
        """Eurekaverse-style three-way evolution direction (Task 18.2 / component 8).

        Priority: fix validity first, then RELAX (too hard) / HARDEN (too easy),
        else PRESERVE_AND_DIVERSIFY (enough boundary mass). Thresholds mirror the
        symmetric saturation escape (r_easy_max / r_hard_max, valid_rate_min).
        """
        v_min = float(self.skill_library_valid_rate_min)
        r_easy_max = float(self.skill_library_r_easy_max)
        r_hard_max = float(self.skill_library_r_hard_max)
        valid_rate = float(boundary_signal.get('valid_rate', 0.0))
        r_hard = float(boundary_signal.get('r_hard', 0.0))
        r_easy = float(boundary_signal.get('r_easy', 0.0))
        if valid_rate < v_min:
            return 'FIX_VALIDITY'
        if r_hard >= r_hard_max:
            return 'RELAX'
        if r_easy >= r_easy_max:
            return 'HARDEN'
        return 'PRESERVE_AND_DIVERSIFY'

    def _load_skill_signal_executor(self, generator_code, obstacle_size, label, evolve_record=None):
        from DIVO.curriculum.candidate_verifier import SandboxPushTExecutor

        executor = SandboxPushTExecutor(obstacle_size=float(obstacle_size))
        if not executor.load(generator_code):
            if evolve_record is not None:
                evolve_record.setdefault('phase4_monitor_errors', []).append(
                    f'{label}_generator_load_failed'
                )
            self._log_acgs(f"[ACGS][Phase4] Failed to load {label} generator for monitor probe")
            return None
        return executor

    def _run_phase4_paired_probe_pair(
        self,
        old_code,
        new_code,
        *,
        M,
        K,
        n,
        seed,
        label,
        evolve_record,
    ):
        if not old_code or not new_code:
            return None, None, {}

        from DIVO.curriculum.skill_signal import (
            build_z_bank,
            extract_skill_signals,
            sample_fixed_starts,
        )

        obstacle_size = float(self._current_obstacle_size())
        old_executor = self._load_skill_signal_executor(
            old_code,
            obstacle_size=obstacle_size,
            label=f'{label}_old',
            evolve_record=evolve_record,
        )
        new_executor = self._load_skill_signal_executor(
            new_code,
            obstacle_size=obstacle_size,
            label=f'{label}_new',
            evolve_record=evolve_record,
        )
        if old_executor is None or new_executor is None:
            return None, None, {}

        M = max(1, int(M))
        K = max(1, int(K))
        n = max(0, int(n))
        seed = int(seed)
        probe_env = self._get_skill_probe_env()
        latent_dim = int(self.cfg.policy.encoder_net.out_chan)
        fixed_starts = sample_fixed_starts(
            env=probe_env,
            M=M,
            seed=seed,
        )
        z_bank = build_z_bank(
            K=K,
            latent_dim=latent_dim,
            seed=seed,
            device=self.device,
        )
        common_kwargs = {
            'env': probe_env,
            'policy': self.model,
            'device': self.device,
            'latent_dim': latent_dim,
            'K': K,
            'M': M,
            'n': n,
            'num_obstacles': int(self.obstacle_num),
            'max_steps': max(1, int(self.max_steps)),
            'seed': seed,
            'generator_timeout_sec': max(1, int(self.candidate_verifier_timeout_sec)),
            'route_waypoints': max(1, int(self.skill_signal_route_waypoints)),
            'include_behavior': False,
            'fixed_starts': fixed_starts,
            'z_bank': z_bank,
            'invalid_scene_policy': 'zero',
            'progress_every': 0,
        }
        was_training = bool(getattr(self.model, "training", False))
        try:
            before = extract_skill_signals(
                generator=old_executor,
                **common_kwargs,
            )
            after = extract_skill_signals(
                generator=new_executor,
                **common_kwargs,
            )
        finally:
            if was_training and hasattr(self.model, "train"):
                self.model.train()

        probe_config = {
            'label': str(label),
            'probe_M': int(M),
            'probe_K': int(K),
            'probe_n': int(n),
            'obstacle_num': int(self.obstacle_num),
            'obstacle_size': float(obstacle_size),
            'max_steps': int(self.max_steps),
            'seed': int(seed),
            'paired_starts_and_z': True,
        }
        before['probe_config'] = dict(probe_config)
        after['probe_config'] = dict(probe_config)
        return before, after, probe_config

    def _compute_phase4_monitor_after_accept(
        self,
        old_code,
        new_code,
        context_signal,
        evolve_record,
    ):
        if not self.phase4_monitor_enabled:
            return None
        # skill_library (Route B V0) intentionally excluded: the phase4 paired
        # probe uses the legacy random-z rollout (no skill code), which is invalid
        # for a skill_enabled policy. V0 monitors via the W_probe boundary signal.
        if getattr(self, 'feedback_mode', 'coarse') not in ('skill_signal', 'boundary_skill', 'ours'):
            return None

        paired_before = paired_after = None
        paired_config = {}
        if self.phase4_paired_shift_enabled:
            paired_seed = int(
                self.candidate_verifier_seed
                + self.phase4_paired_shift_seed_offset
                + self._evolve_count
            )
            paired_before, paired_after, paired_config = self._run_phase4_paired_probe_pair(
                old_code,
                new_code,
                M=self.phase4_paired_shift_M,
                K=self.phase4_paired_shift_K,
                n=self.phase4_paired_shift_n,
                seed=paired_seed,
                label='paired_shift',
                evolve_record=evolve_record,
            )

        fresh_before = fresh_after = None
        fresh_config = {}
        if self.phase4_fresh_validation_enabled:
            fresh_seed = int(
                self.candidate_verifier_seed
                + self.phase4_fresh_validation_seed_offset
                + self._evolve_count
            )
            fresh_before, fresh_after, fresh_config = self._run_phase4_paired_probe_pair(
                old_code,
                new_code,
                M=self.phase4_fresh_validation_M,
                K=self.phase4_fresh_validation_K,
                n=self.phase4_fresh_validation_n,
                seed=fresh_seed,
                label='fresh_validation',
                evolve_record=evolve_record,
            )

        if paired_before is None or paired_after is None:
            return None

        record = build_phase4_record(
            evolve_index=int(self._evolve_count + 1),
            total_episode_count=int(self._total_episode_count),
            context_signal=context_signal,
            verifier_audits=evolve_record.get('candidate_verifier_audits', []),
            paired_before=paired_before,
            paired_after=paired_after,
            fresh_before=fresh_before,
            fresh_after=fresh_after,
            paired_config=paired_config,
            fresh_config=fresh_config,
            coverage_enabled=bool(self.phase4_bin_coverage_enabled),
            coverage_grid=max(1, int(self.phase4_bin_coverage_grid)),
            infeasible_cap=float(self.skill_verifier_infeasible_cap),
        )
        evolve_record['phase4_monitor'] = record
        self._append_phase4_monitor_log(record)

        judgement = record.get('paired_shift', {}).get('judgement', {})
        self._log_acgs(
            "[ACGS][Phase4] paired shift: "
            f"state={judgement.get('state_before')}, "
            f"direction_ok={judgement.get('direction_ok')}, "
            f"d_realized={float(judgement.get('mean_realized_delta', 0.0)):+.3f}, "
            f"d_lv={float(judgement.get('mean_lv_delta', 0.0)):+.3f}, "
            f"d_infeasible={float(judgement.get('frac_infeasible_delta', 0.0)):+.3f}"
        )
        fresh = record.get('fresh_validation')
        if fresh:
            fresh_judgement = fresh.get('judgement', {})
            self._log_acgs(
                "[ACGS][Phase4] fresh validation: "
                f"direction_ok={fresh_judgement.get('direction_ok')}, "
                f"d_realized={float(fresh_judgement.get('mean_realized_delta', 0.0)):+.3f}, "
                f"d_lv={float(fresh_judgement.get('mean_lv_delta', 0.0)):+.3f}, "
                f"d_infeasible={float(fresh_judgement.get('frac_infeasible_delta', 0.0)):+.3f}"
            )
        return record

    def _append_phase4_monitor_log(self, record):
        path = pathlib.Path(self.output_dir) / "phase4_monitor.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")

    def _build_skill_verifier_callback(
        self,
        current_generator_code,
        attribution_output_dir,
        evolve_record,
    ):
        if not self.candidate_verifier_enabled or current_generator_code is None:
            return None

        audit_root = pathlib.Path(
            attribution_output_dir
            or (
                pathlib.Path(self.output_dir)
                / "skill_signal_verifier"
                / f"evolve_{self._evolve_count + 1:03d}_ep_{self._total_episode_count}"
            )
        ) / "candidate_audits"
        verifier_seed = int(self.candidate_verifier_seed + self._evolve_count)
        obstacle_size = self._current_obstacle_size()
        # Route B V0 (skill_library) uses the W_probe skill library + single-scalar
        # boundary_count argmax + saturation escape; legacy modes keep random-z +
        # the four-gate learnable-frontier rule (explicit, so unchanged).
        is_skill_library = getattr(self, 'feedback_mode', 'coarse') == 'skill_library'
        verifier_config = SkillVerifierConfig(
            use_verifier=bool(self.skill_verifier_use_verifier),
            obstacle_num=int(self.obstacle_num),
            obstacle_size=float(obstacle_size),
            probe_M=max(1, int(self.skill_verifier_probe_M)),
            probe_K=max(1, int(self.skill_verifier_probe_K)),
            probe_n=max(0, int(self.skill_verifier_probe_n)),
            max_steps=max(1, int(self.max_steps)),
            seed=verifier_seed,
            infeasible_cap=float(self.skill_verifier_infeasible_cap),
            min_score_delta=float(self.skill_verifier_min_score_delta),
            min_valid_generation_rate=float(self.candidate_verifier_min_valid_generation_rate),
            validity_num_pose_samples=max(1, int(self.skill_verifier_validity_num_pose_samples)),
            timeout_sec=max(1, int(self.candidate_verifier_timeout_sec)),
            probe_source=('w_probe' if is_skill_library else 'random_z'),
            criterion=('boundary_count_argmax_escape' if is_skill_library else 'learnable_frontier_shift'),
        )
        evolve_record['candidate_verifier_enabled'] = True
        evolve_record['candidate_verifier_type'] = 'skill_signal'
        evolve_record['candidate_verifier_audit_dir'] = str(audit_root)
        evolve_record['candidate_verifier_probe_M'] = verifier_config.probe_M
        evolve_record['candidate_verifier_probe_K'] = verifier_config.probe_K
        evolve_record['candidate_verifier_obstacle_size'] = verifier_config.obstacle_size
        evolve_record['candidate_verifier_infeasible_cap'] = verifier_config.infeasible_cap
        evolve_record['candidate_verifier_validity_num_pose_samples'] = (
            verifier_config.validity_num_pose_samples
        )
        candidate_audits = []
        evolve_record['candidate_verifier_audits'] = candidate_audits

        probe_env = self._get_skill_probe_env()
        session = SkillCandidateVerifierSession(
            env=probe_env,
            policy=self.model,
            device=self.device,
            latent_dim=int(self.cfg.policy.encoder_net.out_chan),
            current_generator_code=current_generator_code,
            config=verifier_config,
            current_signal_result=None,
        )
        evolve_record['candidate_verifier_current_score'] = float(session.current_score.score)
        evolve_record['candidate_verifier_current_frac_infeasible'] = float(
            session.current_score.frac_infeasible
        )

        def _callback(candidate_code, attempt_idx):
            result = session.verify(candidate_code, attempt_idx=attempt_idx)
            artifact_paths = {}
            if self.candidate_verifier_save_audits:
                artifact_paths = self._save_skill_verification_artifacts(
                    output_dir=audit_root,
                    attempt_idx=attempt_idx,
                    candidate_code=candidate_code,
                    result=result,
                )
            record = {
                'attempt': int(attempt_idx),
                'accepted': bool(result.accepted),
                'reason': result.reason,
                'current_score': float(result.current.score),
                'candidate_score': float(result.candidate.score),
                'score_delta': float(result.score_delta),
                'candidate_frac_infeasible': float(result.candidate.frac_infeasible),
                'candidate_valid_generation_rate': float(result.candidate.valid_generation_rate),
                'frontier_shift': dict(getattr(result, 'frontier_shift', {}) or {}),
                **artifact_paths,
            }
            candidate_audits.append(record)
            frontier_shift = dict(getattr(result, 'frontier_shift', {}) or {})
            frontier_distance_delta = frontier_shift.get('frontier_distance_delta', 0.0)
            self._log_acgs(
                "[ACGS][SkillVerifier] attempt "
                f"{attempt_idx}: accepted={result.accepted}, "
                f"score {result.current.score:.4f}->{result.candidate.score:.4f} "
                f"(delta={result.score_delta:+.4f}), "
                f"frontier_dist_delta={float(frontier_distance_delta):+.4f}, "
                f"infeasible={result.candidate.frac_infeasible:.3f}, "
                f"valid={result.candidate.valid_generation_rate:.3f}, "
                f"reason={result.reason}"
            )
            return result

        return _callback

    def _evolve_skill_library_best_of_r(
        self,
        *,
        failure_vector_result,
        reason,
        current_generator_code,
        skill_signal_result,
        evolve_record,
    ):
        """Route B V0 best-of-R evolution (plan 1A).

        Samples R candidate generators from ONE evolve prompt, scores the current
        generator and every candidate with the W_probe boundary verifier on the
        SAME fixed starts (paired), and picks the winner via
        ``decide_boundary_selection`` (single-scalar boundary_count argmax over
        ``{G_t} ∪ candidates`` + symmetric saturation escape). Returns the accepted
        candidate code, or None to hold G_t.
        """
        from DIVO.curriculum.candidate_verifier import (
            SkillVerifierConfig,
            SkillCandidateVerifierSession,
            decide_boundary_selection,
        )

        if current_generator_code is None:
            evolve_record['skill_library_error'] = 'current_generator_code_none'
            return None

        R = max(1, int(self.skill_library_candidates_per_cycle))
        candidates = self.acgs_api.generate_candidates(
            batch_stats=dict(self._batch_stats),
            fv_result=failure_vector_result,
            reason=reason,
            current_generator_code=current_generator_code,
            num_obstacles=self.obstacle_num,
            feedback_mode=self.feedback_mode,
            R=R,
            skill_signal_result=skill_signal_result,
            include_behavior=bool(self.skill_signal_include_behavior),
        )
        evolve_record['skill_library_num_candidates'] = len(candidates)
        if not candidates:
            self._log_acgs("[ACGS][SkillLibrary] No sanity-passing candidates; holding G_t")
            return None

        verifier_seed = int(self.candidate_verifier_seed + self._evolve_count)
        obstacle_size = self._current_obstacle_size()
        verifier_config = SkillVerifierConfig(
            use_verifier=True,
            obstacle_num=int(self.obstacle_num),
            obstacle_size=float(obstacle_size),
            probe_M=max(1, int(self.skill_verifier_probe_M)),
            probe_K=max(1, int(self.skill_verifier_probe_K)),
            probe_n=max(0, int(self.skill_verifier_probe_n)),
            max_steps=max(1, int(self.max_steps)),
            seed=verifier_seed,
            min_valid_generation_rate=float(self.candidate_verifier_min_valid_generation_rate),
            validity_num_pose_samples=max(1, int(self.skill_verifier_validity_num_pose_samples)),
            timeout_sec=max(1, int(self.candidate_verifier_timeout_sec)),
            probe_source='w_probe',
            criterion='boundary_count_argmax_escape',
            valid_rate_min=float(self.skill_library_valid_rate_min),
            duplicate_rate_max=float(self.skill_library_duplicate_rate_max),
            min_boundary_count_delta=int(self.skill_library_min_boundary_count_delta),
            r_easy_max=float(self.skill_library_r_easy_max),
            r_hard_max=float(self.skill_library_r_hard_max),
            tau_saturation=float(self.skill_library_tau_saturation),
            target_delta=float(self.skill_library_target_delta),
            candidates_per_cycle=int(self.skill_library_candidates_per_cycle),
            exclude_invalid_from_boundary=bool(self.skill_library_exclude_invalid_from_boundary),
            diversify_on_hold=bool(self.skill_library_diversify_on_hold),
            diversify_bc_tolerance=int(self.skill_library_diversify_bc_tolerance),
            diversify_prefer_lower_easy=bool(self.skill_library_diversify_prefer_lower_easy),
            diversify_r_easy_soft=float(self.skill_library_diversify_r_easy_soft),
            diversify_easy_eps=float(self.skill_library_diversify_easy_eps),
        )

        probe_env = self._get_skill_probe_env()
        was_training = bool(getattr(self.model, "training", False))
        try:
            session = SkillCandidateVerifierSession(
                env=probe_env,
                policy=self.model,
                device=self.device,
                latent_dim=int(self.cfg.policy.encoder_net.out_chan),
                current_generator_code=current_generator_code,
                config=verifier_config,
                # Deviation-2 REVERTED: do NOT reuse the (random-start) context
                # probe. The verifier re-probes G_t on its OWN fixed starts so that
                # G_t and every candidate are scored paired on identical starts.
                # The prompt direction still comes from the separate context probe;
                # a minor direction/acceptance divergence is benign (00.34.42 口径).
                current_signal_result=None,
            )
            cur_sig = dict(session.current_score.difficulty_profile.get('boundary_signal', {}) or {})
            cand_sigs = []
            for i, code in enumerate(candidates):
                score = session._score_code(
                    code=code, label=f'candidate_{i + 1}', cached_signal_result=None,
                )
                cand_sigs.append(dict(score.difficulty_profile.get('boundary_signal', {}) or {}))
        finally:
            if was_training and hasattr(self.model, "train"):
                self.model.train()

        decision = decide_boundary_selection(cur_sig, cand_sigs, verifier_config)
        evolve_record['skill_library_current_boundary'] = cur_sig
        evolve_record['skill_library_candidate_boundaries'] = cand_sigs
        evolve_record['skill_library_decision'] = decision
        self._log_acgs(
            "[ACGS][SkillLibrary] best-of-%d: action=%s reason=%s chosen=%s "
            "cur_bc=%s cand_bc=%s saturated=%s(%s)" % (
                len(candidates), decision.get('action'), decision.get('reason'),
                decision.get('chosen_index'), decision.get('current_boundary_count'),
                [int(s.get('boundary_count', 0)) for s in cand_sigs],
                decision.get('saturated'), decision.get('saturation_direction'),
            )
        )
        if decision.get('action') == 'replace' and decision.get('chosen_index') is not None:
            chosen_code = candidates[int(decision['chosen_index'])]
            if self.acgs_api.load_generator_code(chosen_code):
                return chosen_code
            self._log_acgs("[ACGS][SkillLibrary] chosen candidate failed to load; holding G_t")
        return None

    def _save_skill_verification_artifacts(
        self,
        output_dir,
        attempt_idx,
        candidate_code,
        result,
    ):
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"candidate_{int(attempt_idx):03d}"
        code_path = output_dir / f"{stem}.py"
        audit_path = output_dir / f"{stem}_skill_audit.json"
        feedback_path = output_dir / f"{stem}_feedback.txt"
        code_path.write_text(candidate_code, encoding="utf-8")
        audit_path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        feedback_path.write_text(result.feedback_text + "\n", encoding="utf-8")
        return {
            "code_path": str(code_path),
            "audit_path": str(audit_path),
            "feedback_path": str(feedback_path),
        }

    def _execute_acgs_evolve(self, reason, failure_vector_result):
        evolve_record = {
            'time': time.time(),
            'total_episode_count': self._total_episode_count,
            'batch_episode_count': self._batch_episode_count,
            'reason': reason,
            'dominant_failure_type': failure_vector_result.get('dominant_type', 'success'),
            'feedback_mode': getattr(self, 'feedback_mode', 'coarse'),
            'llm_generation_success': False,
            'load_success': False,
            'accepted': False,
            'rejected_reason': None,
            'error': None,
        }

        if self.acgs_api is None:
            self._log_acgs("[ACGS] No ACGS API, skipping evolve")
            evolve_record['rejected_reason'] = 'no_acgs_api'
            self._acgs_evolve_history.append(evolve_record)
            self._clear_pending_evolve_state()
            self._reset_batch()
            return

        try:
            attribution_result_obj = self._compute_online_attribution_for_evolve(reason)
            attribution_output_dir = self._save_online_attribution_outputs(attribution_result_obj)
            graph_pattern_result = self._compute_online_graph_patterns_for_evolve(reason)
            graph_pattern_output_dir = self._save_online_graph_pattern_outputs(graph_pattern_result)
            attribution_result = None
            coverage_summary = None
            attribution_history = None
            if attribution_result_obj is not None:
                attribution_result = attribution_result_obj.to_dict()
                coverage_summary = attribution_result_obj.coverage.to_dict()
                attribution_history = list(self._attribution_history[-3:])
                evolve_record['attribution_output_dir'] = attribution_output_dir
                evolve_record['attribution_samples'] = attribution_result_obj.global_counts.get(
                    'num_obstacle_samples',
                    0,
                )
                evolve_record['attribution_top_cells'] = len(attribution_result_obj.top_cells)
            if graph_pattern_result is not None:
                graph_metadata = graph_pattern_result.get('metadata', {})
                evolve_record['graph_pattern_output_dir'] = graph_pattern_output_dir
                evolve_record['graph_pattern_scene_graph_records'] = graph_metadata.get(
                    'scene_graph_record_count',
                    0,
                )
                evolve_record['graph_pattern_skipped_without_scene_graph'] = graph_metadata.get(
                    'skipped_without_scene_graph',
                    0,
                )
                evolve_record['graph_pattern_top_patterns'] = len(
                    graph_pattern_result.get('top_patterns', [])
                )

            current_generator_code = self._extract_clean_generate_obstacles()
            skill_signal_result = self._compute_skill_signal_for_evolve(
                current_generator_code=current_generator_code,
                evolve_record=evolve_record,
            )

            system_prompt, user_prompt = self.acgs_api.get_prompt_text(
                batch_stats=dict(self._batch_stats),
                fv_result=failure_vector_result,
                reason=reason,
                current_generator_code=current_generator_code,
                feedback_mode=self.feedback_mode,
                attribution_result=attribution_result,
                coverage_summary=coverage_summary,
                attribution_history=attribution_history,
                graph_pattern_result=graph_pattern_result,
                skill_signal_result=skill_signal_result,
                include_behavior=bool(self.skill_signal_include_behavior),
            )
            self._log_acgs(f"[ACGS] Prompt length: {len(user_prompt)} chars")
            prompt_artifact_dir = self._save_evolve_prompt_artifacts(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                reason=reason,
                fv_result=failure_vector_result,
                attribution_output_dir=attribution_output_dir,
                graph_pattern_output_dir=graph_pattern_output_dir,
            )
            evolve_record['prompt_artifact_dir'] = prompt_artifact_dir
            self._append_evolve_prompt_log(
                prompt=user_prompt,
                reason=reason,
                fv_result=failure_vector_result,
            )
            if attribution_result_obj is not None:
                append_attribution_history(
                    self._attribution_history,
                    attribution_result_obj,
                    max_len=3,
                )

            if getattr(self, 'feedback_mode', 'coarse') == 'skill_library':
                # Route B V0 (plan 1A): best-of-R argmax over {G_t}∪R candidates,
                # bypassing the per-candidate first-accept loop.
                new_code = self._evolve_skill_library_best_of_r(
                    failure_vector_result=failure_vector_result,
                    reason=reason,
                    current_generator_code=current_generator_code,
                    skill_signal_result=skill_signal_result,
                    evolve_record=evolve_record,
                )
            else:
                verifier_callback = self._build_candidate_verifier_callback(
                    current_generator_code=current_generator_code,
                    attribution_result=attribution_result,
                    attribution_output_dir=attribution_output_dir,
                    evolve_record=evolve_record,
                )

                new_code = self.acgs_api.evolve(
                    batch_stats=dict(self._batch_stats),
                    fv_result=failure_vector_result,
                    reason=reason,
                    current_generator_code=current_generator_code,
                    num_obstacles=self.obstacle_num,
                    feedback_mode=self.feedback_mode,
                    attribution_result=attribution_result,
                    coverage_summary=coverage_summary,
                    attribution_history=attribution_history,
                    graph_pattern_result=graph_pattern_result,
                    skill_signal_result=skill_signal_result,
                    include_behavior=bool(self.skill_signal_include_behavior),
                    verifier_callback=verifier_callback,
                )

            evolve_record['llm_generation_success'] = bool(new_code)

            if new_code:
                phase4_monitor_record = self._compute_phase4_monitor_after_accept(
                    old_code=current_generator_code,
                    new_code=new_code,
                    context_signal=skill_signal_result,
                    evolve_record=evolve_record,
                )
                if phase4_monitor_record is not None:
                    evolve_record['phase4_monitor_written'] = True
                evolve_record['load_success'] = True
                self.topology_generator_code = self.acgs_api.export_generator_code()
                self._evolve_count += 1
                self._just_evolved = True
                self._last_evolve_episode = self._total_episode_count
                self._log_acgs(
                    f"[ACGS] Evolve #{self._evolve_count}/{self.max_evolve_count} "
                    f"succeeded, new generator loaded"
                )
                self._success_rate_history.clear()
                evolve_record['accepted'] = True

                if self.save_generator_artifacts and self.topology_generator_code:
                    self.generator_artifacts.save_current(self.topology_generator_code)
                    self.generator_artifacts.save_evolved(self.topology_generator_code, self._evolve_count)
                    self.generator_artifacts.save_generator_version(
                        self.topology_generator_code,
                        generator_id=self._evolve_count,
                    )

            else:
                evolve_record['rejected_reason'] = 'all_retries_failed'
                self._log_acgs("[ACGS] Evolve failed after all retries")

        except Exception as e:
            evolve_record['error'] = str(e)
            evolve_record['rejected_reason'] = 'exception'
            self._log_acgs(f"[ACGS] Evolve error: {e}")
            traceback.print_exc()

        self._acgs_evolve_history.append(evolve_record)
        try:
            self._append_evolve_record_log(evolve_record)
        except Exception as exc:
            LOGGER.warning("[ACGS] Failed to write evolve record log: %s", exc)
        self._clear_pending_evolve_state()
        self._reset_batch()

    def _append_evolve_record_log(self, evolve_record):
        path = pathlib.Path(self.output_dir) / "acgs_evolve_records.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(evolve_record, ensure_ascii=True, default=str) + "\n")

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

    def _trigger_acgs_evolve(self):
        """At batch end, trigger LLM evolve on a fixed episode schedule."""
        total = sum(self._batch_stats.values())
        success_rate = self._batch_stats['success'] / max(total, 1)
        collision_rate = self._batch_stats['collision'] / max(total, 1)

        self._log_acgs(f"\n[ACGS] Batch complete ({self._batch_episode_count} episodes, "
                   f"total {self._total_episode_count})")
        self._log_acgs(f"[ACGS] Stats: {self._batch_stats}")
        self._log_acgs(f"[ACGS] Success rate: {success_rate:.3f}, Collision rate: {collision_rate:.3f}")
        self._log_acgs(
            f"[ACGS] Obstacle rollout records: {len(self._batch_obstacle_rollout_records)}"
        )

        failure_vector_result = self._compute_failure_vector()

        # 记录成功率历史
        self._success_rate_history.append(success_rate)

        is_first_trigger = not self._first_evolve_triggered
        if self.max_evolve_count <= 0:
            reason = "fixed_schedule_disabled(max_evolve_count=0)"
            self._log_acgs("[ACGS] Skipping: max_evolve_count=0")
            self._append_failure_vector_history(failure_vector_result, should_evolve=False, reason=reason)
            self._reset_batch()
            return

        if self._evolve_count >= self.max_evolve_count:
            reason = (
                f"fixed_schedule_complete(evolve_count={self._evolve_count}/"
                f"{self.max_evolve_count})"
            )
            self._log_acgs(f"[ACGS] Skipping: {reason}")
            self._append_failure_vector_history(failure_vector_result, should_evolve=False, reason=reason)
            self._reset_batch()
            return

        target_episode = (
            self.first_evolve_episode
            if is_first_trigger
            else self.first_evolve_episode + self._evolve_count * self.evaluation_interval
        )
        if self._total_episode_count < target_episode:
            reason = (
                f"fixed_wait(eps={self._total_episode_count}, "
                f"target={target_episode}, evolve_count={self._evolve_count}/"
                f"{self.max_evolve_count})"
            )
            self._log_acgs(f"[ACGS] Skipping: {reason}")
            self._append_failure_vector_history(failure_vector_result, should_evolve=False, reason=reason)
            self._reset_batch()
            return

        schedule_reason = (
            f"fixed_schedule(evolve_index={self._evolve_count + 1}/"
            f"{self.max_evolve_count}, eps={self._total_episode_count}, "
            f"target={target_episode})"
        )
        reason = self._difficulty_reason(success_rate, schedule_reason)
        self._append_failure_vector_history(failure_vector_result, should_evolve=True, reason=reason)

        if not self._first_evolve_triggered:
            self._first_evolve_triggered = True
            self._warmup_completed = True

        if is_first_trigger and self.save_pre_evolve_checkpoint:
            self._save_pre_evolve_branch_checkpoint(reason)

        # ---- 触发 evolve ----
        self._log_acgs(f"[ACGS] >>> Triggering evolve: {reason}")
        self._execute_acgs_evolve(reason, failure_vector_result)

    def _reset_batch(self):
        """重置批次数据"""
        self._batch_stats = {'success': 0, 'collision': 0, 'timeout': 0, 'fall': 0}
        self._batch_failure_counts = {k: 0 for k in _FAILURE_KEYS}
        self._batch_episode_count = 0
        self._batch_obstacle_rollout_records = []

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

        resumed = self._resume_from_checkpoint_if_requested()
        if resumed:
            episode_reward_logger = list(self.rl_loop_state.recent_episode_rewards)
            num_episode = int(self.rl_loop_state.num_episode)
            max_test_score = float(self.rl_loop_state.max_test_score)
            val_reward = float(self.rl_loop_state.last_validate_reward)
            self.updates = int(self.rl_loop_state.updates)
            self.critic_update = int(self.rl_loop_state.critic_update)
            self.global_step = int(self.rl_loop_state.global_step)
            self.epoch = int(self.rl_loop_state.epoch)
            self.num_timesteps = int(self.rl_loop_state.num_timesteps)
            self.episode_count = int(self.rl_loop_state.episode_count)
            self.epsilon = float(self.rl_loop_state.epsilon)

        # 初始化拓扑生成器（只调用一次 LLM）
        if not resumed:
            self._init_topology_generator()
        elif self.use_llm_obstacles and not self.topology_generator_code:
            # Resume 但 checkpoint 未携带 generator 代码（例如从未触发过 evolve 的
            # Phase-0 checkpoint）。此时若不补建 generator，整个 batch 会退化为 env
            # 内置 between 障碍，且首次 evolve 的 current_generator_code 为 None，
            # 导致 skill_signal / verifier / phase4_monitor 静默退化为 coarse。
            LOGGER.warning(
                "[Resume] Checkpoint carried no generator code; "
                "bootstrapping an initial topology generator so the "
                "skill-signal evolve pipeline has a valid current generator."
            )
            self._init_topology_generator()

        pending_reason = self.curriculum_state.pending_evolve_reason
        if resumed and pending_reason:
            if self.enable_acgs_loop:
                self._log_acgs(f"[Resume] Executing pending evolve from checkpoint: {pending_reason}")
                failure_vector_result = self._compute_failure_vector()
                self._execute_acgs_evolve(pending_reason, failure_vector_result)
            else:
                self._log_acgs("[Resume] Pending evolve found, but ACGS loop is disabled; continuing with static branch")
                self._clear_pending_evolve_state()
                self._reset_batch()
        
        # 第一次 reset：生成 T-block 位置和障碍物
        obs, tblock_pose = self._generate_obstacles_for_reset()
        if obs is None:
            # 不使用 LLM，正常 reset
            obs = self.env.reset()
            tblock_pose = None

        # ACGS: 记录第一个 episode 的起点
        self._on_episode_start(obs, tblock_pose)
        # Task 6: draw the skill executed by this first episode.
        self._current_skill_id = self._sample_episode_skill_id()

        self.model.reset()
        epsilon = cfg.rl.noise_epsilon
        self.depsilon = 1.0 / epsilon
        if not resumed:
            self.epsilon = 1.0
        
        with tqdm.tqdm(total=cfg.training.num_epochs, ncols=50, desc=f"Train epochs") as pbar:
            if self.updates > 0:
                pbar.update(self.updates)
            with tqdm.tqdm(total=cfg.rl.warmup, ncols=50, desc=f"Warm Up") as pbar2:
                if self.num_timesteps > 0:
                    pbar2.update(min(self.num_timesteps, cfg.rl.warmup))
                if self.num_timesteps > cfg.rl.warmup:
                    pbar2.close()
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
                            action = self.model.predict_action(obs_th, skill_id=self._current_skill_id)
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

                    # Replay buffer (Task 6: tag skill_id / source / reward_task
                    # for the executed skill when the skill library is enabled).
                    if self.skill_enabled:
                        self.replay_buffer.add(
                            obs, next_obs, action, reward, done,
                            skill_id=self._current_skill_id,
                            source='normal',
                            reward_task=reward,
                        )
                    else:
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
                        # Task 10: track reward_task success for beta_div schedule.
                        if self.skill_enabled:
                            self._recent_success.append(1.0 if info.get('success') else 0.0)

                        self.env.seed()
                        self.model.reset()

                        # 每次 reset 前：生成新的 T-block 位置和障碍物配置
                        self.episode_count += 1
                        self.rl_loop_state.episode_count = int(self.episode_count)
                        obs, tblock_pose = self._generate_obstacles_for_reset()
                        if obs is None:
                            # 不使用 LLM，正常 reset
                            obs = self.env.reset()
                            tblock_pose = None

                        # ACGS: 记录新 episode 的起点
                        self._on_episode_start(obs, tblock_pose)
                        # Task 6: draw the skill for the new episode.
                        self._current_skill_id = self._sample_episode_skill_id()

                        episode_length, episode_reward = 0, 0
                        num_episode += 1
                        self.rl_loop_state.num_episode = int(num_episode)
                    
                    # update model
                    if self.num_timesteps > cfg.rl.warmup:
                        if self.num_timesteps == cfg.rl.warmup+1:
                            pbar2.close()
                        self.model.train()
                            
                        training_info = self.update(batch_size=cfg.rl.batch_size)

                        # Task 9: periodic paired diversity batch (default OFF;
                        # only after RL warmup + task warm-up, when beta_div opens).
                        if (self.skill_enabled and self.diversity_every > 0
                                and self.num_timesteps % self.diversity_every == 0
                                and (self.num_timesteps - cfg.rl.warmup) >= self.task_warmup_steps):
                            div_metrics = self._run_diversity_batch()
                            if div_metrics:
                                div_log = {f'[diversity]{k}': v for k, v in div_metrics.items()}
                                training_info.update(div_log)
                                # Bug fix: log diversity metrics immediately (they fire on
                                # a num_timesteps period offset from the log_interval on
                                # self.updates, so they never coincided and were dropped).
                                if cfg.log:
                                    wandb_run.log(div_log, step=self.updates)

                        pbar.update(1)
                        pbar.set_postfix(episode_reward=np.mean(episode_reward_logger[-100:]))
                        
                        # validate
                        policy = self.model
                        policy.eval()

                        # Task 10: adjust beta_div on a fixed eval period. Bug fix: this
                        # must NOT be gated by cfg.training.validate (the schedule is
                        # independent of running the full validation rollout).
                        if self.skill_enabled and self.updates % cfg.training.validate_steps == 0:
                            self._update_beta_div()
                            if cfg.log:
                                wandb_run.log(
                                    {'[skill]beta_div': float(self.beta_div),
                                     '[skill]k_eff': float(self._last_k_eff)},
                                    step=self.updates,
                                )

                        if cfg.training.validate and self.updates % cfg.training.validate_steps == 0:
                            # Model selection on the FIXED validation distribution
                            # (DIVO 'between'), NOT the moving curriculum G_t. We
                            # reuse llm_eval_env (pusht_mujoco_llm, between, num=2,
                            # size=0.01) with a plain reset (reset_fn=None): it is
                            # never set_obstacle_config'd, so env.reset() falls
                            # through to the between program-generated obstacles.
                            # This keeps test_mean_score comparable across evolves
                            # so best-ckpt selection is not frozen by a hardening
                            # seen distribution.
                            #
                            # RNG ISOLATION (做法 A): validation rollouts consume the
                            # GLOBAL numpy/torch RNG (env.reset samples poses, etc.).
                            # Without isolation, validation "steals" a chunk of the
                            # shared random stream and SHIFTS the TRAINING stream
                            # afterwards, so any change to validation (env / #eps /
                            # reset_fn) silently perturbs training and breaks
                            # reproducibility (this is the suspected post-00.34.42
                            # regression). Snapshot the RNG state before and restore
                            # it after so validation is a pure read-only observer and
                            # training's stream is unaffected. try/finally guarantees
                            # restoration even if the evaluator raises.
                            _np_rng_state = np.random.get_state()
                            _torch_rng_state = torch.get_rng_state()
                            _cuda_rng_states = (
                                torch.cuda.get_rng_state_all()
                                if torch.cuda.is_available() else None
                            )
                            try:
                                val_reward, val_log = self.evaluator(
                                    self.llm_eval_env,
                                    policy,
                                    self.no_obs_env,
                                    self.unseen_env,
                                    episode_reset_fn=None,
                                )
                            finally:
                                np.random.set_state(_np_rng_state)
                                torch.set_rng_state(_torch_rng_state)
                                if _cuda_rng_states is not None:
                                    torch.cuda.set_rng_state_all(_cuda_rng_states)
                            wandb_run.log({
                                    'validate_reward': val_reward,
                                },step = self.updates)
                            step_log.update(val_log)
                            wandb_run.log(step_log, step=self.updates)

                        # checkpoint
                        if (self.updates % cfg.training.checkpoint_every) == 0:
                            self._sync_rl_loop_state(
                                max_test_score=max_test_score,
                                val_reward=val_reward,
                                episode_reward_logger=episode_reward_logger,
                                num_episode=num_episode,
                            )
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
                            self._sync_rl_loop_state(
                                max_test_score=max_test_score,
                                val_reward=val_reward,
                                episode_reward_logger=episode_reward_logger,
                                num_episode=num_episode,
                            )
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
                                
                        self._sync_rl_loop_state(
                            max_test_score=max_test_score,
                            val_reward=val_reward,
                            episode_reward_logger=episode_reward_logger,
                            num_episode=num_episode,
                        )
                        self.updates += 1
                        
    def update(self, batch_size: int):
        cfg = copy.deepcopy(self.cfg)
        if self.skill_enabled:
            # Source-stratified batch: P(w_0) >= w0_rollout_ratio, paired <= cap
            # (Task 4.3). Route the critic reward by skill (Task 7.3).
            experience_replay = self.replay_buffer.sample_stratified(
                batch_size=batch_size,
                w0_min_ratio=self.w0_rollout_ratio,
                paired_sample_ratio=self.paired_sample_ratio,
            )
        else:
            experience_replay = self.replay_buffer.sample(batch_size=batch_size)

        # Compute target q values
        with torch.no_grad():
            next_q_value = self.compute_next_q_value(experience_replay)
            if self.skill_enabled:
                reward_np = reward_for_critic(
                    experience_replay.skill_id,
                    experience_replay.reward_task,
                    experience_replay.reward_total,
                ).astype(np.float32)
                rewards = torch.from_numpy(reward_np).to(self.device)
            else:
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

        if self.skill_enabled:
            # w-conditioned target: next action and Q use the transition's skill.
            w = self._skill_w_from_batch(experience_replay)
            next_action = self.model_target.predict_action_with_skill(next_obs_th, w)
            next_q_values = self.critic_target(next_obs_th, next_action, w)
        else:
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
        w = self._skill_w_from_batch(experience_replay)
        current_q_values = self.critic(obs_th, action, w)
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
        reg_z = ('regularize_z' in self.cfg.training)
        z = None
        z_mean = z_var = None
        if self.skill_enabled:
            # L_exec (executed-w only): each transition uses its own skill w_exec
            # and a w-conditioned critic. w_0 -> pure task value (reward_div=0),
            # probe -> reward_total. z = encoder(obs) stays the obstacle channel;
            # diversity is NOT an independent differentiable actor term (it enters
            # via reward_total in the critic, see Task 9).
            z = self.model.encoder(obs_th)
            state = self.model.obs2state(obs_th)
            w_exec = self._skill_w_from_batch(experience_replay)
            action = self.model.decode_with_skill(state, z, w_exec)
            current_q_values = self.critic(obs_th, action, w_exec)
            if reg_z:
                z_mean = z.mean(0)
                z_var = z.var(0)
        else:
            if reg_z:
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

        # Optional all-w coverage (Task 7.4): default lambda_cov_all=0 (off);
        # a critic-stable warm-up ablation only, NOT the Stage 1 main path.
        if self.skill_enabled and self.lambda_cov_all > 0.0:
            state = self.model.obs2state(obs_th)
            cov_terms = []
            for k in range(self.model.num_skills):
                wk = self.model.skill_code(k).unsqueeze(0).expand(obs_th.shape[0], -1)
                a_k = self.model.decode_with_skill(state, z, wk)
                q_k = torch.cat(self.critic(obs_th, a_k, wk), dim=1)
                q_k, _ = torch.min(q_k, dim=1, keepdim=True)
                cov_terms.append(-q_k.mean())
            policy_loss = policy_loss + self.lambda_cov_all * torch.stack(cov_terms).mean()

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
            if z is not None:
                policy_loss_info['[policy]z_norm'] = (torch.norm(z, dim=1)**2).mean().item()
            if z_mean is not None:
                policy_loss_info['[policy]z_mean'] = (z_mean).mean().item()
                policy_loss_info['[policy]z_var'] = (z_var).mean().item()

        return policy_loss.to(self.device), policy_loss_info
