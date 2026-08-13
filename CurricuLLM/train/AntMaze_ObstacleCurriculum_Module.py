import gc
import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import wandb
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv

from DIVO.env.antmaze import format_maze_map
from DIVO.gpt.antmaze_acgs_api import AntMazeACGS_API
from DIVO.utils.util import save_anim
from evaluation.evalcallback_success import SuccessEvalCallback
from utils.antmaze_failure_evidence import collect_antmaze_rollout_evidence
from utils.train_utils import make_env


TRAINING_ALGORITHMS = {
    "PPO": PPO,
    "SAC": SAC,
}


DEFAULT_INITIAL_GENERATOR = '''\
def generate_maze_map(seed: int = None) -> list:
    import numpy as np
    rng = np.random.default_rng(seed)
    layouts = [
        [
            [1, 1, 1, 1, 1],
            [1, "r", 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 0, "g", 1],
            [1, 1, 1, 1, 1],
        ],
        [
            [1, 1, 1, 1, 1],
            [1, "r", 0, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 0, "g", 1],
            [1, 1, 1, 1, 1],
        ],
    ]
    return layouts[int(rng.integers(0, len(layouts)))]
'''


def _cfg_get(cfg: Dict[str, Any], key: str, default=None):
    return cfg[key] if key in cfg else default


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class AntMaze_ObstacleCurriculum_Module:
    """
    AntMaze instantiation of the obstacle-distribution curriculum.

    This first version implements the stage-level loop:
    generator -> maze_map -> SAC/PPO training stage -> rollout evidence -> optional evolve.
    """

    def __init__(self, env_name, env_path, logger_path, cfg, seed=0):
        self.env_name = env_name
        self.env_path = env_path
        self.logger_path = logger_path
        self.cfg = cfg["AntMazeObstacleCfg"]
        self.seed = int(seed)
        self.env_id = self.cfg["env_id"]
        self.eval_env_id = self.cfg.get("eval_env_id", self.env_id)
        self.num_envs = int(self.cfg.get("num_envs", 4))
        self.eval_num_envs = int(self.cfg.get("eval_num_envs", self.num_envs))
        self.training_algorithm = TRAINING_ALGORITHMS[self.cfg.get("training_alg", "SAC")]
        self.policy_network = self.cfg.get("policy_network", "MultiInputPolicy")
        self.num_stages = int(self.cfg.get("num_stages", 3))
        self.stage_timesteps = int(self.cfg.get("stage_timesteps", 100000))
        self.eval_freq = int(self.cfg.get("eval_freq", 5000))
        self.evidence_episodes = int(self.cfg.get("evidence_episodes", 5))
        self.tensorboard_enabled = bool(self.cfg.get("tensorboard_log", False))
        self.wandb_enabled = bool(self.cfg.get("wandb_log", True))
        self.num_save_video = int(self.cfg.get("num_save_video", 3))
        self.video_fps = int(self.cfg.get("video_fps", 30))
        self.evolve_enabled = bool(self.cfg.get("evolve_enabled", False))
        self.evolve_after_stage = int(self.cfg.get("evolve_after_stage", 0))
        self.init_mode = self.cfg.get("init_mode", "manual")
        self.init_generator_path = self.cfg.get("init_generator_path")
        self.prompt_dir = self.cfg.get("prompt_dir", "DIVO/gpt/prompt")
        self.output_generators_dir = os.path.join(self.logger_path, "generators")
        self.checkpoint_dir = os.path.join(self.logger_path, "checkpoints")
        self.log_dir = os.path.join(self.logger_path, "log")
        self.media_dir = os.path.join(self.log_dir, "media")
        os.makedirs(self.output_generators_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.media_dir, exist_ok=True)
        os.makedirs(self.logger_path, exist_ok=True)

        self.acgs_api: Optional[AntMazeACGS_API] = None
        self.current_generator_code: Optional[str] = None
        self.evidence_history = []
        self.evolve_history = []
        self.wandb_run = None

    def _init_api(self):
        if self.acgs_api is not None:
            return
        self.acgs_api = AntMazeACGS_API(
            prompt_dir=self.prompt_dir,
            api_type=self.cfg.get("api_type", "deepseek"),
            api_key=self.cfg.get("api_key"),
            model=self.cfg.get("model", "deepseek-chat"),
            base_url=self.cfg.get("base_url"),
            temperature=float(self.cfg.get("temperature", 0.7)),
            max_tokens=int(self.cfg.get("max_tokens", 2000)),
            max_evolve_retries=int(self.cfg.get("max_evolve_retries", 3)),
            sanity_check_count=int(self.cfg.get("sanity_check_count", 3)),
        )

    def _save_generator(self, name: str, code: str) -> str:
        path = os.path.join(self.output_generators_dir, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(code)
            if not code.endswith("\n"):
                handle.write("\n")
        return path

    def _save_root_generator(self, name: str, code: str) -> str:
        path = os.path.join(self.logger_path, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(code)
            if not code.endswith("\n"):
                handle.write("\n")
        return path

    def _init_wandb(self):
        if not self.wandb_enabled or self.wandb_run is not None:
            return
        logging_cfg = dict(self.cfg.get("wandb", {}))
        logging_cfg.setdefault("project", "ACGS-Comparison")
        logging_cfg.setdefault("resume", True)
        logging_cfg.setdefault("mode", "online")
        logging_cfg.setdefault("name", f"antmaze_obstacle_seed{self.seed}")
        logging_cfg.setdefault("tags", ["AntMaze_UMaze", "antmaze_obstacle", "obstacle_curriculum"])
        logging_cfg.setdefault("group", "antmaze_obstacle")
        self.wandb_run = wandb.init(
            dir=str(self.logger_path),
            config={
                "env_name": self.env_name,
                "env_id": self.env_id,
                "seed": self.seed,
                "AntMazeObstacleCfg": dict(self.cfg),
            },
            **logging_cfg,
        )

    def _load_initial_generator(self):
        self._init_api()
        assert self.acgs_api is not None

        if self.init_mode == "path":
            if not self.init_generator_path:
                raise ValueError("init_mode=path requires init_generator_path")
            with open(self.init_generator_path, "r", encoding="utf-8") as handle:
                code = handle.read()
            if not self.acgs_api.load_generator_code(code):
                raise RuntimeError(f"Failed to load initial generator: {self.init_generator_path}")
        elif self.init_mode == "llm":
            code = self.acgs_api.init_generator()
            if code is None:
                raise RuntimeError("LLM initial AntMaze generator failed")
        else:
            code = DEFAULT_INITIAL_GENERATOR
            if not self.acgs_api.load_generator_code(code):
                raise RuntimeError("Default initial AntMaze generator failed to load")

        self.current_generator_code = self.acgs_api.export_generator_code()
        self._save_root_generator("initial_generator.py", self.current_generator_code)
        self._save_root_generator("current_generator.py", self.current_generator_code)
        self._save_generator("generator_000_initial.py", self.current_generator_code)

    def _sample_stage_env_kwargs(self, stage_idx: int) -> Dict[str, Any]:
        assert self.acgs_api is not None
        maze_map = self.acgs_api.generate_maze_map(seed=self.seed + stage_idx)
        map_path = os.path.join(self.logger_path, f"stage_{stage_idx:03d}_maze.txt")
        with open(map_path, "w", encoding="utf-8") as handle:
            handle.write(format_maze_map(maze_map))
            handle.write("\n")
        return {"maze_map": maze_map}

    def _make_single_env(self, env_id: str, env_kwargs: Dict[str, Any], render_mode: Optional[str] = None):
        return make_env(
            env_id,
            0,
            seed=self.seed,
            render_mode=render_mode,
            env_kwargs=env_kwargs,
        )()

    def _make_vec_env(self, env_id: str, num_envs: int, env_kwargs: Dict[str, Any]):
        return SubprocVecEnv(
            [
                make_env(
                    env_id,
                    i,
                    seed=self.seed + i,
                    env_kwargs=env_kwargs,
                )
                for i in range(num_envs)
            ]
        )

    def _format_evidence_for_prompt(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        counts = evidence.get("failure_counts", evidence.get("batch_stats", {}))
        distribution = evidence.get("failure_distribution", {})
        dominant = evidence.get("dominant_failure_type", "unknown")
        return {
            "counts": counts,
            "distribution": distribution,
            "dominant_failure_type": dominant,
            "dominant_type": dominant,
        }

    def _save_eval_videos(
        self,
        model,
        env_kwargs: Dict[str, Any],
        stage_idx: int,
        n_videos: int,
    ) -> List[str]:
        if n_videos <= 0:
            return []

        video_paths: List[str] = []
        for video_idx in range(n_videos):
            env = self._make_single_env(self.eval_env_id, env_kwargs, render_mode="rgb_array")
            frames = []
            try:
                obs, _ = env.reset(seed=self.seed + stage_idx * 1000 + video_idx)
                done = False
                step_count = 0
                max_steps = int(getattr(getattr(env, "spec", None), "max_episode_steps", 700) or 700)
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                while not done and step_count < max_steps:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = env.step(action)
                    done = bool(terminated or truncated)
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                    step_count += 1
            finally:
                env.close()

            if not frames:
                continue
            filename = os.path.join(self.media_dir, f"stage_{stage_idx:03d}_eval_{video_idx}.mp4")
            save_anim(frames, filename[:-4], fps=self.video_fps, framerate=self.video_fps)
            video_paths.append(filename)

        return video_paths

    def _build_wandb_stage_log(
        self,
        stage_idx: int,
        evidence: Dict[str, Any],
        elapsed: float,
        video_paths: List[str],
    ) -> Dict[str, Any]:
        success_rate = float(evidence.get("success_rate", 0.0))
        mean_reward = float(evidence.get("mean_episode_reward", 0.0))
        mean_final_distance = evidence.get("mean_final_distance")
        mean_progress_ratio = evidence.get("mean_progress_ratio")
        batch_stats = evidence.get("batch_stats", {})
        log_data = {
            "validate_reward": mean_reward,
            "eval/success_rate[seen]": success_rate,
            "eval/mean_reward": mean_reward,
            "eval/mean_final_distance": float(mean_final_distance) if mean_final_distance is not None else np.nan,
            "eval/mean_progress_ratio": float(mean_progress_ratio) if mean_progress_ratio is not None else np.nan,
            "episode_mean_reward": mean_reward,
            "num_timesteps": int((stage_idx + 1) * self.stage_timesteps),
            "num_episode": int(sum(int(v) for v in batch_stats.values())),
            "step_ps": float(((stage_idx + 1) * self.stage_timesteps) / max(elapsed, 1e-6)),
            "acgs/batch_episode_count": int(sum(int(v) for v in batch_stats.values())),
            "acgs/evolve_count": int(len(self.evolve_history)),
            "acgs/batch_success_rate": success_rate,
            "antmaze/stage": int(stage_idx),
            "antmaze/dominant_failure_type": evidence.get("dominant_failure_type", "unknown"),
        }

        for key, value in batch_stats.items():
            log_data[f"antmaze/{key}_count"] = int(value)

        for idx, video_path in enumerate(video_paths):
            log_data[f"eval/sim_video_{idx}"] = wandb.Video(video_path)

        return log_data

    def _maybe_evolve(self, stage_idx: int, evidence: Dict[str, Any]):
        if not self.evolve_enabled or stage_idx < self.evolve_after_stage:
            return
        assert self.acgs_api is not None
        fv_result = self._format_evidence_for_prompt(evidence)
        diagnosis = evidence.get("diagnosis", {})
        success_rate = float(evidence.get("success_rate", 0.0))
        if success_rate < float(self.cfg.get("success_rate_low", 0.2)):
            reason = f"too_hard(sr={success_rate:.3f})"
        elif success_rate > float(self.cfg.get("success_rate_high", 0.8)):
            reason = f"too_easy(sr={success_rate:.3f})"
        else:
            reason = f"stage_feedback(sr={success_rate:.3f})"

        _, prompt = self.acgs_api.get_prompt_text(
            batch_stats=evidence.get("batch_stats", {}),
            fv_result=fv_result,
            diagnosis=diagnosis,
            reason=reason,
            current_generator_code=self.current_generator_code,
            history_records=self.evolve_history[-3:],
        )
        with open(os.path.join(self.logger_path, f"stage_{stage_idx:03d}_evolve_prompt.txt"), "w", encoding="utf-8") as handle:
            handle.write(prompt)

        new_code = self.acgs_api.evolve(
            batch_stats=evidence.get("batch_stats", {}),
            fv_result=fv_result,
            diagnosis=diagnosis,
            reason=reason,
            current_generator_code=self.current_generator_code,
            history_records=self.evolve_history[-3:],
        )
        record = {
            "stage": stage_idx,
            "reason": reason,
            "success_rate": success_rate,
            "accepted": bool(new_code),
        }
        if new_code:
            self.current_generator_code = new_code
            self._save_root_generator("current_generator.py", new_code)
            self._save_generator(f"generator_{stage_idx + 1:03d}_evolved.py", new_code)
        self.evolve_history.append(record)

    def train(self):
        self._init_wandb()
        start_time = time.time()
        self._load_initial_generator()
        model = None

        for stage_idx in range(self.num_stages):
            env_kwargs = self._sample_stage_env_kwargs(stage_idx)
            training_env = self._make_vec_env(self.env_id, self.num_envs, env_kwargs)
            eval_env = self._make_vec_env(self.eval_env_id, self.eval_num_envs, env_kwargs)

            eval_callback = SuccessEvalCallback(
                eval_env,
                log_path=os.path.join(self.logger_path, f"stage_{stage_idx:03d}"),
                best_model_save_path=os.path.join(self.logger_path, f"stage_{stage_idx:03d}"),
                eval_freq=self.eval_freq,
                deterministic=True,
                render=False,
                warn=False,
            )

            if model is None:
                model = self.training_algorithm(
                    self.policy_network,
                    training_env,
                    verbose=1,
                    tensorboard_log=self.logger_path if self.tensorboard_enabled else None,
                )
            else:
                model.set_env(training_env)

            model.learn(
                total_timesteps=self.stage_timesteps,
                callback=eval_callback,
                reset_num_timesteps=(stage_idx == 0),
                tb_log_name=f"antmaze_obstacle_stage_{stage_idx:03d}",
            )
            stage_model_path = os.path.join(self.logger_path, f"stage_{stage_idx:03d}", "stage_model")
            model.save(stage_model_path)
            model.save(os.path.join(self.checkpoint_dir, f"stage_{stage_idx:03d}_model"))

            evidence = collect_antmaze_rollout_evidence(
                model,
                eval_env,
                n_episodes=self.evidence_episodes,
                deterministic=True,
            )
            evidence["stage"] = stage_idx
            self.evidence_history.append(evidence)
            with open(os.path.join(self.logger_path, f"stage_{stage_idx:03d}_evidence.json"), "w", encoding="utf-8") as handle:
                json.dump(evidence, handle, indent=2, default=_json_default)

            video_paths = self._save_eval_videos(
                model,
                env_kwargs=env_kwargs,
                stage_idx=stage_idx,
                n_videos=self.num_save_video,
            )
            if self.wandb_run is not None:
                self.wandb_run.log(
                    self._build_wandb_stage_log(stage_idx, evidence, time.time() - start_time, video_paths),
                    step=(stage_idx + 1) * self.stage_timesteps,
                )

            self._maybe_evolve(stage_idx, evidence)

            training_env.close()
            eval_env.close()
            gc.collect()
            torch.cuda.empty_cache()

        if model is not None:
            final_path = os.path.join(self.logger_path, "final_model")
            model.save(final_path)
            model.save(os.path.join(self.checkpoint_dir, "model_latest"))
            del model

        with open(os.path.join(self.logger_path, "evolve_history.json"), "w", encoding="utf-8") as handle:
            json.dump(self.evolve_history, handle, indent=2, default=_json_default)
        with open(os.path.join(self.logger_path, "evidence_history.json"), "w", encoding="utf-8") as handle:
            json.dump(self.evidence_history, handle, indent=2, default=_json_default)

        gc.collect()
        torch.cuda.empty_cache()
        if self.wandb_run is not None:
            self.wandb_run.finish()
