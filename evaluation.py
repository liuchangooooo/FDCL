import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import copy
import json
import os
import pathlib
import random
from math import floor

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb.sdk.data_types.video as wv
from omegaconf import OmegaConf

from DIVO.env import get_env_class
from DIVO.evaluator import get_evaluator
from DIVO.motion_decoder import get_motion_decoder
from DIVO.policy import get_policy
from DIVO.sampler import get_sampler
from DIVO.utils.util import *


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("config")),
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    evaluator = cls(cfg)
    evaluator.eval()


if __name__ == "__main__":
    main()


class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = str(getattr(cfg, "mode", "full_adaptation")).lower()
        self.device = torch.device(cfg.device)
        self.output_dir = pathlib.Path(cfg.output_dir).expanduser().resolve()

        self._set_seed(cfg.seed)
        self._prepare_output_dir(self.output_dir)

        self.env = get_env_class(**self._to_container(cfg.env))
        self.action_dim, self.obs_dim = self.env.get_info()
        print("\n [1] Env is set:")

        policy_checkpoint = cfg.policy.checkpoint
        self.p_cfg = OmegaConf.load(
            f"{policy_checkpoint.rsplit('/', 2)[0]}/.hydra/config.yaml"
        )
        self.policy = get_policy(self.env, **self.p_cfg.policy).to(self.device)
        self.policy.load_state_dict(
            torch.load(policy_checkpoint, map_location=self.device)
        )
        self.policy.eval()

        print("\n [2] Policy is set:")
        print(self.policy)

        self.sampler = None
        self.motion_decoder = None

        if self.mode == "full_adaptation":
            self._load_full_adaptation_modules()
        else:
            print("\n [3] Policy-only final evaluation mode is enabled.")

    def eval(self):
        if self.mode == "policy_only":
            return self._eval_policy_only()
        return self._eval_full_adaptation()

    def _load_full_adaptation_modules(self):
        sampler_checkpoint = self.cfg.sampler.checkpoint
        if sampler_checkpoint in (None, "None", "null"):
            raise ValueError("Sampler checkpoint is not provided")
        self.s_cfg = OmegaConf.load(
            f"{sampler_checkpoint.rsplit('/', 1)[0]}/.hydra/config.yaml"
        )
        self.sampler = get_sampler(**self.s_cfg.sampler).to(self.device)
        self.sampler.load_state_dict(
            torch.load(sampler_checkpoint, map_location=self.device)
        )
        self.sampler.eval()

        print("\n [3] Sampler is set:")
        print(self.sampler)

        motion_decoder_checkpoint = self.cfg.motion_decoder.checkpoint
        if motion_decoder_checkpoint in (None, "None", "null"):
            raise ValueError("Motion decoder checkpoint is not provided")
        self.m_cfg = OmegaConf.load(
            f"{motion_decoder_checkpoint.rsplit('/', 1)[0]}/.hydra/config.yaml"
        )
        self.motion_decoder = get_motion_decoder(**self.m_cfg.motion_decoder).to(
            self.device
        )
        self.motion_decoder.load_state_dict(
            torch.load(motion_decoder_checkpoint, map_location=self.device)
        )
        self.motion_decoder.eval()

        print("\n [4] Motion decoder is set:")
        print(self.motion_decoder)

    def _eval_policy_only(self):
        benchmarks_cfg = getattr(self.cfg, "benchmarks", None)
        if benchmarks_cfg is not None and bool(benchmarks_cfg.get("enabled", False)):
            return self._eval_policy_only_suite()

        result = self._run_policy_only_benchmark(
            benchmark_name="default",
            env_cfg=self.cfg.env,
            output_dir=self.output_dir,
            num_episodes=int(self.cfg.num_episodes),
            num_render=int(self.cfg.num_render),
            max_steps=int(self.cfg.max_steps),
            reward=str(getattr(self.cfg, "reward", "last")),
        )
        summary_path = self.output_dir / "benchmark_summary.json"
        summary_path.write_text(
            json.dumps({"benchmarks": {"default": result}}, indent=2),
            encoding="utf-8",
        )
        return result

    def _eval_policy_only_suite(self):
        results = {}
        success_rates = []
        mean_rewards = []

        for benchmark_name, benchmark_cfg in self._iter_benchmarks():
            env_cfg = self._merge_env_cfg(benchmark_cfg)
            output_dir = self.output_dir / benchmark_name

            result = self._run_policy_only_benchmark(
                benchmark_name=benchmark_name,
                env_cfg=env_cfg,
                output_dir=output_dir,
                num_episodes=int(benchmark_cfg.get("num_episodes", self.cfg.num_episodes)),
                num_render=int(benchmark_cfg.get("num_render", self.cfg.num_render)),
                max_steps=int(benchmark_cfg.get("max_steps", self.cfg.max_steps)),
                reward=str(benchmark_cfg.get("reward", getattr(self.cfg, "reward", "last"))),
            )

            results[benchmark_name] = result
            success_rates.append(result["success_rate"])
            mean_rewards.append(result["mean_reward"])

        aggregate = {
            "avg_success_rate": float(np.mean(success_rates)) if success_rates else 0.0,
            "avg_mean_reward": float(np.mean(mean_rewards)) if mean_rewards else 0.0,
        }

        summary = {
            "mode": self.mode,
            "policy_checkpoint": self.cfg.policy.checkpoint,
            "benchmarks": results,
            "aggregate": aggregate,
        }
        summary_path = self.output_dir / "benchmark_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print("\n===== Final Benchmark Summary =====")
        for benchmark_name in self.cfg.benchmarks.order:
            bench = results[benchmark_name]
            print(
                f"[{benchmark_name}] success_rate={bench['success_rate']:.4f}, "
                f"mean_reward={bench['mean_reward']:.4f}"
            )
        print(
            f"[AVG] success_rate={aggregate['avg_success_rate']:.4f}, "
            f"mean_reward={aggregate['avg_mean_reward']:.4f}"
        )

        return summary

    def _run_policy_only_benchmark(
        self,
        benchmark_name,
        env_cfg,
        output_dir,
        num_episodes,
        num_render,
        max_steps,
        reward,
    ):
        self._prepare_output_dir(output_dir)
        env_cfg_dict = self._to_container(env_cfg)
        env = get_env_class(**env_cfg_dict)

        rollout_evaluator = get_evaluator(
            _target_="mujoco_pusht",
            output_dir=str(output_dir),
            num_episodes=num_episodes,
            max_steps=max_steps,
            device=str(self.device),
            num_save_video=num_render,
            reward=reward,
        )

        post_step_fn = self._build_post_step_fn(env_cfg_dict)
        mean_reward, log_data = rollout_evaluator(
            env,
            self.policy,
            post_step_fn=post_step_fn,
        )

        success_rate = float(log_data.get("eval/success_rate[seen]", 0.0))
        result = {
            "success_rate": success_rate,
            "mean_reward": float(mean_reward),
            "num_episodes": int(num_episodes),
            "max_steps": int(max_steps),
            "env": env_cfg_dict,
        }

        benchmark_summary_path = pathlib.Path(output_dir) / "summary.json"
        benchmark_summary_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

        print(
            f"\n[{benchmark_name}] success_rate={success_rate:.4f}, "
            f"mean_reward={float(mean_reward):.4f}"
        )
        return result

    def _iter_benchmarks(self):
        order = list(self.cfg.benchmarks.order)
        for benchmark_name in order:
            yield benchmark_name, self.cfg.benchmarks[benchmark_name]

    def _merge_env_cfg(self, benchmark_cfg):
        base_env_cfg = OmegaConf.create(self._to_container(self.cfg.env))
        env_overrides = benchmark_cfg.get("env", benchmark_cfg.get("env_overrides", None))
        if env_overrides is None:
            return base_env_cfg
        return OmegaConf.merge(base_env_cfg, env_overrides)

    def _build_post_step_fn(self, env_cfg):
        if env_cfg.get("obstacle_dist") != "random_step":
            return None

        def _post_step_fn(env, obs, reward, done, info, episode, step_idx):
            if not hasattr(env, "_task") or not hasattr(env._task, "set_random_obs_pose"):
                return obs

            obstacle_pose = env._task.set_random_obs_pose(env.physics, obs)
            updated_obs = np.array(obs, copy=True)
            if updated_obs.ndim == 1:
                updated_obs = updated_obs.reshape(1, -1)

            desk_size = env.task._desk_size
            start = 4
            end = start + len(obstacle_pose)
            if updated_obs.shape[1] >= end:
                updated_obs[0, start:end] = obstacle_pose / desk_size

            env.prev_obs = updated_obs
            return updated_obs

        return _post_step_fn

    def _eval_full_adaptation(self):
        fps = int(1 / self.env.control_timestep())

        self.trajectory_list = []
        video_num = 0
        num_success = 0
        num_episode_fail = 0
        num_initialize_fail = 0
        obstacle_pose_ = None

        with torch.no_grad():
            for episode in range(self.cfg.num_episodes):
                self.env._record_frame = True

                obs = self.env.reset()

                done = False
                episode_length = 0
                trajectory_ = []
                frames = []

                observation_list = []

                while not done:
                    obs_th = torch.from_numpy(obs).to(self.device, torch.float32)
                    if obstacle_pose_ is not None:
                        obstacle_pose = copy.deepcopy(obstacle_pose_)
                    else:
                        obstacle_pose = obs[0, 4:] * self.env.task._desk_size
                    if self.cfg.env.obstacle:
                        num_obstacle = len(obstacle_pose) // 2
                        obstacle_size = self.env._task._obstacle_size

                    state = self.env.obs2state(obs_th)

                    mpc_done = False
                    num_pred = 0
                    num_fail = 0
                    num_mpc_fail = 0
                    state_ = copy.deepcopy(state)
                    while not mpc_done:
                        constraint_done = False
                        num_rod_fail = 0
                        num_constraint_fail = 0
                        while not constraint_done:
                            z_ = self.sampler.sample(state_, dt=0.1)

                            if num_pred == 0:
                                z = copy.deepcopy(z_)
                            action = self.policy.decoder(torch.cat([state_, z_], dim=-1))
                            splined_action = disc_cubic_spline_action(
                                self.env.task._desk_size,
                                action.reshape(1, -1, 2).cpu().detach().numpy(),
                                obs,
                                self.env.action_scale,
                                self.env.len_traj,
                                self.env.total_time_length,
                            )[:, :2]
                            for action_ in splined_action:
                                for idx in range(num_obstacle):
                                    if analytic_rod_obs_collision_check(
                                        action_,
                                        0.01,
                                        obstacle_pose[2 * idx:2 * (idx + 1)],
                                        obstacle_size * 2 + 0.02,
                                    ):
                                        rod_collision = True
                                        break
                                    rod_collision = False
                                if rod_collision:
                                    break
                            if rod_collision and (num_rod_fail < 100):
                                num_rod_fail += 1
                                continue
                            if num_rod_fail >= 100:
                                print("Can't find z to avoid the rod")
                                break

                            motion_pred = self.predict_motion(
                                obs,
                                state_,
                                z_,
                                action,
                                splined_action,
                            )

                            for motion in motion_pred:
                                angle = np.arctan2(motion[-1], motion[-2])
                                pred_tblock_pose = motion[:2] * self.env.task._desk_size
                                for idx in range(num_obstacle):
                                    if analytic_obs_collision_check(
                                        angle,
                                        obstacle_pose[2 * idx:2 * (idx + 1)] - pred_tblock_pose[:2],
                                        obstacle_size * 2,
                                        threshold=0.02,
                                    ):
                                        constraint_done = False
                                        num_constraint_fail += 1
                                        break
                                    constraint_done = True
                                if not constraint_done:
                                    break
                            if (not constraint_done) and (num_constraint_fail > 19):
                                break
                            constraint_done = True
                        if constraint_done:
                            state_ = motion_pred[-1].reshape(1, -1)
                            state_ = torch.from_numpy(state_).to(self.device, torch.float32)
                            num_pred += 1
                            num_fail = 0

                        if constraint_done and num_pred > 1:
                            mpc_done = True
                        elif (num_pred > 0) and (not constraint_done) and (num_fail < 20):
                            num_fail += 1
                            continue
                        elif (num_pred > 0) and (not constraint_done):
                            print(
                                "episode_length : ",
                                episode_length,
                                "Can't find the path in the next state",
                            )
                            state_ = copy.deepcopy(state)
                            num_pred = 0
                            num_mpc_fail += 1
                        elif (num_pred == 0) and (not constraint_done):
                            print(
                                "episode_length : ",
                                episode_length,
                                "Can't find the path in the first state",
                            )
                            num_fail += 1

                        if (num_fail > 19) or (num_mpc_fail > 19):
                            print("Can't find the path to avoid the obstacle")
                            break

                    action = self.policy.decoder(torch.cat([state, z], dim=-1))
                    motion_pred = self.predict_motion(obs, state, z, action, None)
                    action = action.cpu().detach().numpy()

                    next_obs, reward, done, info = self.env.step(action, motion_pred)

                    if self.env.obstacle_dist == "random_step":
                        obstacle_pose_ = self.env._task.set_random_obs_pose(
                            self.env.physics,
                            next_obs,
                        )

                    if episode_length == 0:
                        trajectory_.append(info["trajectory"])

                    frames += self.env.frames

                    obs = next_obs
                    episode_length += 1

                    observation_list.append(info["trajectory"])

                    if episode_length > self.cfg.max_steps:
                        done = True

                trajectory_ = np.concatenate(trajectory_, axis=0)
                observation_list = np.concatenate(observation_list, axis=0)

                observation_path = self.output_dir / "figure" / f"observation_{episode}.npy"
                trajectory_path = self.output_dir / "figure" / f"trajectory_{episode}.npy"
                with open(observation_path, "wb") as f:
                    np.save(f, observation_list)
                with open(trajectory_path, "wb") as f:
                    np.save(f, trajectory_)

                if info["success"]:
                    self.trajectory_list.append(trajectory_)
                    print(
                        f"episode : {episode}, episode_length : {episode_length}, "
                        f"reward : {reward:.3f}"
                    )

                    if (video_num < self.cfg.num_render) and self.env._record_frame:
                        save_anim(
                            frames,
                            str(self.output_dir / "media" / f"mujoco_{episode}_{video_num}"),
                            fps=fps,
                        )

                    video_num += 1
                    num_success += 1
                else:
                    if len(frames) > 0 and self.env._record_frame:
                        save_anim(
                            frames,
                            str(
                                self.output_dir
                                / "media"
                                / f"mujoco_fail_{episode}_{num_episode_fail}"
                            ),
                            fps=fps,
                        )
                    print(
                        f"episode : {episode}, {num_episode_fail}-th fail, "
                        f"episode_length : {episode_length}"
                    )
                    if episode_length == 1:
                        num_initialize_fail += 1
                    num_episode_fail += 1

            print(
                f"success_rate : {num_success / (num_success + num_episode_fail)}, "
                f"num_initialize_fail : {num_initialize_fail}"
            )

            self.calculate_2D_diversity()
            if self.sampler is not None:
                self.calculate_action_diversity()

    def calculate_2D_diversity(self, env=None):
        if len(self.trajectory_list) > 0:
            half_window_size = 1.0
            bin_size = 0.1

            num_bins = int(2 * half_window_size / bin_size)
            for i in range(num_bins + 1):
                line_pos = i * bin_size - half_window_size
                plt.axvline(x=line_pos, color="gray", linewidth=0.5)
                plt.axhline(y=line_pos, color="gray", linewidth=0.5)

            for idx, trajectory_ in enumerate(self.trajectory_list):
                if idx == 0:
                    plt.scatter(0, 0, c="r", label="goal")
                    plt.scatter(trajectory_[0, 0], trajectory_[0, 1], c="g", label="start")
                else:
                    plt.scatter(0, 0, c="r")
                    plt.scatter(trajectory_[0, 0], trajectory_[0, 1], c="g")

                plt.plot(trajectory_[:, 0], trajectory_[:, 1])

            plt.axvline(x=-half_window_size, color="gray", linewidth=5.5)
            plt.axvline(x=half_window_size, color="gray", linewidth=5.5)
            plt.axhline(y=-half_window_size, color="gray", linewidth=5.5)
            plt.axhline(y=half_window_size, color="gray", linewidth=5.5)
            plt.legend()
            plt.xlim(-half_window_size - 0.05, half_window_size + 0.05)
            plt.ylim(-half_window_size - 0.05, half_window_size + 0.05)

            filename = self.output_dir.joinpath("figure", wv.util.generate_id() + ".png")
            filename.parent.mkdir(parents=False, exist_ok=True)
            plt.savefig(str(filename))
            plt.close()

            unique_bins = set()
            for trajectory in self.trajectory_list:
                for point in trajectory:
                    bin_index = (floor(point[0] / bin_size), floor(point[1] / bin_size))
                    unique_bins.add(bin_index)
            diversity = len(unique_bins)
        else:
            diversity = 0

        print("2D diversity : ", diversity)

    def calculate_action_diversity(self):
        num_sample = 100
        num_obs = 500

        diversity_list = []
        for _ in range(num_obs):
            obs = self.env.reset()
            obs_th = torch.from_numpy(obs).to(self.device, torch.float32)

            action_list = []
            for _ in range(num_sample):
                state = self.env.obs2state(obs_th)
                z_ = self.sampler.sample(state, dt=0.1)
                action = self.policy.decoder(torch.cat([state, z_], dim=-1))
                action_list.append(action[0].cpu().detach().numpy())
            action_list = np.array(action_list)

            diversity_list.append(np.var(action_list, axis=0).sum())

        diversity = np.mean(diversity_list)
        print("action diversity : ", diversity)

    def predict_motion(self, obs, state, z, action, splined_action=None):
        if splined_action is None:
            splined_action = disc_cubic_spline_action(
                self.env.task._desk_size,
                action.reshape(1, -1, 2).cpu().detach().numpy(),
                obs,
                self.env.action_scale,
                self.env.len_traj,
                self.env.total_time_length,
            )[:, :2]

        motion_pred = self.motion_decoder.sample(
            state,
            torch.from_numpy(4 * splined_action.reshape(1, -1)).to(
                self.device,
                torch.float32,
            ),
        )
        motion_pred = motion_pred[0].detach().cpu().numpy()
        return motion_pred

    def _prepare_output_dir(self, path):
        os.makedirs(path, exist_ok=True)
        os.makedirs(path / "media", exist_ok=True)
        os.makedirs(path / "figure", exist_ok=True)

    def _set_seed(self, seed):
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _to_container(self, cfg_node):
        if OmegaConf.is_config(cfg_node):
            return OmegaConf.to_container(cfg_node, resolve=True)
        return copy.deepcopy(cfg_node)
