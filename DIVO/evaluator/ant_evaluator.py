import os
import pathlib

import numpy as np
import torch
import wandb
import wandb.sdk.data_types.video as wv

from DIVO.evaluator.base_evaluator import BaseEvaluator
from DIVO.utils.util import save_anim


class AntNavEvaluator(BaseEvaluator):
    def video_render_fn(self, env, policy, render):
        self.render_video = bool(render)
        if render:
            self.frames = []
            self.filename = pathlib.Path(self.output_dir).joinpath(
                "media", wv.util.generate_id() + ".mp4"
            )
            self.filename.parent.mkdir(parents=True, exist_ok=True)
            self.filename = str(self.filename)
            self.video_paths.append(self.filename)

    def frame_save_fn(self, env, policy):
        if not self.render_video:
            return
        frame = env.render()
        if frame is not None:
            self.frames.append(frame)

    def video_save_fn(self, env, policy):
        if getattr(self, "frames", None):
            save_anim(self.frames, self.filename[:-4], fps=20, framerate=20)

    def sample_z_fn(self, policy, obs):
        return None

    def sample_action_fn(self, policy, obs, z=None):
        action = policy.predict_action(obs)
        return action.detach().cpu().numpy()

    def rollout(self, env, policy, episode, episode_reset_fn=None, post_step_fn=None):
        if episode_reset_fn is not None:
            obs = episode_reset_fn(env, episode)
        else:
            obs, _ = env.reset()
        obs_th = torch.tensor(obs, dtype=torch.float32).to(device=self.device)

        episode_steps = 0
        done = False
        reward = 0.0
        info = {"success": False, "trajectory": np.zeros((0, 2), dtype=np.float32)}
        trajectory = []

        while not done:
            obs_th = torch.tensor(obs, dtype=torch.float32).to(device=self.device)
            action = self.sample_action_fn(policy, obs_th)
            obs, reward, terminated, truncated, info = env.step(action[0])
            done = bool(terminated or truncated)
            trajectory.append(np.asarray(info.get("xy_position", [0.0, 0.0]), dtype=np.float32))

            if post_step_fn is not None:
                maybe_obs = post_step_fn(env, obs, reward, done, info, episode, episode_steps)
                if maybe_obs is not None:
                    obs = maybe_obs

            if episode < self.num_save_video:
                self.frame_save_fn(env, policy)

            if self.max_steps and episode_steps >= self.max_steps - 1:
                done = True

            if self.reward == "mean":
                self.episode_reward += reward
            elif self.reward == "max":
                self.episode_reward.append(reward)
            elif self.reward == "last":
                pass
            else:
                raise NotImplementedError

            episode_steps += 1
            self.end_of_step_fn(policy)

        if self.reward == "max":
            self.episode_reward = max(self.episode_reward)
        elif self.reward == "last":
            self.episode_reward = reward

        info["trajectory"] = np.asarray(trajectory, dtype=np.float32)
        return obs, reward, done, info

    def __call__(self, env, policy, no_obs_env=None, unseen_env=None, episode_reset_fn=None, post_step_fn=None):
        mean_reward, log_data = super().__call__(
            env,
            policy,
            no_obs_env=no_obs_env,
            unseen_env=unseen_env,
            episode_reset_fn=episode_reset_fn,
            post_step_fn=post_step_fn,
        )
        return mean_reward, log_data
