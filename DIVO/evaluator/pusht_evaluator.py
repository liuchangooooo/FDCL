import numpy as np
import torch
import tqdm
import pathlib
import wandb
import wandb.sdk.data_types.video as wv
import warnings
from DIVO.utils.util import *
import os
from DIVO.evaluator.base_evaluator import BaseEvaluator

class MujocoPushTEvaluator(BaseEvaluator):
    def __init__(
        self, 
        output_dir=None, 
        num_episodes=20, 
        max_steps=10,
        device='cuda:0',
        num_save_video=3,
        reward='mean',
        *args,
        **kwargs
    ):
        super().__init__(
            output_dir=output_dir, 
            num_episodes=num_episodes, 
            max_steps=max_steps,
            device=device,
            num_save_video=num_save_video,
            reward=reward,
            *args,
            **kwargs
        )
        self._obs_dim_warning_cache = set()
        
    def video_render_fn(self, env, policy, render):
        if render:
            env._record_frame = True
            self.frames = []
            self.filename = pathlib.Path(self.output_dir).joinpath(
                'media', wv.util.generate_id() + ".mp4")
            self.filename.parent.mkdir(parents=False, exist_ok=True)
            self.filename = str(self.filename)
            self.video_paths.append(self.filename)
        else:
            env._record_frame = False

    def frame_save_fn(self, env, policy):
        self.frames += env.frames

    def video_save_fn(self, env, policy):
        save_anim(self.frames, self.filename[:-4], fps=int(1/env.control_timestep()))

    def sample_z_fn(self, policy, obs):
        obs = self._adapt_obs_dim(policy, obs)
        z = None

        return z

    def sample_action_fn(self, policy, obs, z):
        obs = self._adapt_obs_dim(policy, obs)

        action = policy.predict_action(obs)         
        
        return action.detach().cpu().numpy()
    
    def end_of_step_fn(self, policy):
        if hasattr(policy, 'num_skills'):
            policy.reset_skill = False

    def _adapt_obs_dim(self, policy, obs):
        if not hasattr(policy, 'encoder'):
            return obs

        expected_dim = policy.encoder.in_chan
        current_dim = obs.size(1)

        if current_dim < expected_dim:
            random_padding = torch.rand(
                obs.size(0),
                expected_dim - current_dim,
                device=obs.device,
            )
            return torch.cat([obs, random_padding * 2 - 1], dim=1)

        if current_dim > expected_dim:
            key = (current_dim, expected_dim)
            if key not in self._obs_dim_warning_cache:
                warnings.warn(
                    f"Evaluator observed {current_dim} dims but policy expects "
                    f"{expected_dim}; truncating trailing obstacle features.",
                    stacklevel=2,
                )
                self._obs_dim_warning_cache.add(key)
            return obs[:, :expected_dim]

        return obs
