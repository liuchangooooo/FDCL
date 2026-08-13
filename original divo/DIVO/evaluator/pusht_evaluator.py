import numpy as np
import torch
import tqdm
import pathlib
import wandb
import wandb.sdk.data_types.video as wv
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

        if hasattr(policy, 'encoder'):
            if policy.encoder.in_chan > obs.size(1):
                random_padding = torch.rand(obs.size(0), policy.encoder.in_chan - obs.size(1)).to(obs.device)
                obs = torch.cat([obs, random_padding*2-1], dim=1)

        z = None

        return z

    def sample_action_fn(self, policy, obs, z):
        
        if hasattr(policy, 'encoder'):
            if policy.encoder.in_chan > obs.size(1):
                random_padding = torch.rand(obs.size(0), policy.encoder.in_chan - obs.size(1)).to(obs.device)
                obs = torch.cat([obs, random_padding*2-1], dim=1)

        action = policy.predict_action(obs)         
        
        return action.detach().cpu().numpy()
    
    def end_of_step_fn(self, policy):
        if hasattr(policy, 'num_skills'):
            policy.reset_skill = False
