import numpy as np
import torch
import tqdm
import wandb
import os
from math import floor
import matplotlib.pyplot as plt
import pathlib
import wandb.sdk.data_types.video as wv
from DIVO.common.pytorch_util import dict_to_torch

class BaseEvaluator(object):
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
        os.makedirs(output_dir,exist_ok=True)
        self.output_dir = output_dir
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.device = device
        self.reward = reward
        self.num_save_video = num_save_video
        
    def __call__(self,
                 env,
                 policy,
                 no_obs_env=None,
                 unseen_env=None):        
        with torch.no_grad():
            self.result = []
            self.video_paths = []
            self.trajectory_list = []
            num_success = 0

            pbar = tqdm.tqdm(total=self.num_episodes, ncols=50, desc=f"Validate seen env", leave=False)
            for episode in range(self.num_episodes):
                if episode < self.num_save_video:
                    self.video_render_fn(env, policy, True)
                else:
                    self.video_render_fn(env, policy, False)

                if self.reward == 'mean':
                    self.episode_reward = 0.
                elif self.reward == 'max':
                    self.episode_reward = []

                _, _, _, info = self.rollout(env, policy, episode)

                if info['success']:
                    num_success += 1
                
                pbar.update(1)
                self.result.append(self.episode_reward)       
                                
                if episode < self.num_save_video:
                    self.video_save_fn(env, policy)

            pbar.close()
            self.result = np.array(self.result).reshape(-1,1)

            log_data = dict()
            for idx, video_path in enumerate(self.video_paths):
                sim_video = wandb.Video(video_path)
                log_data[f'eval/sim_video_{idx}'] = sim_video
                log_data[f'eval/reward_{idx}'] = self.result[idx]
            log_data[f'eval/success_rate[seen]'] = num_success / self.num_episodes

        return np.mean(self.result), log_data

    def rollout(self, env, policy, episode):
        # start episode
        env.seed()
        obs = env.reset()
        if isinstance(obs, dict):
            obs_th = dict_to_torch(obs, device=self.device)
        else:
            obs_th = torch.tensor(obs, dtype=torch.float32).to(device=self.device)

        episode_steps = 0
        done = False

        z = self.sample_z_fn(policy, obs_th)

        trajectory = []
        while not done:
            if isinstance(obs, dict):
                obs_th = dict_to_torch(obs, device=self.device)
            else:
                obs_th = torch.tensor(obs, dtype=torch.float32).to(device=self.device)

            action = self.sample_action_fn(policy, obs_th, z)         
            
            obs, reward, done, info = env.step(action[0])

            if len(trajectory) == 0:
                trajectory.append(info['trajectory'])

            if episode < self.num_save_video:                        
                self.frame_save_fn(env, policy)

            if self.max_steps and episode_steps >= self.max_steps -1:
                done = True

            # update
            if self.reward == 'mean':
                self.episode_reward += reward
            elif self.reward == 'max':
                self.episode_reward.append(reward)
            elif self.reward == 'last':
                pass
            else:
                raise NotImplementedError
            episode_steps += 1

            self.end_of_step_fn(policy)

        
        if self.reward == 'max':
            self.episode_reward = max(self.episode_reward)
        elif self.reward == 'last':
            self.episode_reward = reward

        if info['success']:
            trajectory = np.concatenate(trajectory, axis=0)
            self.trajectory_list.append(trajectory)

        return obs, reward, done, info

    def video_render_fn(self, env, policy, render):
        pass

    def video_save_fn(self, env, policy):
        pass

    def sample_action_fn(self, policy, obs):
        return None

    def sample_z_fn(self, policy, obs):
        return None

    def frame_save_fn(self, env, policy):
        pass

    def diversity_fn(self, policy):
        pass

    def end_of_step_fn(self, policy):
        pass
    
    def measure_diversity(self):
        if len(self.trajectory_list)>0:
            half_window_size = 1.
            bin_size = 0.1

            num_bins = int(2 * half_window_size / bin_size)
            for i in range(num_bins + 1):
                line_pos = i * bin_size - half_window_size
                plt.axvline(x=line_pos, color='gray', linewidth=0.5)
                plt.axhline(y=line_pos, color='gray', linewidth=0.5)

            for idx, trajectory_ in enumerate(self.trajectory_list):
                if idx == 0:
                    plt.scatter(0,0,c='r',label='goal')
                    plt.scatter(trajectory_[0,0],trajectory_[0,1],c='g',label='start')
                else:
                    plt.scatter(0,0,c='r')
                    plt.scatter(trajectory_[0,0],trajectory_[0,1],c='g')
                
                plt.plot(trajectory_[:,0],trajectory_[:,1])

            plt.axvline(x=-half_window_size,color='gray',linewidth=5.5)
            plt.axvline(x=half_window_size,color='gray',linewidth=5.5)
            plt.axhline(y=-half_window_size,color='gray',linewidth=5.5)
            plt.axhline(y=half_window_size,color='gray',linewidth=5.5)
            plt.legend()
            plt.xlim(-half_window_size-0.05,half_window_size+0.05)
            plt.ylim(-half_window_size-0.05,half_window_size+0.05)

            filename = pathlib.Path(self.output_dir).joinpath(
                'figure', wv.util.generate_id() + ".png")
            filename.parent.mkdir(parents=False, exist_ok=True)
            filename = str(filename)
            plt.savefig(filename)
            plt.close()

            unique_bins = set()
            for trajectory in self.trajectory_list:
                for point in trajectory:
                    bin_index = (floor(point[0] / bin_size), floor(point[1] / bin_size))
                    unique_bins.add(bin_index)
            diversity = len(unique_bins)
        else:
            diversity = 0
            filename = None

        return diversity, filename

    

