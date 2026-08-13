if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import torch
from torch.nn import functional as F
from omegaconf import OmegaConf
import pathlib
import random
import numpy as np
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

class TD3Workspace(BaseWorkspace):
    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        self.device = torch.device(cfg.training.device)
        
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # set env
        self.env = get_env_class(**cfg.env)
        self.no_obs_env = get_env_class(**cfg.no_obs_env)
        self.unseen_env = get_env_class(**cfg.unseen_env)
        self.action_dim, self.obs_dim = self.env.get_info()
        
        print("\n [1] Env is set:")
        
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
        self.gamma=cfg.rl.gamma
        self.save_dir = os.path.join(
            self.output_dir, 
            'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        
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

        obs = self.env.reset()
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

                    #sample action
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
                    
                    #env step
                    next_obs, reward, done, info = self.env.step(action[0]) 
                    
                    if isinstance(obs, dict):
                        obs_th = dict_to_torch(obs, device=self.device)
                    else:
                        if len(next_obs.shape) < 3:
                            next_obs = np.concatenate((next_obs,np.random.uniform(-1,1,self.env.obs_dim[0]-next_obs.shape[-1]).reshape(1,-1)),axis=1)

                    #Replay buffer
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
                        self.env.seed()
                        self.model.reset()
                        obs = self.env.reset()

                        episode_length, episode_reward = 0, 0
                        num_episode += 1
                    
                    #update model
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

                        #log
                        if cfg.log and self.updates % cfg.log_interval == 0:
                            with torch.no_grad():
                                wandb_run.log({
                                    **training_info,
                                    'epispde_mean_reward': np.mean(episode_reward_logger[-100:]),
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
            next_obs_th =  torch.tensor(next_obs, dtype=torch.float32).to(device=self.device)
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
            obs_th =  torch.tensor(obs, dtype=torch.float32).to(device=self.device)
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
            obs_th =  torch.tensor(obs, dtype=torch.float32).to(device=self.device)
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

                policy_loss += self.cfg.training.reg_coeff* feature_loss
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