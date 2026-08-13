import copy
import os
import random
import time

import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf

from DIVO.RL.component import OrnsteinUhlenbeckProcess, StateDictReplayBuffer, hard_update, soft_update
from DIVO.common.checkpoint_util import TopKCheckpointManager
from DIVO.common.pytorch_util import dict_to_torch, optimizer_to
from DIVO.critic import get_critic
from DIVO.env import get_env_class
from DIVO.evaluator import get_evaluator
from DIVO.policy import get_policy
from DIVO.workspace.base_workspace import BaseWorkspace


class TD3AntWorkspace(BaseWorkspace):
    """TD3 workspace for Gymnasium-style AntNavObstacle.

    This class intentionally lives beside the original TD3 workspace instead of
    editing it. It adapts reset/step signatures and keeps the DIVO TD3 model,
    critic, replay buffer, and logging structure.
    """

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        self.device = torch.device(cfg.training.device)
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.env = get_env_class(**cfg.env)
        self.no_obs_env = get_env_class(**cfg.no_obs_env)
        self.unseen_env = get_env_class(**cfg.unseen_env)
        self.action_dim, self.obs_dim = self.env.get_info()

        self.model = get_policy(self.env, **cfg.policy).to(self.device)
        self._load_policy_init_checkpoint_if_needed()
        self.model_target = get_policy(self.env, **cfg.policy).to(self.device)
        hard_update(self.model_target, self.model)

        self.critic = get_critic(**cfg.critic).to(self.device)
        self.critic_target = get_critic(**cfg.critic).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.evaluator = get_evaluator(**cfg.evaluator)
        self.optimizer = hydra.utils.get_class(cfg.optimizer._target_)(
            self.model.parameters(),
            lr=cfg.optimizer.lr,
        )
        self.critic_optimizer = hydra.utils.get_class(cfg.critic_optimizer._target_)(
            self.critic.parameters(),
            lr=cfg.critic_optimizer.lr,
        )

        self.critic_gradient_clip = cfg.rl.critic_gradient_clip
        self.critic_gradient_max_norm = cfg.rl.critic_gradient_max_norm
        self.policy_gradient_clip = cfg.rl.policy_gradient_clip
        self.policy_gradient_max_norm = cfg.rl.policy_gradient_max_norm

        if cfg.rl.add_noise:
            self.random_process = OrnsteinUhlenbeckProcess(
                size=cfg.action_size,
                theta=0.15,
                mu=0,
                sigma=cfg.rl.noise_sigma,
            )

        self.replay_buffer = StateDictReplayBuffer(
            cfg.rl.replay_buffer_size,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
        )
        self.global_step = 0
        self.epoch = 0
        self.num_timesteps = 0
        self.gamma = cfg.rl.gamma
        self.save_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

    def _load_policy_init_checkpoint_if_needed(self):
        init_path = self.cfg.training.get("policy_init_checkpoint", None)
        if init_path in (None, "", "null"):
            return

        init_path = os.path.expanduser(str(init_path))
        if not os.path.isabs(init_path):
            init_path = os.path.abspath(init_path)
        if not os.path.exists(init_path):
            raise FileNotFoundError(f"Ant policy_init_checkpoint not found: {init_path}")

        payload = torch.load(init_path, map_location=self.device)
        if isinstance(payload, dict) and "state_dicts" in payload:
            state_dict = payload["state_dicts"]["model"]
        else:
            state_dict = payload
        self.model.load_state_dict(state_dict)
        print(f"[Ant] Initialized policy from checkpoint: {init_path}")

    def reset_env(self, env, seed=None):
        result = env.reset(seed=seed)
        if isinstance(result, tuple):
            return result[0]
        return result

    def step_env(self, env, action):
        result = env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return obs, reward, bool(terminated or truncated), info
        return result

    def obs_to_tensor(self, obs):
        if isinstance(obs, dict):
            return dict_to_torch(obs, device=self.device)
        return torch.tensor(obs, dtype=torch.float32).to(device=self.device)

    def learn(self):
        start_time = time.time()
        episode_reward_logger = []
        episode_reward = 0.0
        episode_length = 0
        num_episode = 0
        max_test_score = -100.0
        val_reward = -100.0
        step_log = {}
        reward_component_logger = {}
        self.updates = 0
        self.critic_update = 0

        cfg = copy.deepcopy(self.cfg)
        if cfg.log:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging,
            )
        else:
            wandb_run = None

        optimizer_to(self.optimizer, self.device)
        optimizer_to(self.critic_optimizer, self.device)
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        obs = self.reset_env(self.env, seed=cfg.training.seed)
        self.model.reset()
        epsilon = cfg.rl.noise_epsilon
        self.depsilon = 1.0 / epsilon
        self.epsilon = 1.0

        with tqdm.tqdm(total=cfg.training.num_epochs, ncols=80, desc="Train updates") as pbar:
            with tqdm.tqdm(total=cfg.rl.warmup, ncols=80, desc="Warm Up", leave=False) as pbar2:
                while self.updates < cfg.training.num_epochs:
                    self.model.eval()
                    obs_th = self.obs_to_tensor(obs)

                    with torch.no_grad():
                        if self.num_timesteps <= cfg.rl.warmup:
                            action = np.random.uniform(
                                self.env.action_space.low.flat[0],
                                self.env.action_space.high.flat[0],
                                self.action_dim,
                            ).reshape(1, *self.action_dim)
                        else:
                            action = self.model.predict_action(obs_th)
                            action = action.detach().cpu().numpy()
                            if cfg.rl.add_noise:
                                random_noise = self.random_process.sample().reshape(*action.shape)
                                action += random_noise * max(self.epsilon, 0)
                                self.epsilon -= self.depsilon
                                action = np.clip(
                                    action,
                                    self.env.action_space.low.flat[0],
                                    self.env.action_space.high.flat[0],
                                )

                    next_obs, reward, done, info = self.step_env(self.env, action[0])
                    self.replay_buffer.add(obs, next_obs, action, reward, done)
                    reward_dict = info.get("reward_dict", {}) if isinstance(info, dict) else {}
                    for key in (
                        "goal_velocity_reward",
                        "raw_goal_velocity",
                        "progress_reward",
                        "weighted_progress_reward",
                        "ctrl_cost",
                        "goal_bonus",
                        "collision_penalty",
                        "unhealthy_penalty",
                        "near_obstacle_penalty",
                        "clearance_gap",
                    ):
                        if key in reward_dict:
                            reward_component_logger.setdefault(key, []).append(float(reward_dict[key]))

                    obs = next_obs
                    episode_reward += float(reward)
                    episode_length += 1
                    self.num_timesteps += 1

                    if episode_length >= cfg.max_steps - 1:
                        done = True

                    pbar2.update(1)
                    if done:
                        episode_reward_logger.append(episode_reward)
                        self.model.reset()
                        obs = self.reset_env(self.env)
                        episode_length, episode_reward = 0, 0.0
                        num_episode += 1

                    if self.num_timesteps > cfg.rl.warmup:
                        if self.num_timesteps == cfg.rl.warmup + 1:
                            pbar2.close()
                        self.model.train()
                        training_info = self.update(batch_size=cfg.rl.batch_size)
                        self.updates += 1
                        pbar.update(1)
                        pbar.set_postfix(
                            episode_reward=np.mean(episode_reward_logger[-100:])
                            if episode_reward_logger
                            else 0.0
                        )

                        step_log = {
                            "train/episode_reward": float(
                                np.mean(episode_reward_logger[-100:])
                                if episode_reward_logger
                                else 0.0
                            ),
                            "train/num_episode": int(num_episode),
                            "train/num_timesteps": int(self.num_timesteps),
                            "time/step_ps": float(self.num_timesteps / max(time.time() - start_time, 1e-6)),
                            **training_info,
                        }
                        for key, values in reward_component_logger.items():
                            if values:
                                step_log[f"train/reward/{key}"] = float(
                                    np.mean(values[-cfg.log_interval :])
                                )

                        if cfg.training.validate and self.updates % cfg.training.validate_steps == 0:
                            val_reward, val_log = self.evaluator(self.env, self.model, self.no_obs_env, self.unseen_env)
                            _ = self.reset_env(self.env)
                            step_log.update({"validate_reward": float(val_reward), **val_log})

                        if wandb_run is not None and self.updates % cfg.log_interval == 0:
                            wandb_run.log(step_log, step=self.updates)

                        if self.updates % cfg.training.checkpoint_every == 0:
                            if cfg.checkpoint.save_last_ckpt:
                                self.save_checkpoint()
                                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "model_latest.pt"))
                            if cfg.checkpoint.save_last_snapshot:
                                self.save_snapshot()

                            if max_test_score < val_reward:
                                max_test_score = val_reward
                                metric_dict = {"test_mean_score": val_reward, "epoch": self.updates}
                                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                                if topk_ckpt_path is not None:
                                    torch.save(self.model.state_dict(), topk_ckpt_path)

        if cfg.checkpoint.save_last_ckpt:
            self.save_checkpoint()
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, "model_latest.pt"))

        if wandb_run is not None:
            wandb_run.finish()

    def update(self, batch_size):
        cfg = copy.deepcopy(self.cfg)
        experience_replay = self.replay_buffer.sample(batch_size=batch_size)

        with torch.no_grad():
            next_q_value = self.compute_next_q_value(experience_replay)
            rewards = torch.from_numpy(experience_replay.rewards).to(self.device)
            dones = torch.from_numpy(experience_replay.dones).to(self.device)
            target_q_value = rewards + (1 - dones) * self.gamma * next_q_value
            target_q_value = target_q_value.detach()

        self.critic_optimizer.zero_grad()
        critic_loss, critic_loss_info = self.compute_critic_loss(experience_replay, target_q_value)
        critic_loss.backward()
        if self.critic_gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_gradient_max_norm)
        parameters = [p for p in self.critic.parameters() if p.grad is not None and p.requires_grad]
        if parameters:
            critic_gradient_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach()).to(self.device) for p in parameters]),
                2.0,
            ).item()
            critic_loss_info["[critic]gradient_norm"] = critic_gradient_norm
        self.critic_optimizer.step()

        if self.critic_update % cfg.rl.policy_update == 0:
            self.optimizer.zero_grad()
            policy_loss, policy_loss_info = self.compute_policy_loss(experience_replay)
            policy_loss.backward()
            if self.policy_gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.policy_gradient_max_norm)
            parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
            if parameters:
                policy_gradient_norm = torch.norm(
                    torch.stack([torch.norm(p.grad.detach()).to(self.device) for p in parameters]),
                    2.0,
                ).item()
                policy_loss_info["[policy]gradient_norm"] = policy_gradient_norm
            policy_loss_info["[policy]replay_rewards"] = float(experience_replay.rewards.mean())
            self.optimizer.step()
        else:
            policy_loss_info = {}

        soft_update(self.critic_target, self.critic, cfg.rl.soft_update_tau)
        if self.critic_update % cfg.rl.policy_update == 0:
            soft_update(self.model_target, self.model, cfg.rl.soft_update_tau)

        self.critic_update += 1
        return {**critic_loss_info, **policy_loss_info}

    def compute_next_q_value(self, experience_replay):
        next_obs_th = self.obs_to_tensor(experience_replay.next_observations)
        next_action = self.model_target.predict_action(next_obs_th)
        next_q_values = self.critic_target(next_obs_th, next_action)
        next_q_values = torch.cat(next_q_values, dim=1)
        next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
        return next_q_values

    def compute_critic_loss(self, experience_replay, target_q_value):
        obs_th = self.obs_to_tensor(experience_replay.observations)
        action = torch.from_numpy(experience_replay.actions).to(self.device)
        current_q_values = self.critic(obs_th, action)
        critic_loss = 0.5 * sum(
            [torch.nn.functional.mse_loss(current_q, target_q_value) for current_q in current_q_values]
        )
        return critic_loss.to(self.device), {"[critic]loss": critic_loss.item()}

    def compute_policy_loss(self, experience_replay):
        obs_th = self.obs_to_tensor(experience_replay.observations)
        if "regularize_z" in self.cfg.training:
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
        policy_loss = -current_q_values.mean()
        if "regularize_z" in self.cfg.training:
            if self.cfg.training.regularize_z == "norm":
                policy_loss += self.cfg.training.reg_coeff * (torch.norm(z, dim=1) ** 2).mean()
            elif self.cfg.training.regularize_z == "gaussian":
                feature_loss = torch.nn.functional.mse_loss(z_mean, torch.full_like(z_mean, 0)) + torch.nn.functional.mse_loss(
                    z_var, torch.full_like(z_var, 1)
                )
                policy_loss += self.cfg.training.reg_coeff * feature_loss
            elif self.cfg.training.regularize_z is False:
                pass

        return policy_loss.to(self.device), {"[policy]loss": policy_loss.item()}
