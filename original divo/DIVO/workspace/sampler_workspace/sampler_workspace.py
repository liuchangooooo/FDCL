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
import dill
import pickle
import shutil
import datetime

from DIVO.common.pytorch_util import optimizer_to
from DIVO.model.common.lr_scheduler import get_scheduler                       

from DIVO.env import get_env_class
from DIVO.sampler import get_sampler
from DIVO.policy import get_policy
from DIVO.utils.util import StateDataset
from diffusers.training_utils import EMAModel
from DIVO.utils.util import calculate_MMD, analytic_obs_collision_check

class SamplerWorkspace(BaseWorkspace):
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
        self.env.seed(seed)
        self.action_dim, self.obs_dim = self.env.get_info()
        
        print("\n [1] Env is set:")
        
        # configure model
        self.model = get_sampler(
            **cfg.sampler
        ).to(self.device)

        if cfg.ema._target_ != 'None':
            self.ema_model = get_sampler(
                **cfg.sampler
            ).to(self.device)
            self.ema_model = copy.deepcopy(self.model)
        
        print("\n [2] Sampler is set:")
        print(self.model)

        #configure policy
        policy_checkpoint = cfg.policy.checkpoint
        p_cfg = OmegaConf.load(f'{policy_checkpoint.rsplit("/",2)[0]}/.hydra/config.yaml')
        self.policy = get_policy(self.env, **p_cfg.policy).to(self.device)
        self.policy.load_state_dict(torch.load(f'{policy_checkpoint}'))

        print("\n [3] Policy is set:")
        print(self.policy)
        
        self.optimizer = hydra.utils.get_class(
            cfg.optimizer._target_)(
            self.model.parameters(), 
            lr=cfg.optimizer.lr)
        
        self.gradient_clip = cfg.training.gradient_clip
        self.gradient_max_norm = cfg.training.gradient_max_norm
        
    def learn(self):
        cfg = copy.deepcopy(self.cfg)
        dataset_dir = cfg.dataset_dir
        os.makedirs(dataset_dir, exist_ok=True)
            
        # Generate Dataset
        if cfg.dataset.generate:
            self.generate_dataset(cfg, dataset_dir)
        # Train Sampler
        if cfg.sampler.train:
            self.train_sampler(cfg, dataset_dir)

    def generate_dataset(self, cfg, dataset_dir):
        dataset_dir = os.path.join(dataset_dir, 'sampler_dataset')
        os.makedirs(dataset_dir, exist_ok=True)
        num_episodes = cfg.dataset.num_episodes
        num_samples = cfg.dataset.num_samples
        pbar = tqdm.tqdm(total=num_episodes, ncols=50, desc=f"Generate Dataset")
        with torch.no_grad():
            for episode in range(num_episodes):            
                obs = self.env.reset()         
                episode_length = 0
                done = False
                data_list = []
                while not done:
                    num=0
                    random_obs_list = []
                    while num<num_samples:
                        theta = np.arctan2(obs[0,3],obs[0,2])
                        random_obs = np.random.uniform(-1.,1.,(2))
                        if analytic_obs_collision_check(Tblock_angle=theta,
                                                obs_center=random_obs[:2]*self.env._task._desk_size-obs[0,:2]*self.env._task._desk_size,
                                                obs_size=self.env.task._obstacle_size*2,
                                                threshold=0.02*2):
                            continue
                        else:
                            random_obs_list.append(random_obs)
                            num+=1

                    random_obs_list = np.array(random_obs_list)
                    random_obs_th = torch.from_numpy(random_obs_list).to(self.device,torch.float32)
                    new_obs = torch.from_numpy(obs.reshape(1,-1)).to(self.device,torch.float32)
                    new_obs = new_obs.repeat(num_samples,1)
                    new_obs[:,-2:] = random_obs_th

                    z = self.policy.encoder(new_obs)
                    new_obs = new_obs[:,:self.env.observation_space.shape[0]]
                    
                    obs_th = torch.tensor(obs, dtype=torch.float32).to(device=self.device)
                    action = self.policy.predict_action(obs_th)
                    action = action.detach().cpu().numpy()

                    next_obs, reward, done, info = self.env.step(action[0])

                    obs = next_obs
                    episode_length += 1
                    
                    motion = info['trajectory']
                    splined_action = info['splined_action']/self.env._task._desk_size

                    data = {
                            'obs' : new_obs, # num_samplesXobs_dim
                            'action' : action[0], # action_dim
                            'splined_action' : splined_action, # len_trajX2
                            'z' : z.cpu().numpy(), # num_samplesX3
                            'reward' : reward, # 1
                            'motion' : motion # len_trajXobs_dim
                            }
                    data_list.append(data)
                    
                    if info['success']:
                        done = True

                    if episode_length >= cfg.max_steps-1:
                        done = True

                if info['success']:
                    for episode_length, data in enumerate(data_list):
                        with open(f'{dataset_dir}/{episode}_{episode_length}_{reward:.3f}.pickle', 'wb') as f:
                            pickle.dump(data, f)
                            
                if done:
                    self.env.seed()

                pbar.update(1)
            pbar.close()

    def load_dataloader(self, cfg, dataset_dir):
        dataset_dir = os.path.join(dataset_dir, 'dataset')
        dataset = StateDataset(dataset_dir=dataset_dir,
                                obs_dim=self.env.obs_dim[0],
                                action_dim=self.env.action_dim[0],
                                len_traj=self.env.len_traj,
                                latent_dim=self.policy.encoder.out_chan,
                                dataset_size=cfg.training.dataset_size,
                                num_samples=cfg.dataset.num_samples,)

        batch_size = cfg.training.batch_size
        print("Dataset Loaded : ", len(dataset))
        
        train_ratio = cfg.training.train_ratio
        validate_ratio = cfg.training.validate_ratio

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        np.random.shuffle(indices)

        split = int(np.floor(validate_ratio * dataset_size))
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

        train_dataloader = torch.utils.data.DataLoader(dataset, 
                                                batch_size=batch_size, 
                                                sampler=train_sampler,
                                                num_workers=4,
                                                pin_memory=True,
                                                persistent_workers=True)
        val_dataloader = torch.utils.data.DataLoader(dataset, 
                                                batch_size=batch_size, 
                                                sampler=valid_sampler,
                                                num_workers=4,
                                                pin_memory=True,
                                                persistent_workers=True)
        
        return train_dataloader, val_dataloader
    
    def train_sampler(self, cfg, dataset_dir):
        now = cfg.now
        sampler_path = os.path.join(dataset_dir, 'sampler', f'{now}_{cfg.task_name}')
        os.makedirs(sampler_path, exist_ok=True)
        if cfg.log:
            wandb_run = wandb.init(
                dir=str(sampler_path),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
        
        train_dataloader, val_dataloader = self.load_dataloader(cfg, dataset_dir)
        print("Dataloader Loaded")

        optimizer_to(self.optimizer, self.device)

        self.model.normalize(train_dataloader, self.device)

        if cfg.ema._target_ != 'None':
            ema: EMAModel = None
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
            self.ema_model.normalize(train_dataloader, self.device)
        
        lr_scheduler = get_scheduler(
            name=cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs),
            last_epoch=-1
        )

        min_valid = 1e6
        with tqdm.tqdm(range(cfg.training.num_epochs), ncols=50, desc='Epoch') as tglobal:
            # epoch loop
            for epoch_idx in tglobal:
                epoch_loss = list()
                with tqdm.tqdm(train_dataloader, ncols=50, desc='Batch', leave=False) as tepoch:
                    self.model.train()
                    
                    if cfg.ema._target_ != 'None':
                        self.ema_model.train()

                    for obs, z in tepoch:
                        obs = obs.to(self.device)
                        z = z.to(self.device)

                        state = self.env.obs2state(obs)

                        # optimize
                        self.optimizer.zero_grad()

                        loss = self.model.compute_loss(state,z)

                        loss.backward()
                        self.optimizer.step()
                        lr_scheduler.step()

                        if cfg.ema._target_ != 'None':
                            ema.step(self.model)

                        # logging
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)
                    tglobal.set_postfix(loss=np.mean(epoch_loss))
                
                if epoch_idx % cfg.log_interval == 0:
                    wandb_run.log({'[sampler] train_loss': np.mean(epoch_loss).item()}, step = epoch_idx)
                
                if epoch_idx % cfg.training.validate_steps == 0:
                    val_loss = list()
                    MMD_score_list = list()
                    with torch.no_grad():
                        if cfg.ema._target_ != 'None':
                            model = copy.deepcopy(self.ema_model)
                        else:
                            model = copy.deepcopy(self.model)

                        model.eval()
                        
                        with tqdm.tqdm(val_dataloader, ncols=50, desc='Validate', leave=False) as tval:
                            for idx, (obs, z) in enumerate(tval):
                                obs = obs.to(self.device)
                                z = z.to(self.device)

                                state = self.env.obs2state(obs)

                                loss = model.compute_loss(state,z)

                                loss_cpu = loss.item()
                                val_loss.append(loss_cpu)

                                if idx == 0:
                                    MMD_score, MMD_log = calculate_MMD(env=self.env,
                                                            policy=self.policy,
                                                            sampler=model,
                                                            observation=obs,
                                                            output_dir=sampler_path)
                                    MMD_score_list.append(MMD_score)

                        if epoch_idx % cfg.log_interval == 0:
                            wandb_run.log({'[sampler] val_loss': np.mean(val_loss).item()},step = epoch_idx)
                            wandb_run.log({'MMD_score': np.mean(MMD_score_list).item()},step = epoch_idx)
                            wandb_run.log(MMD_log, step=epoch_idx)

                    if min_valid > np.mean(val_loss).item():
                        min_valid = np.mean(val_loss).item()
                        torch.save(self.model.state_dict(), f'{sampler_path}/{cfg.sampler._target_}_epoch={epoch_idx}_valloss={min_valid:.4f}.pt')
                        if cfg.ema._target_ != 'None':
                            self.ema_model.load_state_dict(ema.averaged_model.state_dict(), strict=False)
                            torch.save(self.ema_model.state_dict(), f'{sampler_path}/{cfg.sampler._target_}_ema_epoch={epoch_idx}_valloss={min_valid:.4f}.pt')

                    if epoch_idx%cfg.training.checkpoint_every == 0:
                        torch.save(self.model.state_dict(), f'{sampler_path}/{cfg.sampler._target_}_epoch={epoch_idx}_valloss={min_valid:.4f}.pt')
                        if cfg.ema._target_ != 'None':
                            torch.save(self.ema_model.state_dict(), f'{sampler_path}/{cfg.sampler._target_}_ema_epoch={epoch_idx}_valloss={min_valid:.4f}.pt')
                            self.ema_model.load_state_dict(ema.averaged_model.state_dict(), strict=False)
