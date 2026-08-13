if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import torch
from omegaconf import OmegaConf
import pathlib
import random
import numpy as np
from DIVO.workspace.base_workspace import BaseWorkspace
import hydra
import wandb
import tqdm
import copy

from DIVO.common.pytorch_util import optimizer_to
from DIVO.model.common.lr_scheduler import get_scheduler                       

from DIVO.motion_decoder import get_motion_decoder
from DIVO.env import get_env_class
from DIVO.policy import get_policy
from DIVO.utils.util import StateDataset, save_anim, disc_cubic_spline_action
import wandb.sdk.data_types.video as wv

class MotionDecoderWorkspace(BaseWorkspace):
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
        self.action_dim, self.obs_dim = self.env.get_info()
        
        print("\n [1] Env is set:")
        
        # configure model
        self.model = get_motion_decoder(
            **cfg.motion_decoder
        ).to(self.device)
        
        print("\n [2] Motion Decoder is set:")
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
        
        self.load_dataloader(cfg, dataset_dir)
        self.train_motion_decoder(cfg, dataset_dir)

    def load_dataloader(self, cfg, dataset_dir):
        dataset_dir = os.path.join(cfg.dataset_dir,'dataset')
        batch_size = cfg.training.batch_size        
        dataset = StateDataset(dataset_dir=dataset_dir,
                                obs_dim=self.env.obs_dim[0],
                                action_dim=self.env.action_dim[0],
                                len_traj=self.env.len_traj,
                                latent_dim=self.policy.encoder.out_chan,
                                dataset_size=cfg.training.dataset_size)
        print("Dataset Loaded")
        
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
    
    def train_motion_decoder(self, cfg, dataset_dir):
        now = cfg.now
        motion_decoder_path = os.path.join(dataset_dir, 'motion_decoder', f'{now}_{cfg.task_name}')
        os.makedirs(motion_decoder_path, exist_ok=True)
        if cfg.log:
            wandb_run = wandb.init(
                dir=str(motion_decoder_path),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
        
        train_dataloader, val_dataloader = self.load_dataloader(cfg, dataset_dir)
        print("Dataloader Loaded")

        optimizer_to(self.optimizer, self.device)
        
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
                    for obs, action, splined_action, z, _, motion in tepoch:
                        obs = obs.to(self.device, dtype=torch.float32)
                        action = action.to(self.device, dtype=torch.float32)
                        splined_action = splined_action.to(self.device, dtype=torch.float32)
                        z = z.to(self.device, dtype=torch.float32)
                        motion = motion[:,:,:4].to(self.device, dtype=torch.float32)

                        state = self.env.obs2state(obs)
                        state = state
                        # optimize
                        self.optimizer.zero_grad()
                        with torch.autograd.set_detect_anomaly(True):
                            splined_action = splined_action.reshape(state.shape[0],-1)
                            loss = self.model.compute_loss(state,splined_action,motion)

                            loss.backward()
                        self.optimizer.step()
                        lr_scheduler.step()

                        # logging
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)
                tglobal.set_postfix(loss=np.mean(epoch_loss))

                if epoch_idx % cfg.log_interval == 0:
                    wandb_run.log({'[motion_decoder] train_loss': np.mean(epoch_loss).item()}, step = epoch_idx)

                if epoch_idx % cfg.training.validate_steps == 0:
                    val_loss = list()
                    with torch.no_grad():
                        self.model.eval()
                        for obs, action, splined_action, z, _, motion in val_dataloader:
                            obs = obs.to(self.device, dtype=torch.float32)
                            action = action.to(self.device, dtype=torch.float32)
                            splined_action = splined_action.to(self.device, dtype=torch.float32)
                            z = z.to(self.device, dtype=torch.float32)
                            motion = motion[:,:,:4].to(self.device, dtype=torch.float32)

                            state = self.env.obs2state(obs)
                            state = state

                            splined_action = splined_action.reshape(state.shape[0],-1)
                            loss = self.model.compute_loss(state,splined_action,motion)

                            loss_cpu = loss.item()
                            val_loss.append(loss_cpu)

                        if epoch_idx % cfg.log_interval == 0:
                            video_log = self.save_video(self.env, self.policy, self.model, motion_decoder_path)
                            wandb_run.log(video_log, step = epoch_idx)
                            wandb_run.log({'[motion_decoder] val_loss': np.mean(val_loss).item()},step = epoch_idx)

                    if min_valid > np.mean(val_loss).item():
                        min_valid = np.mean(val_loss).item()
                        torch.save(self.model.state_dict(), f'{motion_decoder_path}/{cfg.motion_decoder._target_}_epoch={epoch_idx}_valloss={min_valid:.4f}.pt')
                        
                    if epoch_idx%cfg.training.checkpoint_every == 0:
                        torch.save(self.model.state_dict(), f'{motion_decoder_path}/{cfg.motion_decoder._target_}_epoch={epoch_idx}_valloss={min_valid:.4f}.pt')

    def save_video(self, env, policy, motion_decoder, motion_decoder_path, sampler=None):
        video_paths = []

        for _ in range(self.cfg.training.num_validate_videos):
            
            env._record_frame = True
            frames = []
            filename = pathlib.Path(motion_decoder_path).joinpath(
                'media', wv.util.generate_id() + ".mp4")
            filename.parent.mkdir(parents=False, exist_ok=True)
            filename = str(filename)
            video_paths.append(filename)
            
            env.seed()
            obs = env.reset()

            episode_steps = 0
            done = False
            while not done:
                obs_th = torch.tensor(obs, dtype=torch.float32).to(device=self.device)

                if policy.encoder.in_chan > obs_th.size(1):
                    random_padding = torch.rand(obs_th.size(0), policy.encoder.in_chan - obs_th.size(1)).to(self.device)
                    obs_th = torch.cat([obs_th, random_padding*2-1], dim=1)
                z = policy.encoder(obs_th)
                
                state = env.obs2state(obs_th)
                action = policy.decoder(torch.cat([state,z],dim=-1))
                
                splined_action = disc_cubic_spline_action(env._task._desk_size,
                                                    action.reshape(1,-1,2).cpu().detach().numpy(),
                                                    obs,
                                                    env.action_scale,
                                                    env.len_traj,
                                                    env.total_time_length
                                                    )[:,:2]
                motion_pred = motion_decoder.sample(state,torch.from_numpy((splined_action/self.env._task._desk_size).reshape(1,-1)).to(self.device,torch.float32))
                motion_pred = motion_pred[0].detach().cpu().numpy()

                obs, reward, done, info = env.step(action[0].detach().cpu().numpy(), motion_pred)
                   
                frames += env.frames

                if self.cfg.max_steps and episode_steps >= self.cfg.max_steps -1:
                    done = True

                episode_steps += 1
            
            save_anim(frames, filename[:-4], fps=int(1/env.control_timestep()))
        
        log_data = dict()
        for idx, video_path in enumerate(video_paths):
            sim_video = wandb.Video(video_path)
            log_data[f'eval/sim_video_{idx}'] = sim_video

        return log_data
