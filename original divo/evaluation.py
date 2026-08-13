import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
import sys
import pathlib
import hydra
import torch
import numpy as np
import random
from DIVO.utils.util import *

from DIVO.env import get_env_class                
from DIVO.policy import get_policy
from DIVO.sampler import get_sampler
from DIVO.motion_decoder import get_motion_decoder
from math import floor

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('config'))
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    evaluator: Evaluator = cls(cfg)
    evaluator.eval()

if __name__ == "__main__":
    main()

class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        seed = cfg.seed
        self.device = torch.device(cfg.device)

        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.env = get_env_class(**cfg.env)
        self.action_dim, self.obs_dim = self.env.get_info()

        print("\n [1] Env is set:")

        policy_checkpoint = cfg.policy.checkpoint
        self.p_cfg = OmegaConf.load(f'{policy_checkpoint.rsplit("/",2)[0]}/.hydra/config.yaml')
        self.policy = get_policy(self.env, **self.p_cfg.policy).to(self.device)
        self.policy.load_state_dict(torch.load(f'{policy_checkpoint}'))
        self.policy.eval()

        print("\n [2] Policy is set:")
        print(self.policy)

        if cfg.sampler.checkpoint != 'None':
            sampler_checkpoint = cfg.sampler.checkpoint
            self.s_cfg = OmegaConf.load(f'{sampler_checkpoint.rsplit("/",1)[0]}/.hydra/config.yaml')
            self.sampler = get_sampler(**self.s_cfg.sampler).to(self.device)
            self.sampler.load_state_dict(torch.load(f'{sampler_checkpoint}', map_location=self.device))
            self.sampler.eval()

            print("\n [3] Sampler is set:")
            print(self.sampler)

        else:
            raise ValueError("Sampler checkpoint is not provided")
        
        if cfg.motion_decoder.checkpoint != 'None':
            motion_decoder_checkpoint = cfg.motion_decoder.checkpoint
            self.m_cfg = OmegaConf.load(f'{motion_decoder_checkpoint.rsplit("/",1)[0]}/.hydra/config.yaml')
            self.motion_decoder = get_motion_decoder(**self.m_cfg.motion_decoder).to(self.device)
            self.motion_decoder.load_state_dict(torch.load(f'{motion_decoder_checkpoint}'))
            self.motion_decoder.eval()
            
            print("\n [3] Motion decoder is set:")
            print(self.motion_decoder)

        else:
            raise ValueError("Motion decoder checkpoint is not provided")

        self.output_dir = cfg.output_dir
        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(self.output_dir,'media')).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(self.output_dir,'figure')).mkdir(parents=True, exist_ok=True)
        
        retain_z = False
        sac = False
    def eval(self):
        fps = int(1/self.env.control_timestep())
        
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
                    obs_th = torch.from_numpy(obs).to(self.device,torch.float32)
                    if obstacle_pose_ is not None:
                        obstacle_pose = copy.deepcopy(obstacle_pose_)
                    else:
                        obstacle_pose = obs[0, 4:]*self.env.task._desk_size
                    if self.cfg.env.obstacle:
                        num_obstacle =  len(obstacle_pose)//2
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
                            action = self.policy.decoder(torch.cat([state_,z_],dim=-1))
                            splined_action = disc_cubic_spline_action(self.env.task._desk_size,
                                                                    action.reshape(1,-1,2).cpu().detach().numpy(),
                                                                    obs,
                                                                    self.env.action_scale,
                                                                    self.env.len_traj,
                                                                    self.env.total_time_length
                                                                    )[:,:2]
                            for action_ in splined_action:
                                for idx in range(num_obstacle):
                                    if analytic_rod_obs_collision_check(action_,
                                                                        0.01,
                                                                        obstacle_pose[2*idx:2*(idx+1)],
                                                                        obstacle_size*2+0.02):
                                        rod_collision = True
                                        break
                                    else:
                                        rod_collision = False
                                if rod_collision:
                                    break
                            if rod_collision and (num_rod_fail < 100):
                                num_rod_fail += 1
                                continue
                            if num_rod_fail >= 100:
                                print("Can't find z to avoid the rod")
                                break

                            motion_pred = self.predict_motion(obs, state_, z_, action, splined_action)

                            for motion in motion_pred:
                                angle = np.arctan2(motion[-1], motion[-2])
                                pred_tblock_pose = motion[:2]*self.env.task._desk_size
                                for idx in range(num_obstacle):
                                    if analytic_obs_collision_check(angle,
                                                                        obstacle_pose[2*idx:2*(idx+1)]-pred_tblock_pose[:2],
                                                                        obstacle_size*2,
                                                                        threshold=0.02):
                                        constraint_done = False
                                        num_constraint_fail += 1
                                        break
                                    else:
                                        constraint_done = True
                                if not constraint_done:
                                    break
                            if (not constraint_done) and (num_constraint_fail > 19):
                                break
                            else:
                                constraint_done = True
                        if constraint_done:
                            state_ = motion_pred[-1].reshape(1,-1)
                            state_ = torch.from_numpy(state_).to(self.device,torch.float32)
                            num_pred += 1
                            num_fail = 0
                        
                        if constraint_done and num_pred > 1:
                            mpc_done = True

                        elif (num_pred > 0) and (not constraint_done) and (num_fail < 20):
                            num_fail += 1
                            continue

                        elif (num_pred > 0) and (not constraint_done) :
                            print("episode_length : ",episode_length, "Can't find the path in the next state")
                            state_ = copy.deepcopy(state)
                            num_pred = 0
                            num_mpc_fail += 1

                        elif (num_pred == 0) and (not constraint_done):
                            print("episode_length : ",episode_length, "Can't find the path in the first state")
                            num_fail += 1
                        
                        if (num_fail > 19) or (num_mpc_fail > 19):
                            print("Can't find the path to avoid the obstacle")
                            break

                    action = self.policy.decoder(torch.cat([state,z],dim=-1))
                    motion_pred = self.predict_motion(obs, state, z, action, None)
                    action = action.cpu().detach().numpy()

                    next_obs, reward, done, info = self.env.step(action,motion_pred)

                    if self.env.obstacle_dist == 'random_step':
                        obstacle_pose_ = self.env._task.set_random_obs_pose(self.env.physics, next_obs)

                    if episode_length == 0:
                        trajectory_.append(info['trajectory'])

                    frames += self.env.frames

                    obs = next_obs
                    episode_length += 1

                    observation_list.append(info['trajectory'])

                    if episode_length > self.cfg.max_steps:
                        done = True

                trajectory_ = np.concatenate(trajectory_,axis=0)

                observation_list = np.concatenate(observation_list,axis=0)
                with open(f'{self.output_dir}/figure/observation_{episode}.npy', 'wb') as f:
                    np.save(f, observation_list)
                with open(f'{self.output_dir}/figure/trajectory_{episode}.npy', 'wb') as f:
                    np.save(f, trajectory_)

                if info['success']:
                    self.trajectory_list.append(trajectory_)
                    print(f"episode : {episode}, episode_length : {episode_length}, reward : {reward:.3f}")

                    if (video_num < self.cfg.num_render) and self.env._record_frame:
                        save_anim(frames, f'{self.output_dir}/media/mujoco_{episode}_{video_num}', fps=fps)

                    video_num += 1
                    num_success += 1
                else:
                    if len(frames)>0 and self.env._record_frame:
                        save_anim(frames, f'{self.output_dir}/media/mujoco_fail_{episode}_{num_episode_fail}', fps=fps)
                    print(f"episode : {episode}, {num_episode_fail}-th fail, episode_length : {episode_length}")
                    if episode_length == 1:
                        num_initialize_fail += 1
                    num_episode_fail += 1

            print(f"success_rate : {num_success/(num_success+num_episode_fail)}, num_initialize_fail : {num_initialize_fail}")

            self.calculate_2D_diversity()
            if self.sampler is not None:
                self.calculate_action_diversity()

    def calculate_2D_diversity(self, env=None):
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
        
        print("2D diversity : ", diversity)

    def calculate_action_diversity(self):
        length_scale = 0.005
        num_sample = 100
        num_obs = 500
        
        diversity_list = []
        for i in range(num_obs):
            obs = self.env.reset()
            obs_th = torch.from_numpy(obs).to(self.device,torch.float32)

            action_list = []
            
            for idx in range(num_sample):
                state = self.env.obs2state(obs_th)
                z_ = self.sampler.sample(state, dt=0.1)
                action = self.policy.decoder(torch.cat([state,z_],dim=-1))
                action_list.append(action[0].cpu().detach().numpy())
            action_list = np.array(action_list)
            
            var = np.var(action_list,axis=0).sum()
            diversity = var
            
            diversity_list.append(diversity)
        
        diversity = np.mean(diversity_list)
        print("action diversity : ", diversity)

    def predict_motion(self, obs, state, z, action, splined_action=None):
        if splined_action is None:
            splined_action = disc_cubic_spline_action(self.env.task._desk_size,
                                            action.reshape(1,-1,2).cpu().detach().numpy(),
                                            obs,
                                            self.env.action_scale,
                                            self.env.len_traj,
                                            self.env.total_time_length
                                            )[:,:2]

        motion_pred = self.motion_decoder.sample(state,torch.from_numpy(4*splined_action.reshape(1,-1)).to(self.device,torch.float32))
        motion_pred = motion_pred[0].detach().cpu().numpy()

        return motion_pred