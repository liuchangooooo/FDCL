import numpy as np
import matplotlib
from matplotlib import animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import torch
import copy
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
import pathlib
import wandb
import wandb.sdk.data_types.video as wv
import os
import pickle
from DIVO.utils.LieGroup_torch import log_SO3, skew
import random
from scipy.interpolate import CubicSpline
import logging
logging.getLogger('matplotlib.animation').setLevel(logging.WARNING)

def get_anim(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.close('all')
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
      im.set_data(frame)
      return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    
    return anim

def display_video(anim):
    res = anim.to_jshtml()

    return HTML(res)

def save_anim(frames, name, fps=200, framerate=500):
    FFwriter = animation.FFMpegWriter(fps=fps)
    anim = get_anim(frames, framerate=framerate)
    anim.save(f'{name}.mp4', writer = FFwriter)

def disc_cubic_spline_action(obs_lim,action,obs,action_scale,len_traj,total_time_length, vel_lim=1.0):
    dt = float(total_time_length)/len_traj
    spline_lim = obs_lim - 0.015
    action_lim = 1.0

    theta = np.arctan2(obs[0,3],obs[0,2])

    for idx in range(1000):
        if analytic_rod_collision_check(theta, action[0,0,:2]*(2*obs_lim)/action_scale, 0.01):
            action[0,0,:2] = action[0,0,:2] + action[0,0,:2] * 0.1
            action[0,0,:2] = np.clip(action[0,0,:2],-action_lim,action_lim)
        else:
            break

    action_ = copy.deepcopy(action[0])
    action_ = action_*(2*obs_lim)/action_scale + obs[0,0:2]*obs_lim

    t = np.linspace(0, 1, len(action_))
    cs_x = CubicSpline(t, action_[:, 0], bc_type=((1, 0), (1, 0)))
    cs_y = CubicSpline(t, action_[:, 1], bc_type=((1, 0), (1, 0)))

    t_ = np.linspace(t.min(), t.max(), len_traj)
    x_ = cs_x(t_)
    y_ = cs_y(t_)

    cubic_spline_pos = np.zeros((len_traj, 2))
    cubic_spline_pos[:,0] = x_
    cubic_spline_pos[:,1] = y_
    
    cubic_spline_pos = np.clip(cubic_spline_pos,-spline_lim,spline_lim)

    cubic_spline_vel = np.concatenate([[[0.0, 0.0]], (cubic_spline_pos[1:] - cubic_spline_pos[:-1])/dt])

    if (cubic_spline_vel>vel_lim).any() or (cubic_spline_vel<-vel_lim).any():
        cubic_spline_vel[cubic_spline_vel>vel_lim] = vel_lim
        cubic_spline_vel[cubic_spline_vel<-vel_lim] = -vel_lim

        pos_recon = np.zeros_like(cubic_spline_pos)
        pos_recon[0,:] = cubic_spline_pos[0,:]

        for i in range(2):
            for j in range(len_traj-1):
                pos_recon[j+1,i] = pos_recon[j,i] + cubic_spline_vel[j,i]*dt

        cubic_spline_pos = pos_recon
        cubic_spline_pos = np.clip(cubic_spline_pos,-action_lim,action_lim)

    action_ = np.concatenate([cubic_spline_pos, cubic_spline_vel], axis=1)
    return action_

def conti_cubic_spline_action(env,action,obs,action_scale,len_traj,total_time_length,vel_lim=1.0):
    dt = float(total_time_length)/len_traj
    action_ = action[0]

    action_ = action_*env.task._desk_size/action_scale + obs[0,2:4]
    
    init_rod_pos = env.task._desk_size*obs[0,:2].reshape(1,2)
    action_ = np.concatenate([init_rod_pos,action_])

    t = torch.linspace(0,1,action_.shape[0])
    x = torch.from_numpy(action_[:, :2])
    coeffs = natural_cubic_spline_coeffs(t, x)
    spline = NaturalCubicSpline(coeffs)
    point = torch.linspace(0, 1, len_traj)
    out = spline.evaluate(point)
    cubic_spline_pos = np.zeros((len_traj, 2))
    cubic_spline_pos[:,1] = out[:,1]
    cubic_spline_pos[:,0] = out[:,0]

    action_lim = env.task._desk_size - 0.05
    cubic_spline_pos = np.clip(cubic_spline_pos,-action_lim,action_lim)

    cubic_spline_vel = np.concatenate([[[0.0, 0.0]], (cubic_spline_pos[1:] - cubic_spline_pos[:-1])/dt])

    if (cubic_spline_vel>vel_lim).any() or (cubic_spline_vel<-vel_lim).any():
        cubic_spline_vel[cubic_spline_vel>vel_lim] = vel_lim
        cubic_spline_vel[cubic_spline_vel<-vel_lim] = -vel_lim

        pos_recon = np.zeros_like(cubic_spline_pos)
        pos_recon[0,:] = cubic_spline_pos[0,:]

        for i in range(2):
            for j in range(len_traj-1):
                pos_recon[j+1,i] = pos_recon[j,i] + cubic_spline_vel[j,i]*dt

        cubic_spline_pos = pos_recon
        cubic_spline_pos = np.clip(cubic_spline_pos,-action_lim,action_lim)

    action_ = np.concatenate([cubic_spline_pos, cubic_spline_vel], axis=1)

    return action_

def analytic_obs_collision_check(Tblock_angle, obs_center, obs_size, threshold=0.01):
    horizontal_length = 0.10
    horizontal_thickness = 0.03
    horizontal_center = (0, 0)

    vertical_length = 0.07
    vertical_thickness = 0.03
    vertical_center = (0, -0.05)

    def rotate_point_around_origin(point, center, angle):
        px, py = point
        ox, oy = center
        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy
    
    def edge_vectors(points):
        return [points[(i+1) % len(points)] - points[i] for i in range(len(points))]

    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm

    def project_polygon(axis, polygon):
        dots = [np.dot(vertex, axis) for vertex in polygon]
        return min(dots), max(dots)

    def overlap(minA, maxA, minB, maxB):
        return maxA >= minB and maxB >= minA

    def separating_axis_theorem(polygon1, polygon2):
        edges = edge_vectors(polygon1) + edge_vectors(polygon2)
        axes = [normalize(np.array([-edge[1], edge[0]])) for edge in edges]
        
        for axis in axes:
            minA, maxA = project_polygon(axis, polygon1)
            minB, maxB = project_polygon(axis, polygon2)
            if not overlap(minA, maxA, minB, maxB):
                return False
        return True
    
    half_side = (obs_size+threshold) / 2
    vertices_arbitrary_square = np.array([
        (obs_center[0] - half_side, obs_center[1] - half_side),
        (obs_center[0] + half_side, obs_center[1] - half_side),
        (obs_center[0] + half_side, obs_center[1] + half_side),
        (obs_center[0] - half_side, obs_center[1] + half_side),
    ])

    for rect_center, width, height in [(horizontal_center, horizontal_length, horizontal_thickness), 
                                       (vertical_center, vertical_thickness, vertical_length)]:

        vertices = [
            (rect_center[0] - width / 2, rect_center[1] - height / 2),
            (rect_center[0] + width / 2, rect_center[1] - height / 2),
            (rect_center[0] + width / 2, rect_center[1] + height / 2),
            (rect_center[0] - width / 2, rect_center[1] + height / 2)
        ]
        rotated_vertices = np.array([rotate_point_around_origin(v, (0, 0), Tblock_angle) for v in vertices])

        if separating_axis_theorem(rotated_vertices, vertices_arbitrary_square):
            return True

    return False

def analytic_rod_collision_check(Tblock_angle, circle_center, circle_radius, threshold=0.01):
    horizontal_length = 0.1
    horizontal_thickness = 0.03
    horizontal_center = (0, 0)

    vertical_length = 0.07
    vertical_thickness = 0.03
    vertical_center = (0, -0.05)

    def rotate_point_around_origin(point, angle):
        x, y = point
        qx = np.cos(angle) * x - np.sin(angle) * y
        qy = np.sin(angle) * x + np.cos(angle) * y
        return (qx, qy)

    def rect_circle_collision_check(rect_center, width, height, angle, circle_center, circle_radius):
        cx, cy = rect_center
        half_width = width / 2
        half_height = height / 2
        vertices = [
            (cx - half_width, cy - half_height),
            (cx + half_width, cy - half_height),
            (cx + half_width, cy + half_height),
            (cx - half_width, cy + half_height)
        ]

        rotated_vertices = [rotate_point_around_origin((vx - cx, vy - cy), angle) for vx, vy in vertices]
        rotated_vertices = [(vx + cx, vy + cy) for vx, vy in rotated_vertices]

        for i in range(len(rotated_vertices)):
            start = rotated_vertices[i]
            end = rotated_vertices[(i + 1) % len(rotated_vertices)]

            edge_vec = (end[0] - start[0], end[1] - start[1])
            circle_vec = (circle_center[0] - start[0], circle_center[1] - start[1])
            edge_length = np.sqrt(edge_vec[0]**2 + edge_vec[1]**2)
            edge_unit_vec = (edge_vec[0] / edge_length, edge_vec[1] / edge_length)
            projection_length = edge_unit_vec[0] * circle_vec[0] + edge_unit_vec[1] * circle_vec[1]
            closest_point = (start[0] + edge_unit_vec[0] * projection_length, start[1] + edge_unit_vec[1] * projection_length)

            if projection_length < 0:
                closest_point = start
            elif projection_length > edge_length:
                closest_point = end
            
            distance_to_circle = np.sqrt((circle_center[0] - closest_point[0])**2 + (circle_center[1] - closest_point[1])**2)
            
            if distance_to_circle < circle_radius:
                return True
        
        return False
    
    if rect_circle_collision_check(horizontal_center, horizontal_length, horizontal_thickness, Tblock_angle, circle_center, circle_radius+threshold):
        return True
    if rect_circle_collision_check(vertical_center, vertical_thickness, vertical_length, Tblock_angle, circle_center, circle_radius+threshold):
        return True

    return False

def analytic_rod_obs_collision_check(rod_center, rod_radius, obs_center, obs_size):
    half_width = obs_size / 2

    dx = abs(rod_center[0] - obs_center[0])
    dy = abs(rod_center[1] - obs_center[1])

    if dx > (half_width + rod_radius) or dy > (half_width + rod_radius):
        return False

    if dx <= half_width or dy <= half_width:
        return True

    corner_distance_sq = (dx - half_width) ** 2 + (dy - half_width) ** 2

    return corner_distance_sq <= (rod_radius ** 2)

class StateDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, obs_dim, action_dim, len_traj, latent_dim, dataset_size='all', num_samples=1):
        file_list = os.listdir(dataset_dir)
        data_list = [file for file in file_list if file.endswith(".pickle")]
        random.shuffle(data_list)

        if dataset_size != 'all':
            data_list = data_list[:dataset_size]

        self.len = len(data_list)*num_samples
        print(self.len)
        if num_samples>1:
            self.motion_decoder = False
            dict_for_evals = {}
        else:
            self.motion_decoder = True

        if self.motion_decoder:
            self.obs_buf= np.zeros((self.len, obs_dim), dtype=np.float32)
            self.act_buf = np.zeros((self.len, action_dim))
            self.splined_act_buf = np.zeros((self.len, len_traj, 2))
            self.z_buf = np.zeros((self.len, latent_dim))
            self.reward_buf = np.zeros((self.len, 1))
            self.motion_buf = np.zeros((self.len, len_traj, obs_dim))
        else:
            self.obs_buf= np.zeros((self.len, obs_dim), dtype=np.float32)
            self.z_buf = np.zeros((self.len, latent_dim))

        for idx, data in enumerate(data_list):
            with open(os.path.join(dataset_dir, data), 'rb') as f:
                data = pickle.load(f)
                if self.motion_decoder:
                    if data['motion'].shape[0] < len_traj:
                        continue
                    elif data['motion'].shape[0] == len_traj:
                        self.obs_buf[idx] = data['obs'][0].cpu().detach().numpy()
                        self.act_buf[idx] = data['action']
                        self.splined_act_buf[idx] = data['splined_action'][:,:2]
                        self.z_buf[idx] = data['z'][0]
                        self.reward_buf[idx] = data['reward']
                        self.motion_buf[idx] = data['motion']
                else:
                    if data['motion'].shape[0] < len_traj:
                        continue
                    elif data['motion'].shape[0] == len_traj:
                        self.obs_buf[idx*num_samples:(idx+1)*num_samples] = data['obs'].cpu().detach().numpy()
                        self.z_buf[idx*num_samples:(idx+1)*num_samples] = data['z']

    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        if self.motion_decoder:
            return self.obs_buf[idx], self.act_buf[idx], self.splined_act_buf[idx], self.z_buf[idx], self.reward_buf[idx], self.motion_buf[idx]
        else:
            return self.obs_buf[idx], self.z_buf[idx]
        
    def get_dict_for_evals(self):
        dict_for_evals = {}
        for te, tr in zip(self.obs_buf, self.z_buf):
            if te in dict_for_evals.keys():
                dict_for_evals[te] += [tr.unsqueeze(0)]
            else:
                dict_for_evals[te] = [tr.unsqueeze(0)]
                
        for key, item in dict_for_evals.items():
            dict_for_evals[key] = torch.cat(item, dim=0)
        return dict_for_evals

class MMDCalculator:
    def __init__(self, type, num_episodes=100, kernel_mul=1, kernel_num=1, bandwidth_base=None):
        self.num_episodes = num_episodes
        self.type = type
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.bandwidth_base = bandwidth_base

    def calculate_squared_distance(self, x, y):
        if self.type == 'SE3':
            batch_size = len(x)

            T_x = x.reshape(-1, 4, 4)
            T_y = y.reshape(-1, 4, 4)

            dist_R = skew(log_SO3(torch.einsum('bij,bjk->bik', T_x[:, :3, :3].permute(0, 2, 1), T_y[:, :3, :3])))
            dist_p = T_x[:, :3, 3] - T_y[:, :3, 3]
            
            return torch.sum(dist_R**2 + dist_p**2, dim=1).reshape(batch_size, batch_size)
        elif self.type == 'L2':
            return ((x - y)**2).sum(dim=2)
        else:
            raise NotImplementedError(f"Type {self.type} is not implemented. Choose type between 'SE3' and 'L2'.")

    def gaussian_kernel(self, source, target):
        total = torch.cat([source, target], dim=0)
        total0 = torch.repeat_interleave(total.unsqueeze(1), len(total), dim=1)
        total1 = torch.repeat_interleave(total.unsqueeze(0), len(total), dim=0)

        distance_squared = self.calculate_squared_distance(total0, total1)

        if self.bandwidth_base == None:
            self.bandwidth_base = torch.sum(distance_squared) / (len(total)**2 - len(total))

        self.bandwidth_base /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [self.bandwidth_base * (self.kernel_mul**i) for i in range(self.kernel_num)]

        kernel_val = [torch.exp(-distance_squared / bandwidth) for bandwidth in bandwidth_list]
        
        kernel_matrix = sum(kernel_val)
        if not torch.all(kernel_matrix >= 0):
            print(kernel_matrix)
            raise ValueError("Kernel matrix contains negative values, which should not happen.")
        
        return sum(kernel_val)

    def __call__(self, source, target):
        assert len(source) <= len(target), f"The number of samples in source {len(source)} must be less than or equal to the number of samples in target {len(target)}."

        batch_size = len(source)

        mmd_list = []

        for _ in range(self.num_episodes):
            target_ = target[np.random.choice(range(len(target)), len(source), replace=False)]

            kernels = self.gaussian_kernel(source, target_)

            XX = kernels[:batch_size, :batch_size]
            YY = kernels[batch_size:, batch_size:]
            XY = kernels[:batch_size, batch_size:]
            YX = kernels[batch_size:, :batch_size]
            
            for mat in [XX, YY, XY, YX]:
                if not torch.all(mat >= 0):
                    raise ValueError("One of the kernel matrices XX, YY, XY, YX contains negative values.")

            mmd = torch.mean(XX + YY - XY - YX).item()

            mmd_list += [mmd]

        mmd_avg = sum(mmd_list) / len(mmd_list)

        return mmd_avg
    
def calculate_MMD(env, policy, sampler, observation, output_dir):
    batch_size = observation.shape[0]
    random_indices = torch.randint(low=0, high=batch_size, size=(100,))
    observation_subset = observation[random_indices]

    device = observation.device

    mmd_calculator = MMDCalculator('L2', num_episodes=100, kernel_mul=1, kernel_num=1)

    MMD_log = {}
    MMD_score = []
    z_list = []
    z_sample_list = []
    z_sample_traj_list = []
    num_sample = 100

    for idx, obs in enumerate(observation_subset):
        obs = obs.cpu().detach().numpy()
        num=0
        random_obs_list = []
        if obs.shape[-1] == 6:
            while num<num_sample:
                theta = np.arctan2(obs[3],obs[2])
                random_obs = np.random.uniform(-1.,1.,(2))
                if analytic_obs_collision_check(Tblock_angle=theta,
                                                    obs_center=random_obs[:2]*env._task._desk_size-obs[:2]*env._task._desk_size,
                                                    obs_size=env.task._obstacle_size*2,
                                                    threshold=0.02*2):
                    continue
                else:
                    random_obs_list.append(random_obs)
                    num+=1
            random_obs_list = np.array(random_obs_list)
            random_obs_th = torch.from_numpy(random_obs_list).to(device,torch.float32)
            new_obs = torch.from_numpy(obs.reshape(1,-1)).to(device,torch.float32)
            new_obs = new_obs.repeat(num_sample,1)
            new_obs[:,-2:] = random_obs_th
        else:
            random_obs = env.sample_obstacle_pose(num_sample)[:,:2]
            random_obs_th = torch.from_numpy(random_obs).to(device,torch.float32)
            new_obs = torch.from_numpy(obs.reshape(1,-1)).to(device,torch.float32)
            new_obs = new_obs.repeat(num_sample,1)
            new_obs[:,-2:] = random_obs_th

        z = policy.encoder(new_obs)

        state = env.obs2state(new_obs)
        z_sample = sampler.sample(state,dt=0.1,output_traj=True,device=device)

        z_traj = copy.deepcopy(z_sample)
        z_sample = z_sample[:,-1,:] # last state

        MMD_score.append(mmd_calculator(z,z_sample))

        if idx < 3:
            z_list.append(z.cpu().detach().numpy())
            z_sample_list.append(z_sample.cpu().detach().numpy())
            z_sample_traj_list.append(z_traj.cpu().detach().numpy())

    MMD_score = np.array(MMD_score).mean().item()

    z_paths, z_traj_paths = plot_z(z_list, z_sample_list, z_sample_traj_list, output_dir)

    for idx, (z_path, z_traj_path) in enumerate(zip(z_paths, z_traj_paths)):
        MMD_log[f'eval/z_{idx}'] = wandb.Image(z_path)
        MMD_log[f'eval/z_traj_{idx}'] = wandb.Image(z_traj_path)

    return MMD_score, MMD_log

def plot_z(z_list, z_sample_list, z_sample_traj_list, output_dir):

    z_paths = []
    for z, z_sample in zip(z_list, z_sample_list):
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(221, projection='3d')
        ax.scatter(z[:,0], z[:,1], z[:,2], c='b', label='z')
        ax.scatter(z_sample[:,0], z_sample[:,1], z_sample[:,2], c='r', label='z_sample')
        ax.legend()
        ax.view_init(elev=30, azim=30)
        ax2 = fig.add_subplot(222, projection='3d')
        ax2.scatter(z[:,0], z[:,1], z[:,2], c='b', label='z')
        ax2.scatter(z_sample[:,0], z_sample[:,1], z_sample[:,2], c='r', label='z_sample')
        ax2.legend()
        ax2.view_init(elev=30, azim=120)
        ax3 = fig.add_subplot(223, projection='3d')
        ax3.scatter(z[:,0], z[:,1], z[:,2], c='b', label='z')
        ax3.scatter(z_sample[:,0], z_sample[:,1], z_sample[:,2], c='r', label='z_sample')
        ax3.legend()
        ax3.view_init(elev=30, azim=210)
        ax4 = fig.add_subplot(224, projection='3d')
        ax4.scatter(z[:,0], z[:,1], z[:,2], c='b', label='z')
        ax4.scatter(z_sample[:,0], z_sample[:,1], z_sample[:,2], c='r', label='z_sample')
        ax4.legend()
        ax4.view_init(elev=30, azim=300)
        plt.tight_layout()
        
        filename = pathlib.Path(output_dir).joinpath(
            'figure', wv.util.generate_id() + ".png")
        filename.parent.mkdir(parents=False, exist_ok=True)
        filename = str(filename)
        plt.savefig(filename)
        plt.close()
        z_paths.append(filename)

    z_traj_paths = []
    for i, (z, z_sample_traj) in enumerate(zip(z_list, z_sample_traj_list)):
        traj_len = z_sample_traj.shape[1]
        fig, axs = plt.subplots(4, traj_len, figsize=(3*traj_len,12), subplot_kw={'projection':'3d'})

        for j in range(traj_len):
            axs[0,j].scatter(z[:,0], z[:,1], z[:,2], s=10, c='b', label='z')
            axs[0,j].scatter(z_sample_traj[:,j,0], z_sample_traj[:,j,1], z_sample_traj[:,j,2], s=10, c='r', alpha=0.3, label='z_sample')
            axs[0,j].view_init(elev=30, azim=30)

            axs[1,j].scatter(z[:,0], z[:,1], z[:,2], s=10, c='b', label='z')
            axs[1,j].scatter(z_sample_traj[:,j,0], z_sample_traj[:,j,1], z_sample_traj[:,j,2], s=10, c='r', alpha=0.3, label='z_sample')
            axs[1,j].view_init(elev=30, azim=120)

            axs[2,j].scatter(z[:,0], z[:,1], z[:,2], s=10, c='b', label='z')
            axs[2,j].scatter(z_sample_traj[:,j,0], z_sample_traj[:,j,1], z_sample_traj[:,j,2], s=10, c='r', alpha=0.3, label='z_sample')
            axs[2,j].view_init(elev=30, azim=210)

            axs[3,j].scatter(z[:,0], z[:,1], z[:,2], s=10, c='b', label='z')
            axs[3,j].scatter(z_sample_traj[:,j,0], z_sample_traj[:,j,1], z_sample_traj[:,j,2], s=10, c='r', alpha=0.3, label='z_sample')
            axs[3,j].view_init(elev=30, azim=300)
        handles, labels = axs[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        plt.tight_layout()

        filename = pathlib.Path(output_dir).joinpath(
            'figure', wv.util.generate_id() + ".png")
        filename.parent.mkdir(parents=False, exist_ok=True)
        filename = str(filename)
        plt.savefig(filename)
        plt.close()
        z_traj_paths.append(filename)

    return z_paths, z_traj_paths

def splined_action2SE3_traj(env, action):
    time_step = action.shape[0]
    SE3_traj = np.tile(np.array([[np.sin(np.pi/4), -np.cos(np.pi/4), 0, 0.0],
                                    [-np.cos(np.pi/4), -np.sin(np.pi/4), 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]], dtype=float), (time_step, 1, 1))
    SE3_traj[:, :2, 3] = action
    SE3_traj[:, :3, 3] += env._task.robot_bias

    set_qpos = np.array([-np.pi/2, 0, 0, -1.07079, 0, 1.07079, -0.7853])
    qpos_seq = [set_qpos]

    return SE3_traj

def crop_leftmost_red_box(image_np, red_threshold, low_threshold, crop_size):
    """
    Crops an image centered around the leftmost box with high red intensity
    and low green and blue values (white background scenario).

    Parameters:
        image_np (np.ndarray): The input image as a NumPy array with shape (H, W, 3).
        red_threshold (int): Minimum value for the red channel to consider a box (e.g., 200).
        low_threshold (int): Maximum value for the green and blue channels (e.g., 50).
        crop_size (int): Half of the width and height of the cropping box.

    Returns:
        cropped_image (np.ndarray): The cropped image as a NumPy array.
    """
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError("Input image must be a 3D NumPy array with shape (H, W, 3).")

    # Create a mask for high red intensity and low green/blue values
    red_mask = (image_np[:, :, 0] > red_threshold) & \
               (image_np[:, :, 1] < low_threshold) & \
               (image_np[:, :, 2] < low_threshold)

    # Get the coordinates of the red pixels
    red_coords = np.column_stack(np.where(red_mask))

    if len(red_coords) == 0:
        raise ValueError("No high red boxes found in the image!")

    # Find the leftmost red pixel
    leftmost_index = np.argmin(red_coords[:, 1])  # Index of the smallest x-coordinate
    center_y, center_x = red_coords[leftmost_index]
    center_x += int(crop_size/2)
    # Define the crop box
    height, width, _ = image_np.shape
    top = max(center_y - crop_size, 0)
    bottom = min(center_y + crop_size, height)
    left = max(center_x - crop_size, 0)
    right = min(center_x + crop_size, width)
    # Crop the image
    cropped_image = image_np[top:bottom, left:right, :]

    return cropped_image