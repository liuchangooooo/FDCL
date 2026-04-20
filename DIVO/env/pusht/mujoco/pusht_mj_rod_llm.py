"""
Push-T 环境 - 支持 LLM 障碍物配置的版本
基于 pusht_mj_rod.py 修改，添加了 set_obstacle_config() 方法
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

t1 = time.time()
from dm_control import mjcf
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.composer import variation
from dm_control.composer.variation import distributions
from dm_control.locomotion.arenas import floors
import copy
import random

import gym.spaces as spaces
from DIVO.utils.util import disc_cubic_spline_action, analytic_obs_collision_check

global lateral_friction 
lateral_friction = 0.5

class Rod(composer.Entity):
    def _build(self, rod_length = 0.1):
        rod_bias = 0.0
        self._model = mjcf.RootElement()
        rodbody = self._model.worldbody.add('body', name='rodbody')
        rodbody.add('geom', type='cylinder', size=[0.01, rod_length], pos=[rod_bias, rod_bias, 0], mass=1., contype=1, conaffinity=1)
        rodbody.add('joint', name='transX', type='slide', axis=[1, 0, 0])
        rodbody.add('joint', name='transY', type='slide', axis=[0, 1, 0], ref=rod_bias)
        self._model.actuator.add('general', joint='transX')
        self._model.actuator.add('general', joint='transY')
        
    def _build_observables(self):
        return RodObservables(self)

    @property
    def mjcf_model(self):
        return self._model
    
    @property
    def actuators(self):
        return tuple(self._model.find_all("actuator"))
    
    def configure_joints_vel(self, physics, vel):
        joints = self.mjcf_model.find_all('joint', exclude_attachments=True)
        physics.bind(joints).qvel = vel
    
class RodObservables(composer.Observables):
    @composer.observable
    def joint_positions(self):
        all_joints = self._entity.mjcf_model.find_all("joint")
        return observable.MJCFFeature("qpos", all_joints)
    @composer.observable
    def joint_velocities(self):
        all_joints = self._entity.mjcf_model.find_all("joint")
        return observable.MJCFFeature("qvel", all_joints)
    
class TBlock(composer.Entity):
    def _build(self, size=0.04, rgba=[0, 0, 1, 1], contact_dict= {'contype': 1, 'conaffinity':1}):
        self._size = size
        self.rgba = rgba
        self._model = mjcf.RootElement()
        self._model.worldbody.add('geom', name='1', type='box', size=[0.05,0.015,0.01], pos=[0, 0, 0.01],rgba=rgba, **contact_dict, mass=0.1, friction=[lateral_friction, 0.005, 0.0001])
        self._model.worldbody.add('geom', name='2', type='box', size=[0.015,0.035,0.01], pos=[0, -0.05, 0.01], rgba=rgba, **contact_dict, mass=0.1, friction=[lateral_friction, 0.005, 0.0001])

    @property
    def size(self):
        return self._size
    
    @size.setter
    def size(self, value):
        self._size = value
    
    @property
    def mjcf_model(self):
        return self._model

class UniformTBlockSE2(variation.Variation):
    def __init__(self, x=[-1, 1], y=[-1, 1]):
        self._x = distributions.Uniform(low=x[0], high=x[1])
        self._y = distributions.Uniform(low=y[0], high=y[1])
        self._heading = distributions.Uniform(0, 2*np.pi)

    def __call__(self, initial_value=None, current_value=None, random_state=None):
        done = False
        while not done:
            x, y, heading = variation.evaluate(
            (self._x, self._y, self._heading), random_state=random_state)
            if abs(x) < 0.05 and abs(y) < 0.05:
                pass
            else:
                done = True
        return (x, y, heading)
    
class Obstacle(composer.Entity):
    def _build(self, obstacle_size=0.05, obstacle_shape='square'):
        self._model = mjcf.RootElement()
        obstacle_body = self._model.worldbody.add('body', name='obstacle_body')
        if obstacle_shape == 'box':
            obstacle_body.add('geom', name='obstacle', type='box', size=[obstacle_size, obstacle_size, obstacle_size], rgba=[1, 0, 0, 1], mass=100, contype=2, conaffinity=3)
        elif obstacle_shape == 'sphere':
            obstacle_body.add('geom', name='obstacle', type='sphere', size=[obstacle_size], rgba=[1, 0, 0, 1], mass=100, contype=2, conaffinity=3)
        elif obstacle_shape == 'unstructured':
            obstacle_body.add('geom', name='obstacle_1', type='box', size=[obstacle_size, obstacle_size/4, 0.01], pos=[0, obstacle_size*3/4, 0.01], rgba=[1, 0, 0, 1], mass=100, contype=2, conaffinity=3)
            obstacle_body.add('geom', name='obstacle_2', type='box', size=[obstacle_size/4, obstacle_size/2, 0.01], pos=[obstacle_size*3/4, 0, 0.01], rgba=[1, 0, 0, 1], mass=100, contype=2, conaffinity=3)
            obstacle_body.add('geom', name='obstacle_3', type='box', size=[obstacle_size, obstacle_size/4, 0.01], pos=[0, -obstacle_size*3/4, 0.01], rgba=[1, 0, 0, 1], mass=100, contype=2, conaffinity=3)
        self._size = obstacle_size
        
    @property
    def mjcf_model(self):
        return self._model
    
    @property
    def size(self):
        return self._size
        
def zyx_euler_to_mjquat(zyx_euler):
    quat = R.from_euler('zyx', zyx_euler).as_quat()
    return [quat[-1], *quat[:-1]]

def mjquat_to_zyx_euler(mjquat):
    return R.from_quat([*mjquat[1:], mjquat[0]]).as_euler('zyx')


class PushTaskLLM(composer.Task):
    """
    支持 LLM 障碍物配置的 PushTask
    新增: llm_obstacle_config 参数，可以指定障碍物位置
    """

    def __init__(self,
                 obstacle=True,
                 obstacle_num=1,
                 obstacle_size=0.05,
                 obstacle_shape='box',
                 obstacle_dist='random',
                 eval=False,
                 eval_pose=None,
                 motion_pred=False,
                 NUM_SUBSTEPS=None,
                 dynamics_randomization=False):
        
        if NUM_SUBSTEPS==None and motion_pred:
            NUM_SUBSTEPS = 5
        elif NUM_SUBSTEPS==None:
            NUM_SUBSTEPS = 25
        else:
            NUM_SUBSTEPS = NUM_SUBSTEPS
        
        # ===== 新增: LLM 配置存储 =====
        self.llm_obstacle_config = None  # 格式: [{'x': float, 'y': float}, ...]
            
        self.obstacle = obstacle
        self.obstacle_num = obstacle_num
        self.obstacle_shape = obstacle_shape
        self.obstacle_dist = obstacle_dist
        self.eval = eval
        self.eval_pose = eval_pose
        # 一次性强制起点（由 env.reset(force_tblock_pos=True) 写入后在采样时消费）
        self._forced_tblock_pos = None
        self.motion_pred = motion_pred
        self.dynamics_randomization = dynamics_randomization
        self._arena = floors.Floor()

        # Set visualization
        self._arena.mjcf_model.statistic.set_attributes(extent=1.0, center=[0.0, 0, 0.9])
        self._arena.mjcf_model.visual.get_children('global').set_attributes(azimuth=90, elevation=-90, fovy=25)
        self._arena.mjcf_model.visual.get_children('headlight').set_attributes(diffuse=[0.6, 0.6, 0.6], ambient=[0.3, 0.3, 0.3], specular=[0.5, 0.5, 0.5])
        
        light_pos_lst = [[0, 0, 1.5]]
        for light_pos in light_pos_lst:
            self._arena.mjcf_model.worldbody.add('light', pos=light_pos, dir=[0, 0, -1], directional=True)
        
        # Set fixed objects (desk)
        self._desk_size = 0.25
        self._arena.mjcf_model.worldbody.add('geom', name='desk', type='box', 
            size=[self._desk_size, self._desk_size, 0.02/2], pos=[0.0, 0, 1.1], 
            rgba=[250.0/255, 198.0/255, 122.0/255, 1], 
            friction=[lateral_friction, 0.005, 0.0001], conaffinity=3)

        # Set Entities
        rod_length = 0.1
        self._rod = Rod(rod_length)
        self._rod_initial_pose = (0, 0, 1.1 + rod_length + 0.02/2)
        self._arena.attach(self._rod, self._arena.mjcf_model.worldbody.add('site', size=[1e-6]*3, pos=self._rod_initial_pose))

        self._tblock = TBlock(size=0.01, contact_dict={'contype': 1, 'conaffinity': 2})
        self._target_block = TBlock(size=0.01, rgba=[0.17, 0.62, 0.17, 1], contact_dict={'contype': 0, 'conaffinity': 2})

        self._arena.add_free_entity(self._tblock)
        self._arena.attach(self._target_block)

        self._tblock_pos = UniformTBlockSE2(x=[-0.15, 0.15], y=[-0.15, 0.15])
        self._target_block_pos = (0, 0, -np.pi/4)
        
        # Configure variators
        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        # Configure observables
        self._rod.observables.joint_positions.enabled = True
        self._rod.observables.joint_velocities.enabled = True
        self._task_observables = {}

        def get_tblock_pose_decorator(_block_entity):
            def _get_tblock_pose(physics):
                position, mj_quat = _block_entity.get_pose(physics)
                return *position[:2], mjquat_to_zyx_euler(mj_quat)[0]
            return _get_tblock_pose

        self._task_observables['tblock_se2'] = observable.Generic(get_tblock_pose_decorator(self._tblock))
        self._task_observables['target_block_se2'] = observable.Generic(get_tblock_pose_decorator(self._target_block))
        
        if obstacle:
            self._obstacle_size = obstacle_size
            self._obstacle = []
            for i in range(self.obstacle_num):
                self._obstacle.append(Obstacle(obstacle_size=self._obstacle_size, obstacle_shape=self.obstacle_shape))
                self._arena.add_free_entity(self._obstacle[i])
            
            def get_obstacle_pose(physics):
                position = []
                for i in range(self.obstacle_num):
                    pos, _ = self._obstacle[i].get_pose(physics)
                    position.append(pos[:2])
                return np.array(position).reshape(-1)
            
            self._task_observables['obstacle_pos'] = observable.Generic(get_obstacle_pose)

        for obs in self._task_observables.values():
            obs.enabled = True
        self.control_timestep = NUM_SUBSTEPS * self.physics_timestep

        if self.motion_pred:
            self._pred_block = TBlock(size=0.04, rgba=[1.0, 0.01, 0.01, 0.3], contact_dict={'contype': 4, 'conaffinity': 4})
            self._arena.attach(self._pred_block)

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def initialize_episode_mjcf(self, random_state):
        self._mjcf_variator.apply_variations(random_state)
        if self.dynamics_randomization:
            tblock_mass = np.random.uniform(0.05, 0.5)
            for geom in self._tblock.mjcf_model.find_all('geom'):
                geom.set_attributes(mass=tblock_mass)

    def initialize_episode(self, physics, random_state):
        """
        初始化 episode
        
        修改后的流程（与原版一致）:
        1. 设置目标 T 位置（固定）
        2. 如果使用 LLM：先生成 T-block，再生成障碍物
        3. 如果不使用 LLM（验证时）：先生成障碍物，再根据 obstacle_dist 策略生成 T-block
        """
        self._physics_variator.apply_variations(physics, random_state)
        target_block_pose = variation.evaluate((self._target_block_pos), random_state=random_state)
        self._target_block.set_pose(physics, 
            position=(target_block_pose[0], target_block_pose[1], 1.11-0.01*1.99), 
            quaternion=zyx_euler_to_mjquat([target_block_pose[2], 0, 0]))

        if not self.obstacle:
            # 无障碍物模式：直接采样 T-block 位置
            tblock_pose = self._sample_tblock_pose_no_obstacle(target_block_pose)
            self._tblock.set_pose(physics, 
                position=(tblock_pose[0], tblock_pose[1], 1.111), 
                quaternion=zyx_euler_to_mjquat([tblock_pose[2], 0, 0]))
        else:
            if self.llm_obstacle_config is not None:
                # ===== LLM 模式: 先 T-block，再障碍物 =====
                tblock_pose = self._sample_tblock_pose_no_obstacle(target_block_pose)
                self._tblock.set_pose(physics, 
                    position=(tblock_pose[0], tblock_pose[1], 1.111), 
                    quaternion=zyx_euler_to_mjquat([tblock_pose[2], 0, 0]))
                self._apply_llm_obstacle_config(physics, tblock_pose, target_block_pose)
            else:
                # ===== 原版模式（验证时）: 先障碍物，再根据 obstacle_dist 策略生成 T-block =====
                tblock_pose = self._generate_obstacles_and_tblock_original_style(physics, target_block_pose)
                self._tblock.set_pose(physics, 
                    position=(tblock_pose[0], tblock_pose[1], 1.111), 
                    quaternion=zyx_euler_to_mjquat([tblock_pose[2], 0, 0]))

        if self.motion_pred:
            self._pred_block.set_pose(physics, 
                position=(tblock_pose[0], tblock_pose[1], 1.111), 
                quaternion=zyx_euler_to_mjquat([tblock_pose[2], 0, 0]))

    def _sample_tblock_pose_no_obstacle(self, target_block_pose):
        """采样 T-block 位置（不考虑障碍物碰撞）"""
        # 优先级1：一次性强制起点（用于双 reset 的第二次 reset）
        if self._forced_tblock_pos is not None:
            forced_pose = np.array(self._forced_tblock_pos)
            self._forced_tblock_pos = None
            return forced_pose

        # 优先级2：评估模式下的固定起点
        if self.eval and self.eval_pose is not None:
            return np.array(self.eval_pose)
        
        max_attempts = 1000
        for _ in range(max_attempts):
            tblock_pose = np.array(
                np.random.uniform(-0.18, 0.18, 2).tolist() + [np.random.uniform(0, 2*np.pi)]
            )
            
            # 约束1: 不能在中心区域（太靠近目标）
            if abs(tblock_pose[0]) < 0.1 and abs(tblock_pose[1]) < 0.1:
                continue
            
            # 约束2: 初始奖励要足够低（任务要有意义）
            if self.calc_reward(tblock_pose, target_block_pose) > -3.0:
                continue
            
            return tblock_pose
        
        # 找不到则使用默认位置
        return np.array([0.15, 0.15, 3*np.pi/4])


    def _apply_llm_obstacle_config(self, physics, tblock_pose, target_block_pose):
        """
        应用 LLM 生成的障碍物配置
        
        Args:
            physics: MuJoCo physics
            tblock_pose: 初始 T-block 位置 [x, y, theta]
            target_block_pose: 目标 T 位置 [x, y, theta]
        """
        for i in range(min(self.obstacle_num, len(self.llm_obstacle_config))):
            cfg = self.llm_obstacle_config[i]
            obs_x = float(cfg['x'])
            obs_y = float(cfg['y'])
            
            # 验证不与 T-block 碰撞，如果碰撞则微调位置
            obs_x, obs_y = self._adjust_obstacle_position(
                obs_x, obs_y, tblock_pose, target_block_pose
            )
            
            z_pos = 1.111 + self._obstacle_size
            self._obstacle[i].set_pose(physics, 
                position=(obs_x, obs_y, z_pos), 
                quaternion=(1, 0, 0, 0))
        
        # 多余的障碍物隐藏到地下
        for i in range(len(self.llm_obstacle_config), self.obstacle_num):
            self._obstacle[i].set_pose(physics, position=(0, 0, -10), quaternion=(1, 0, 0, 0))

    def _adjust_obstacle_position(self, obs_x, obs_y, tblock_pose, target_block_pose):
        """
        调整障碍物位置，确保不与 T-block 和目标碰撞
        """
        # 检查与初始 T-block 的碰撞
        if analytic_obs_collision_check(
            Tblock_angle=tblock_pose[-1],
            obs_center=np.array([obs_x, obs_y]) - tblock_pose[:2],
            obs_size=self._obstacle_size * 2,
            threshold=0.04 * 2):
            # 碰撞了，向外移动
            direction = np.array([obs_x, obs_y]) - tblock_pose[:2]
            if np.linalg.norm(direction) > 0.01:
                direction = direction / np.linalg.norm(direction)
                obs_x = tblock_pose[0] + direction[0] * 0.12
                obs_y = tblock_pose[1] + direction[1] * 0.12
        
        # 检查与目标 T 的碰撞
        if analytic_obs_collision_check(
            Tblock_angle=target_block_pose[-1],
            obs_center=np.array([obs_x, obs_y]) - target_block_pose[:2],
            obs_size=self._obstacle_size * 2,
            threshold=0.04 * 2):
            # 碰撞了，向外移动
            direction = np.array([obs_x, obs_y]) - target_block_pose[:2]
            if np.linalg.norm(direction) > 0.01:
                direction = direction / np.linalg.norm(direction)
                obs_x = target_block_pose[0] + direction[0] * 0.12
                obs_y = target_block_pose[1] + direction[1] * 0.12
        
        # 确保在有效范围内
        obs_x = np.clip(obs_x, -0.2, 0.2)
        obs_y = np.clip(obs_y, -0.2, 0.2)
        
        return obs_x, obs_y

    def _generate_random_obstacles(self, physics, tblock_pose, target_block_pose):
        """
        随机生成障碍物（确保不与 T-block 和目标碰撞）
        注意：这个方法仅用于 LLM 模式的后备方案，验证时不使用
        """
        for i in range(self.obstacle_num):
            max_attempts = 100
            for _ in range(max_attempts):
                obstacle_pose = np.array(
                    np.random.uniform(-0.2, 0.2, 2).tolist() + [0]
                )
                
                # 检查与目标 T 的碰撞
                if analytic_obs_collision_check(
                    Tblock_angle=target_block_pose[-1],
                    obs_center=obstacle_pose[:2] - target_block_pose[:2],
                    obs_size=self._obstacle_size * 2,
                    threshold=0.04 * 2):
                    continue
                
                # 检查与初始 T-block 的碰撞
                if analytic_obs_collision_check(
                    Tblock_angle=tblock_pose[-1],
                    obs_center=obstacle_pose[:2] - tblock_pose[:2],
                    obs_size=self._obstacle_size * 2,
                    threshold=0.04 * 2):
                    continue
                
                break
            
            self._obstacle[i].set_pose(physics, 
                position=(obstacle_pose[0], obstacle_pose[1], 1.111 + self._obstacle_size), 
                quaternion=(1, 0, 0, 0))

    def _generate_obstacles_and_tblock_original_style(self, physics, target_block_pose):
        """
        完全复制原版的障碍物和 T-block 生成逻辑（用于验证）
        
        流程：
        1. 先生成所有障碍物（只检查不与目标碰撞）
        2. 根据 obstacle_dist 策略生成 T-block 位置
        3. 检查 T-block 不与障碍物碰撞
        
        Returns:
            tblock_pose: [x, y, theta]
        """
        # Step 1: 先生成所有障碍物
        obstacle_pose_list = []
        for i in range(self.obstacle_num):
            done = False
            while not done:
                obstacle_pose = np.array(
                    np.random.uniform(-0.2, 0.2, 2).tolist() + [np.random.uniform(0, 2*np.pi)]
                )
                
                # 只检查不与目标碰撞
                if analytic_obs_collision_check(
                    Tblock_angle=target_block_pose[-1],
                    obs_center=obstacle_pose[:2] - target_block_pose[:2],
                    obs_size=self._obstacle_size * 2,
                    threshold=0.04 * 2):
                    continue
                
                done = True
            
            # 放置障碍物
            self._obstacle[i].set_pose(physics, 
                position=(obstacle_pose[0], obstacle_pose[1], 1.111 + self._obstacle_size), 
                quaternion=(1, 0, 0, 0))
            obstacle_pose_list.append(obstacle_pose)
        
        # Step 2: 根据 obstacle_dist 策略生成 T-block 位置
        num_between_fail = 0
        done = False
        while not done:
            tblock_pose = np.array(
                np.random.uniform(-0.18, 0.18, 2).tolist() + [np.random.uniform(0, 2*np.pi)]
            )
            
            # 约束1: 不在中心区域
            if abs(tblock_pose[0]) < 0.1 and abs(tblock_pose[1]) < 0.1:
                continue
            
            # 约束2: 初始奖励要足够低
            if self.calc_reward(tblock_pose, target_block_pose) > -3.0:
                continue
            
            # 约束3: 根据 obstacle_dist 策略调整 T-block 位置
            if self.obstacle_dist == 'between':
                # 如果失败次数过多，重新生成最后一个障碍物
                if num_between_fail > 10:
                    obstacle_pose = np.array(
                        np.random.uniform(-0.15, 0.15, 2).tolist() + [np.random.uniform(0, 2*np.pi)]
                    )
                    if analytic_obs_collision_check(
                        Tblock_angle=target_block_pose[-1],
                        obs_center=obstacle_pose[:2] - target_block_pose[:2],
                        obs_size=self._obstacle_size * 2,
                        threshold=0.04 * 2):
                        continue
                    
                    # 更新最后一个障碍物
                    i = self.obstacle_num - 1
                    self._obstacle[i].set_pose(physics, 
                        position=(obstacle_pose[0], obstacle_pose[1], 1.111 + self._obstacle_size), 
                        quaternion=(1, 0, 0, 0))
                    obstacle_pose_list.pop()
                    obstacle_pose_list.append(obstacle_pose)
                    num_between_fail = 0
                
                # 根据最后一个障碍物位置调整 T-block 位置
                obstacle_pose = obstacle_pose_list[-1]
                tblock_pose[:2] = (obstacle_pose[:2]) * (1 + np.random.rand()) + \
                                  np.array([-obstacle_pose[1], obstacle_pose[0]]) / \
                                  np.linalg.norm(obstacle_pose[:2]) * 0.1 * np.random.uniform(-1, 1)
                tblock_pose[:2] = np.clip(tblock_pose[:2], -0.18, 0.18)
            
            elif self.obstacle_dist == 'aroundT':
                # aroundT 策略（如果需要）
                aroundT = True
                for obstacle_pose in obstacle_pose_list:
                    tblock_center = np.zeros(2)
                    tblock_center[0] = tblock_pose[0] + 0.05 * np.sin(tblock_pose[2])
                    tblock_center[1] = tblock_pose[1] - 0.05 * np.cos(tblock_pose[2])
                    
                    if np.linalg.norm(tblock_center[:2] - obstacle_pose[:2]) > 0.17:
                        aroundT = False
                        break
                    elif analytic_obs_collision_check(
                        Tblock_angle=tblock_pose[-1],
                        obs_center=obstacle_pose[:2] - tblock_pose[:2],
                        obs_size=self._obstacle_size * 2,
                        threshold=0.04 * 2):
                        aroundT = False
                        break
                
                if not aroundT:
                    continue
            
            elif self.obstacle_dist in ['random', 'random_step']:
                # random 策略：不调整 T-block 位置
                pass
            
            # 约束4: 检查 T-block 不与所有障碍物碰撞
            between_collision = False
            for obstacle_pose in obstacle_pose_list:
                if analytic_obs_collision_check(
                    Tblock_angle=tblock_pose[-1],
                    obs_center=obstacle_pose[:2] - tblock_pose[:2],
                    obs_size=self._obstacle_size * 2,
                    threshold=0.04 * 2):
                    between_collision = True
                    num_between_fail += 1
                    break
            
            if between_collision:
                continue
            
            done = True
        
        return tblock_pose

    def calc_reward(self, tblock_pose, target_block_pose):
        R_b = np.array([
            [np.cos(tblock_pose[-1]), -np.sin(tblock_pose[-1]), tblock_pose[0]],
            [np.sin(tblock_pose[-1]), np.cos(tblock_pose[-1]), tblock_pose[1]],
            [0, 0, 1]
        ])
        R_g = np.array([
            [np.cos(target_block_pose[-1]), -np.sin(target_block_pose[-1]), target_block_pose[0]],
            [np.sin(target_block_pose[-1]), np.cos(target_block_pose[-1]), target_block_pose[1]],
            [0, 0, 1]
        ])
        p = np.linalg.norm((R_b - R_g)[:2, -1]) / (0.12 / np.pi)
        w = np.arccos(np.clip((np.linalg.inv(R_b) @ R_g)[0, 0], -1, 1))
        return -(w + p)

    def get_reward(self, physics):
        tblock_pos, tblock_mjquat = self._tblock.get_pose(physics)
        target_block_pose, target_block_mjquat = self._target_block.get_pose(physics)
        block_angle = mjquat_to_zyx_euler(tblock_mjquat)[0]
        target_angle = mjquat_to_zyx_euler(target_block_mjquat)[0]

        R_b = np.array([
            [np.cos(block_angle), -np.sin(block_angle), tblock_pos[0]],
            [np.sin(block_angle), np.cos(block_angle), tblock_pos[1]],
            [0, 0, 1]
        ])
        R_g = np.array([
            [np.cos(target_angle), -np.sin(target_angle), target_block_pose[0]],
            [np.sin(target_angle), np.cos(target_angle), target_block_pose[1]],
            [0, 0, 1]
        ])
        p = np.linalg.norm((R_b - R_g)[:2, -1]) / (0.12 / np.pi)
        w = np.arccos(np.clip((np.linalg.inv(R_b) @ R_g)[0, 0], -1, 1))
        return -(w + p)
    
    def set_rod_joint_pos(self, physics, qpos):
        self._rod.configure_joints(physics, qpos)
        self._rod.configure_joints_vel(physics, [0, 0])
    
    def collide_obstacle(self, physics):
        if self.obstacle:
            arr = physics.data.contact.geom
            block_collision = []
            rod_collision = []

            if self.obstacle_shape == 'unstructured':
                obstacle_num = 3
            else:
                obstacle_num = self.obstacle_num

            for i in range(obstacle_num):
                block_collision.append(np.concatenate([arr == np.array([3, 7+i]), arr == np.array([4, 7+i])], axis=0).prod(axis=1).sum() != 0)
                block_collision.append(np.concatenate([arr == np.array([7+i, 3]), arr == np.array([7+i, 4])], axis=0).prod(axis=1).sum() != 0)
                rod_collision.append(np.concatenate([arr == np.array([2, 7+i])], axis=0).prod(axis=1).sum() != 0)
                rod_collision.append(np.concatenate([arr == np.array([7+i, 2])], axis=0).prod(axis=1).sum() != 0)

            return np.array(block_collision).any() or np.array(rod_collision).any()
        return False

    def collide_obstacle_detail(self, physics):
        """返回细粒度碰撞信息: (has_collision, detail_dict)

        detail_dict: {'type': 'rod'/'tblock'/'none', 'obstacle_id': int or -1}
        """
        if not self.obstacle:
            return False, {'type': 'none', 'obstacle_id': -1}

        arr = physics.data.contact.geom

        if self.obstacle_shape == 'unstructured':
            obstacle_num = 3
        else:
            obstacle_num = self.obstacle_num

        for i in range(obstacle_num):
            # T-block 撞障碍物 i
            tblock_hit = (
                np.concatenate([arr == np.array([3, 7+i]), arr == np.array([4, 7+i])], axis=0).prod(axis=1).sum() != 0
                or np.concatenate([arr == np.array([7+i, 3]), arr == np.array([7+i, 4])], axis=0).prod(axis=1).sum() != 0
            )
            if tblock_hit:
                return True, {'type': 'tblock', 'obstacle_id': i}

            # Rod 撞障碍物 i
            rod_hit = (
                np.concatenate([arr == np.array([2, 7+i])], axis=0).prod(axis=1).sum() != 0
                or np.concatenate([arr == np.array([7+i, 2])], axis=0).prod(axis=1).sum() != 0
            )
            if rod_hit:
                return True, {'type': 'rod', 'obstacle_id': i}

        return False, {'type': 'none', 'obstacle_id': -1}
        
    def fallen_block(self, physics):
        tblock_pos, _ = self._tblock.get_pose(physics)
        return tblock_pos[2] < 0.9


class PIDController():
    def __init__(self, KP=1, KI=1, KD=1, timestep=0.002):
        self.kp = KP
        self.ki = KI
        self.kd = KD
        self.timestep = timestep
        self.integral_error = 0
        self.error_last = None
    
    def compute_by_pos(self, currentpos, targetpos):
        error = targetpos - currentpos
        self.integral_error += error * self.timestep
        if self.error_last is None:
            derivative_error = 0
        else:
            derivative_error = (error - self.error_last) / self.timestep
        self.error_last = error
        return self.kp * error + self.ki * self.integral_error + self.kd * derivative_error

    def compute_by_posvel(self, current_state, target_state):
        current_pos = current_state[:2]
        current_vel = current_state[2:]
        target_pos = target_state[:2]
        target_vel = target_state[2:]
        error = target_pos - current_pos
        self.integral_error += error * self.timestep
        derivative_error = target_vel - current_vel
        return self.kp * error + self.ki * self.integral_error + self.kd * derivative_error


class PushT_mj_rod_LLM(composer.Environment):
    """
    支持 LLM 障碍物配置的 Push-T 环境
    
    使用方法:
        env = PushT_mj_rod_LLM(obstacle=True, obstacle_num=2)
        
        # 设置 LLM 生成的障碍物配置
        llm_config = [{'x': 0.1, 'y': 0.1}, {'x': -0.1, 'y': -0.1}]
        env.set_obstacle_config(llm_config)
        
        # 重置环境 (会应用 LLM 配置)
        obs = env.reset()
    """
    
    def __init__(
        self, 
        obstacle=True,
        obstacle_num=1,
        obstacle_size=0.05, 
        obstacle_shape='box',
        obstacle_dist="random",
        eval=False, 
        eval_pose=None,
        motion_pred=False, 
        record_frame=False, 
        NUM_SUBSTEPS=None,
        action_scale=4,
        action_dim=(6,),
        obs_dim=(4,),
        action_reg=True,
        reg_coeff=1.0,
        generate_dataset=False,
        dynamics_randomization=False,
        **kwargs
    ):
        super().__init__(
            task=PushTaskLLM(
                obstacle=obstacle,
                obstacle_num=obstacle_num,
                obstacle_size=obstacle_size,
                obstacle_shape=obstacle_shape,
                obstacle_dist=obstacle_dist,
                eval=eval,
                eval_pose=eval_pose,
                motion_pred=motion_pred,
                NUM_SUBSTEPS=NUM_SUBSTEPS,
                dynamics_randomization=dynamics_randomization
            ),
            **kwargs
        )
        
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        
        if NUM_SUBSTEPS is None and motion_pred:
            NUM_SUBSTEPS = 5
        elif NUM_SUBSTEPS is None:
            NUM_SUBSTEPS = 25

        self.controller = PIDController(400, 0., 30, NUM_SUBSTEPS * 0.002)
        self._record_frame = record_frame
        self.motion_pred = motion_pred
        self.obstacle = obstacle
        self.action_scale = action_scale
        self.len_traj = int(1000 / NUM_SUBSTEPS)
        self.total_time_length = 2.0
        self.action_reg = action_reg
        self.reg_coeff = reg_coeff
        self.eval = eval
        self.generate_dataset = generate_dataset
        self.obstacle_dist = obstacle_dist

        if self.obstacle:
            self.observation_space = spaces.Box(
                low=-np.ones(4 + 2 * obstacle_num),
                high=np.ones(4 + 2 * obstacle_num),
                shape=(4 + 2 * obstacle_num,),
                dtype=np.float64
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.ones(4),
                high=np.ones(4),
                shape=(4,),
                dtype=np.float64
            )
        
        self.action_space = spaces.Box(
            low=-np.ones(2),
            high=np.ones(2),
            shape=(2,),
            dtype=np.float64
        )

    # ===== 新增: LLM 配置接口 =====
    def set_obstacle_config(self, config_list):
        """
        设置障碍物配置 (在下次 reset 时生效)
        
        Args:
            config_list: 障碍物位置列表，格式为 [{'x': float, 'y': float}, ...]
                        x, y 范围应在 [-0.2, 0.2] 内
        
        Example:
            env.set_obstacle_config([
                {'x': 0.1, 'y': 0.1},
                {'x': -0.1, 'y': -0.05}
            ])
        """
        self.task.llm_obstacle_config = config_list
    
    def clear_obstacle_config(self):
        """清除 LLM 配置，恢复随机生成模式"""
        self.task.llm_obstacle_config = None

    def get_obstacle_positions(self):
        """获取当前障碍物位置"""
        positions = []
        for i in range(self.task.obstacle_num):
            pos, _ = self.task._obstacle[i].get_pose(self.physics)
            positions.append({'x': pos[0], 'y': pos[1], 'z': pos[2]})
        return positions
        
    def control_pt2traj(self, action, obs):
        action = action.reshape(1, -1, 2)
        action = disc_cubic_spline_action(
            self.task._desk_size, action, obs,
            self.action_scale, self.len_traj, self.total_time_length
        )
        return action
    
    def step(self, action, motion=None):
        action = action.reshape(1, -1, 2)
        obs = self.prev_obs
        
        action = disc_cubic_spline_action(
            self.task._desk_size, action, obs,
            self.action_scale, self.len_traj, self.total_time_length
        )

        self.frames = []
        physics = self.physics

        self._task.set_rod_joint_pos(physics, action[0, :2])
        super().step([0, 0])
        done = False

        trajectory = []
        info = {'success': False, 'termination': 'timeout', 'collision_detail': None}

        for idx, action_el in enumerate(action):
            raw_state = self._observation_updater.get_observation()
            qqdot = np.concatenate([
                raw_state['unnamed_model/joint_positions'],
                raw_state['unnamed_model/joint_velocities']
            ], axis=1)[0]
            torque = self.controller.compute_by_posvel(qqdot, action_el)
            res = super().step(torque)
            reward = float(res.reward)

            obs = self.get_obs(res)
            trajectory.append(obs)

            if self._record_frame:
                self.frames.append(self.physics.render())

            has_collision, collision_detail = self.task.collide_obstacle_detail(self.physics)
            if has_collision:
                reward = -10
                done = True
                info['termination'] = 'collision'
                info['collision_detail'] = collision_detail
                break

            if float(res.reward) > -0.2:
                info['success'] = True
                info['termination'] = 'success'
                done = True
                if not self.generate_dataset:
                    break

            if self.task.fallen_block(self.physics):
                reward = -10
                done = True
                info['termination'] = 'fall'
                break

            if self.motion_pred and (motion is not None):
                theta = np.arctan2(motion[idx, 3].item(), motion[idx, 2].item())
                self._task._pred_block.set_pose(
                    physics,
                    position=(motion[idx, 0] * self.task._desk_size, motion[idx, 1] * self.task._desk_size, 1.11),
                    quaternion=zyx_euler_to_mjquat([theta, 0, 0])
                )
            
        self.prev_obs = obs
        info['trajectory'] = np.array(trajectory).reshape(-1, obs.shape[-1])
        info['splined_action'] = action[:idx+1, :2].reshape(-1, 2) if idx + 5 > action.shape[0] else action[:idx+5, :2].reshape(-1, 2)
        
        if self.action_reg:
            action_norm = sum(np.linalg.norm(action[:-1, :2] - action[1:, :2], axis=1))
            reward -= self.reg_coeff * action_norm
        
        return obs, reward, done, info
    
    def reset(self, tblock_pos=None, force_tblock_pos=False):
        """
        Reset 环境

        Args:
            tblock_pos: 指定的 T-block 起点位置 [x, y, theta]
            force_tblock_pos: 是否强制使用 tblock_pos（用于 LLM 模式的第二次 reset）
                             如果为 True，即使 eval=False 也会使用 tblock_pos
        """
        self.task.eval = self.eval

        # 设置强制使用的起点（优先级最高）
        if force_tblock_pos and tblock_pos is not None:
            self.task._forced_tblock_pos = tblock_pos
        else:
            self.task._forced_tblock_pos = None

        # 设置/清理 eval_pose：
        # - 传入了 tblock_pos：更新 eval_pose
        # - 训练模式且未传入：清空，避免复用上一次起点污染随机采样
        if tblock_pos is not None:
            self.task.eval_pose = tblock_pos
        elif not self.eval:
            self.task.eval_pose = None

        if force_tblock_pos:
            # LLM 模式：起点和障碍物绑定，不在内部重试
            # ncon 检查由 workspace 层负责，如果不通过则整体重新生成
            res = super().reset()
            self.task._forced_tblock_pos = None  # 清除
        else:
            # 原版模式：内部重试直到无初始接触
            while True:
                res = super().reset()
                if self.physics.data.ncon == 0:
                    break

        obs = self.get_obs(res)
        self.prev_obs = obs
        return obs

    def get_ncon(self):
        """获取当前物理引擎中的接触数量"""
        return self.physics.data.ncon
    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def get_frames(self):
        return self.frames
    
    def set_state(self, state):
        rod_state = np.array([-0.3, -0.3])
        tblock_state = state[0, :2] * self.task._desk_size
        theta = np.arctan2(state[0, 3], state[0, 2])
        self._task.set_rod_joint_pos(self.physics, rod_state[:2])
        self._task._tblock.set_pose(
            self.physics, 
            position=(tblock_state[0], tblock_state[1], 1.11), 
            quaternion=zyx_euler_to_mjquat([theta, 0, 0])
        )
        self._task._tblock.set_velocity(
            self.physics, 
            velocity=np.array([0.0, 0.0, 0.0]), 
            angular_velocity=np.array([0.0, 0.0, 0.0])
        )
        
        if self.obstacle:
            try:
                for i in range(self._task.obstacle_num):
                    obstacle_state = state[0, 4 + 2*i : 4 + 2*(i+1)]
                    self._task._obstacle[i].set_pose(
                        self.physics, 
                        position=(obstacle_state[0], obstacle_state[1], 1.11), 
                        quaternion=zyx_euler_to_mjquat([0, 0, 0])
                    )
                    self._task._obstacle[i].set_velocity(
                        self.physics, 
                        velocity=np.array([0.0, 0.0, 0.0]), 
                        angular_velocity=np.array([0.0, 0.0, 0.0])
                    )
            except:
                pass
                
        res = super().step([0, 0])
        self.prev_obs = self.get_obs(res)

    def obs2state(self, obs):
        return obs[:, :4]
    
    def get_obs(self, res):
        obs = np.concatenate([
            res.observation['tblock_se2'][:, :2] / self.task._desk_size,
            np.cos(res.observation['tblock_se2'][:, -1]).reshape(1, -1),
            np.sin(res.observation['tblock_se2'][:, -1]).reshape(1, -1)
        ], axis=1)
        
        if self.obstacle:
            block_pos = self._observation_updater.get_observation()['obstacle_pos'] / self.task._desk_size
            obs = np.concatenate((obs, block_pos), axis=1)
            
        return obs
    
    def get_info(self):
        return self.action_dim, self.obs_dim
