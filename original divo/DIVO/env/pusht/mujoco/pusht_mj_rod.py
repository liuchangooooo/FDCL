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
    """Add simple observable features for joint angles and velocities."""
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
            obstacle_body.add('geom',
                                name='obstacle_1',
                                type='box',
                                size=[obstacle_size, obstacle_size/4, 0.01],
                                pos = [0, obstacle_size*3/4, 0.01],
                                rgba=[1, 0, 0, 1],
                                mass=100,
                                contype=2,
                                conaffinity=3)
            obstacle_body.add('geom',
                                name='obstacle_2',
                                type='box',
                                size=[obstacle_size/4, obstacle_size/2, 0.01],
                                pos = [obstacle_size*3/4, 0, 0.01],
                                rgba=[1, 0, 0, 1],
                                mass=100,
                                contype=2,
                                conaffinity=3)
            obstacle_body.add('geom',
                                name='obstacle_3',
                                type='box',
                                size=[obstacle_size, obstacle_size/4, 0.01],
                                pos = [0, -obstacle_size*3/4, 0.01],
                                rgba=[1, 0, 0, 1],
                                mass=100,
                                contype=2,
                                conaffinity=3)

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

class PushTask(composer.Task):

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
                 dynamics_randomization=False,):
        
        if NUM_SUBSTEPS==None and motion_pred:
            NUM_SUBSTEPS = 5 # The number of physics substeps per control timestep.
        elif NUM_SUBSTEPS==None:
            NUM_SUBSTEPS = 25  
        else:
            NUM_SUBSTEPS = NUM_SUBSTEPS
            
        self.obstacle = obstacle
        self.obstacle_num = obstacle_num
        self.obstacle_shape = obstacle_shape
        self.obstacle_dist = obstacle_dist
        self.eval = eval
        self.eval_pose = eval_pose
        self.motion_pred = motion_pred
        self.dynamics_randomization = dynamics_randomization
        self._arena = floors.Floor()

        # Set visualization
        self._arena.mjcf_model.statistic.set_attributes(extent = 1.0, center=[0.0, 0, 0.9])
        self._arena.mjcf_model.visual.get_children('global').set_attributes(azimuth=90, elevation=-90, fovy=25)
        self._arena.mjcf_model.visual.get_children('headlight').set_attributes(diffuse=[0.6, 0.6, 0.6], ambient=[0.3, 0.3, 0.3], specular=[0.5, 0.5 ,0.5])
        
        light_pos_lst = [[0, 0, 1.5]]
        
        for light_pos in light_pos_lst:
            self._arena.mjcf_model.worldbody.add('light', pos=light_pos, dir=[0, 0, -1], directional=True)
        
        # Set fixed objects(robot base, desk)
        self._desk_size = 0.25
        desk_geom = self._arena.mjcf_model.worldbody.add('geom', name='desk', type='box', size=[self._desk_size, self._desk_size, 0.02/2], pos=[0.0, 0, 1.1], rgba=[250.0/255, 198.0/255, 122.0/255, 1], friction=[lateral_friction, 0.005, 0.0001], conaffinity=3)

        # Set Entities
        rod_length = 0.1
        self._rod = Rod(rod_length)
        self._rod_initial_pose = (0, 0, 1.1 + rod_length + 0.02/2)
        self._arena.attach(self._rod, self._arena.mjcf_model.worldbody.add('site', size=[1e-6]*3, pos=self._rod_initial_pose))

        self._tblock = TBlock(size=0.01, contact_dict={'contype': 1, 'conaffinity':2})
        self._target_block = TBlock(size=0.01, rgba=[0.17, 0.62, 0.17, 1], contact_dict={'contype': 0, 'conaffinity':2})

        # Configure initial poses of Entities
        self._arena.add_free_entity(self._tblock)        
        self._arena.attach(self._target_block)

        self._tblock_pos = UniformTBlockSE2(x=[-0.15, 0.15], y=[-0.15, 0.15])
        self._target_block_pos = (0,0,-np.pi/4)
        
        # Configure variators
        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        # Configure and enable observables
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
            self._obstacle_size = obstacle_size # np.round(np.random.uniform(0.04,0.06),3)

            self._obstacle = []
            for i in range(self.obstacle_num):
                self._obstacle.append(Obstacle(obstacle_size=self._obstacle_size,
                                    obstacle_shape=self.obstacle_shape))
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
            self._pred_block = TBlock(size=0.04, rgba=[1.0, 0.01, 0.01, 0.3], contact_dict={'contype': 4, 'conaffinity':4})        
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
        self._physics_variator.apply_variations(physics, random_state)
        target_block_pose = variation.evaluate((self._target_block_pos), random_state=random_state)
        self._target_block.set_pose(physics, position=(target_block_pose[0], target_block_pose[1], 1.11-0.01*1.99), quaternion=zyx_euler_to_mjquat([target_block_pose[2], 0, 0]))

        if not self.obstacle:
            if not self.eval:
                done = False
                while not done:
                    tblock_pose = np.array(np.random.uniform(-0.18, 0.18, 2).tolist() + [np.random.uniform(0, 2*np.pi)])
                    if abs(tblock_pose[0])<0.1 and abs(tblock_pose[1])<0.1:
                        continue
                    if self.calc_reward(tblock_pose, target_block_pose) > -3.0:
                        continue
                    done = True
                self._tblock.set_pose(physics, position=(tblock_pose[0], tblock_pose[1], 1.111), quaternion=zyx_euler_to_mjquat([tblock_pose[2], 0, 0]))
                
            else:
                if self.eval_pose is not None:
                    tblock_pose = np.array(self.eval_pose)
                else:
                    tblock_pose = np.array([0.15,0.15,3*np.pi/4])
                self._tblock.set_pose(physics, position=(tblock_pose[0], tblock_pose[1], 1.111), quaternion=zyx_euler_to_mjquat([tblock_pose[2], 0, 0]))
                
        else:
            if not self.eval:
                obstacle_pose_list = []
                for i in range(self.obstacle_num):
                    done = False
                    while not done:
                        obstacle_pose = np.array(np.random.uniform(-0.2, 0.2, 2).tolist() + [np.random.uniform(0, 2*np.pi)])

                        if analytic_obs_collision_check(Tblock_angle=target_block_pose[-1],
                                                            obs_center=obstacle_pose[:2]-target_block_pose[:2],
                                                            obs_size=self._obstacle_size*2,
                                                            threshold=0.04*2):
                            continue
                        
                        done = True
                    self._obstacle[i].set_pose(physics, position=(obstacle_pose[0], obstacle_pose[1], 1.111+self._obstacle_size), quaternion=(1, 0, 0, 0))
                    num_between_fail = 0
                    obstacle_pose_list.append(obstacle_pose)
                done = False
                while not done:
                    tblock_pose = np.array(np.random.uniform(-0.18, 0.18, 2).tolist() + [np.random.uniform(0, 2*np.pi)])
                    if abs(tblock_pose[0])<0.1 and abs(tblock_pose[1])<0.1:
                        continue
                    if self.calc_reward(tblock_pose, target_block_pose) > -3.0:
                        continue
                    if self.obstacle_dist == 'aroundT':
                        for obstacle_pose in obstacle_pose_list:
                            tblock_center = np.zeros(2)
                            tblock_center[0] = tblock_pose[0]+0.05*np.sin(tblock_pose[2])
                            tblock_center[1] = tblock_pose[1]-0.05*np.cos(tblock_pose[2])

                            if np.linalg.norm(tblock_center[:2]-obstacle_pose[:2]) > 0.17:
                                aroundT = False
                                break
                            elif analytic_obs_collision_check(Tblock_angle=tblock_pose[-1],
                                                                  obs_center=obstacle_pose[:2]-tblock_pose[:2],
                                                                  obs_size=self._obstacle_size*2,
                                                                  threshold=0.04*2):
                                aroundT = False
                                break
                            else:
                                aroundT = True
                        if not aroundT:
                            continue
                    elif (self.obstacle_dist == 'random') or (self.obstacle_dist == 'random_step'):
                        pass
                    elif self.obstacle_dist == 'between':
                        if num_between_fail > 10:
                            obstacle_pose = np.array(np.random.uniform(-0.15, 0.15, 2).tolist() + [np.random.uniform(0, 2*np.pi)])
                            if analytic_obs_collision_check(Tblock_angle=target_block_pose[-1],
                                                                obs_center=obstacle_pose[:2]-target_block_pose[:2],
                                                                obs_size=self._obstacle_size*2,
                                                                threshold=0.04*2):
                                continue
                            self._obstacle[i].set_pose(physics, position=(obstacle_pose[0], obstacle_pose[1], 1.111+self._obstacle_size), quaternion=(1, 0, 0, 0))
                            obstacle_pose_list.pop()
                            obstacle_pose_list.append(obstacle_pose)
                            num_between_fail = 0
                        tblock_pose[:2] = (obstacle_pose[:2])*(1+np.random.rand()) + np.array([-obstacle_pose[1],obstacle_pose[0]])/np.linalg.norm(obstacle_pose[:2])*0.1*np.random.uniform(-1,1)
                        tblock_pose[:2] = np.clip(tblock_pose[:2],-0.18,0.18)

                    for obstacle_pose in obstacle_pose_list:
                        if analytic_obs_collision_check(Tblock_angle=tblock_pose[-1],
                                                            obs_center=obstacle_pose[:2]-tblock_pose[:2],
                                                            obs_size=self._obstacle_size*2,
                                                            threshold=0.04*2):
                            between_collision = True
                            num_between_fail += 1
                            break
                        else:
                            between_collision = False

                    if between_collision:
                        continue
                    
                    done = True
                self._tblock.set_pose(physics, position=(tblock_pose[0], tblock_pose[1], 1.111), quaternion=zyx_euler_to_mjquat([tblock_pose[2], 0, 0]))
            else:
                if self.eval_pose is not None:
                    tblock_pose = np.array(self.eval_pose)
                else:
                    done = False
                    while not done:
                        tblock_pose = np.array(np.random.uniform(-0.18, 0.18, 2).tolist() + [np.random.uniform(0, 2*np.pi)])
                        if abs(tblock_pose[0])<0.1 and abs(tblock_pose[1])<0.1:
                            continue
                        else:
                            done = True
                    tblock_pose = np.array([0.15,0.15,3*np.pi/4])
                self._tblock.set_pose(physics, position=(tblock_pose[0], tblock_pose[1], 1.111), quaternion=zyx_euler_to_mjquat([tblock_pose[2], 0, 0]))

                for i in range(self.obstacle_num):
                    done = False
                    while not done:
                        obstacle_pose = np.array(np.random.uniform(-0.2, 0.2, 2).tolist() + [np.random.uniform(0, 2*np.pi)])

                        if analytic_obs_collision_check(Tblock_angle=target_block_pose[-1],
                                                            obs_center=obstacle_pose[:2]-target_block_pose[:2],
                                                            obs_size=self._obstacle_size*2,
                                                            threshold=0.04*2):
                            continue

                        if analytic_obs_collision_check(Tblock_angle=tblock_pose[-1],
                                                            obs_center=obstacle_pose[:2]-tblock_pose[:2],
                                                            obs_size=self._obstacle_size*2,
                                                            threshold=0.02*2):
                            continue
                        
                        if self.obstacle_dist == 'aroundT':
                            tblock_center = np.zeros(2)
                            tblock_center[0] = tblock_pose[0]+0.05*np.sin(tblock_pose[2])
                            tblock_center[1] = tblock_pose[1]-0.05*np.cos(tblock_pose[2])

                            if np.linalg.norm(tblock_center[:2]-obstacle_pose[:2]) > 0.17:
                                continue
                        elif (self.obstacle_dist == 'random') or (self.obstacle_dist == 'random_step'):
                            pass
                        elif self.obstacle_dist == 'between':
                            obstacle_pose[:2] = (tblock_pose[:2])*(1-np.random.rand()) + np.array([-tblock_pose[1,],tblock_pose[0]])/np.linalg.norm(tblock_pose[:2])*0.1*np.random.uniform(-1,1)
                        
                            if analytic_obs_collision_check(Tblock_angle=tblock_pose[-1],
                                                                obs_center=obstacle_pose[:2]-tblock_pose[:2],
                                                                obs_size=self._obstacle_size*2,
                                                                threshold=0.04*2):
                                continue
                            if analytic_obs_collision_check(Tblock_angle=target_block_pose[-1],
                                                                obs_center=obstacle_pose[:2]-target_block_pose[:2],
                                                                obs_size=self._obstacle_size*2,
                                                                threshold=0.04*2):
                                continue

                        done = True
                    if self.obstacle_shape == 'unstructured':
                        self._obstacle[i].set_pose(physics, position=(obstacle_pose[0], obstacle_pose[1], 1.111+self._obstacle_size), quaternion=zyx_euler_to_mjquat([obstacle_pose[-1], 0, 0]))
                    else:
                        self._obstacle[i].set_pose(physics, position=(obstacle_pose[0], obstacle_pose[1], 1.111+self._obstacle_size), quaternion=(1, 0, 0, 0))
        if self.motion_pred:
            self._pred_block.set_pose(physics, position=(tblock_pose[0], tblock_pose[1], 1.111), quaternion=zyx_euler_to_mjquat([tblock_pose[2], 0, 0]))
    
    def calc_reward(self, tblock_pose, target_block_pose):
        R_b = np.array([[np.cos(tblock_pose[-1]), -np.sin(tblock_pose[-1]), tblock_pose[0]],
                        [np.sin(tblock_pose[-1]), np.cos(tblock_pose[-1]), tblock_pose[1]],
                        [0, 0, 1]])
        R_g = np.array([[np.cos(target_block_pose[-1]), -np.sin(target_block_pose[-1]), target_block_pose[0]],
                        [np.sin(target_block_pose[-1]), np.cos(target_block_pose[-1]), target_block_pose[1]],
                        [0, 0, 1]])
        p = np.linalg.norm((R_b - R_g)[:2, -1])/(0.12/np.pi)
        w = np.arccos((np.linalg.inv(R_b)@R_g)[0,0])
        return -(w + p)

    def get_reward(self, physics):
        tblock_pos, tblock_mjquat = self._tblock.get_pose(physics)
        target_block_pose, target_block_mjquat = self._target_block.get_pose(physics)
        block_angle = mjquat_to_zyx_euler(tblock_mjquat)[0]
        target_angle = mjquat_to_zyx_euler(target_block_mjquat)[0]

        R_b = np.array([[np.cos(block_angle), -np.sin(block_angle), tblock_pos[0]],
                        [np.sin(block_angle), np.cos(block_angle), tblock_pos[1]],
                        [0, 0, 1]])
        R_g = np.array([[np.cos(target_angle), -np.sin(target_angle), target_block_pose[0]],
                        [np.sin(target_angle), np.cos(target_angle), target_block_pose[1]],
                        [0, 0, 1]])
        p = np.linalg.norm((R_b - R_g)[:2, -1])/(0.12/np.pi)
        w = np.arccos((np.linalg.inv(R_b)@R_g)[0,0])
        
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
                block_collision.append(np.concatenate([arr == np.array([7+i,3]), arr == np.array([7+i,4])], axis=0).prod(axis=1).sum() != 0)
                rod_collision.append(np.concatenate([arr == np.array([2, 7+i])], axis=0).prod(axis=1).sum() != 0)
                rod_collision.append(np.concatenate([arr == np.array([7+i,2])], axis=0).prod(axis=1).sum() != 0)

            block_collision = np.array(block_collision).any()
            rod_collision = np.array(rod_collision).any()

            return block_collision or rod_collision
        else:
            return False
        
    def fallen_block(self, physics):
        tblock_pos, tblock_mjquat = self._tblock.get_pose(physics)

        return tblock_pos[2] < 0.9
    
    def set_random_obs_pose(self, physics, obs):
        tblock_pose = np.zeros(3)
        tblock_pose[:2] = obs[0,:2]*self._desk_size
        tblock_pose[2] = np.arctan2(obs[0,3], obs[0,2])
        target_block_pose = np.array(self._target_block_pos)
        obstacle_pose_list = []
        for i in range(self.obstacle_num):
            done = False
            num_trial = 0
            while not done:
                obstacle_pose = np.array(np.random.uniform(-0.1, 0.1, 2).tolist() + [np.random.uniform(0, 2*np.pi)])
                obstacle_pose[:2] = obstacle_pose[:2] + tblock_pose[:2]
                obstacle_pose[:2] = np.clip(obstacle_pose[:2],-0.2,0.2)

                close = False
                if len(obstacle_pose_list) > 0:
                    for obstacle_pose_ in obstacle_pose_list:
                        if np.linalg.norm(obstacle_pose_[:2]-obstacle_pose[:2]) < 0.1:
                            close = True
                            break
                    if close:
                        num_trial += 1
                        if num_trial > 1000:
                            break
                        continue
                if analytic_obs_collision_check(Tblock_angle=tblock_pose[-1],
                                                    obs_center=obstacle_pose[:2]-tblock_pose[:2],
                                                    obs_size=self._obstacle_size*2,
                                                    threshold=0.02*2):
                    num_trial += 1

                    if num_trial > 1000:
                        break
                    
                    continue
                
                if analytic_obs_collision_check(Tblock_angle=target_block_pose[-1],
                                                    obs_center=obstacle_pose[:2]-target_block_pose[:2],
                                                    obs_size=self._obstacle_size*2,
                                                    threshold=0.01*2):
                    continue
                

                done = True
                
            self._obstacle[i].set_pose(physics, position=(obstacle_pose[0], obstacle_pose[1], 1.111+self._obstacle_size), quaternion=(1, 0, 0, 0))
            obstacle_pose_list.append(obstacle_pose[:2])
        
        return np.array(obstacle_pose_list).reshape(-1)

class PIDController():
    def __init__(self, KP=1, KI=1, KD=1, timestep=0.002) -> None:
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
            derivative_error = (error - self.error_last)/self.timestep
        self.error_last = error
        
        return self.kp*error + self.ki*self.integral_error + self.kd*derivative_error

    def compute_by_posvel(self, current_state, target_state):
        current_pos = current_state[:2]; current_vel = current_state[2:]
        target_pos = target_state[:2]; target_vel = target_state[2:]
        
        error = target_pos - current_pos
        self.integral_error += error * self.timestep
        derivative_error = target_vel - current_vel
        
        return self.kp*error + self.ki*self.integral_error + self.kd*derivative_error

class PushT_mj_rod(composer.Environment):
    def __init__(
        self, 
        obstacle=True,
        obstacle_num=1,
        obstacle_size=0.01, 
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
    ): # Add PID control parameters here
        super().__init__(task=PushTask(obstacle=obstacle,
                                       obstacle_num=obstacle_num,
                                       obstacle_size=obstacle_size,
                                       obstacle_shape=obstacle_shape,
                                       obstacle_dist=obstacle_dist,
                                       eval=eval,
                                       eval_pose=eval_pose,
                                       motion_pred=motion_pred,
                                       NUM_SUBSTEPS=NUM_SUBSTEPS,
                                       dynamics_randomization=dynamics_randomization),
                                       **kwargs)
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        if NUM_SUBSTEPS==None and motion_pred:
            NUM_SUBSTEPS = 5
        elif NUM_SUBSTEPS==None:
            NUM_SUBSTEPS = 25

        controller= PIDController(400, 0., 30, NUM_SUBSTEPS*0.002)

        self.controller = controller
        self._record_frame = record_frame
        self.motion_pred = motion_pred
        self.obstacle = obstacle
        self.action_scale = action_scale
        self.len_traj = int(1000/NUM_SUBSTEPS)
        self.total_time_length = 2.0
        self.action_reg = action_reg
        self.reg_coeff = reg_coeff
        self.eval = eval
        self.generate_dataset = generate_dataset
        self.obstacle_dist = obstacle_dist

        if self.obstacle:
            self.observation_space = spaces.Box(
            low=-np.ones(4+2*obstacle_num),
            high=np.ones(4+2*obstacle_num),
            shape=(4+2*obstacle_num,),
            dtype=np.float64)

        else:
            self.observation_space = spaces.Box(
            low=-np.ones(4),
            high=np.ones(4),
            shape=(4,),
            dtype=np.float64
            )
        # positional goal for agent
        self.action_space = spaces.Box(
            low=-np.ones(2),
            high=np.ones(2),
            shape=(2,),
            dtype=np.float64
        )
        
    def control_pt2traj(self, action, obs):
        action = action.reshape(1,-1,2)
        
        action = disc_cubic_spline_action(self.task._desk_size,
                                          action,
                                          obs,
                                          self.action_scale,
                                          self.len_traj,
                                          self.total_time_length
                                          )
        
        return action
    
    def step(self, action, motion=None):
        action = action.reshape(1,-1,2)
        obs = self.prev_obs
        
        action = disc_cubic_spline_action(self.task._desk_size,
                                          action,
                                          obs,
                                          self.action_scale,
                                          self.len_traj,
                                          self.total_time_length
                                          )

        self.frames = []
        
        physics = self.physics
        
        self._task.set_rod_joint_pos(physics, action[0, :2])
        super().step([0, 0])
        done = False
        
        trajectory = []
        info = {}
        info['success'] = False

        for idx, action_el in enumerate(action):
            raw_state = self._observation_updater.get_observation()
            qqdot = np.concatenate([raw_state['unnamed_model/joint_positions'],
                                    raw_state['unnamed_model/joint_velocities']],
                                    axis=1)[0]
            torque = self.controller.compute_by_posvel(qqdot, action_el)
            res = super().step(torque)
            reward = float(res.reward)
            
            obs = self.get_obs(res)
            trajectory.append(obs)
            if self._record_frame:
                self.frames.append(self.physics.render())
            
            if self.task.collide_obstacle(self.physics):
                reward = -10
                done = True
                break                
            if float(res.reward) > -0.2:
                info['success'] = True
                done = True
                if not self.generate_dataset:
                    break

            if self.task.fallen_block(self.physics):
                reward = -10
                done = True
                break

            if self.motion_pred and (motion is not None):
                theta = np.arctan2(motion[idx,3].item(), motion[idx,2].item())
                self._task._pred_block.set_pose(physics,
                                                position=(motion[idx, 0]*self.task._desk_size, motion[idx, 1]*self.task._desk_size, 1.11),
                                                quaternion=zyx_euler_to_mjquat([theta, 0, 0])
                                                )
            
        self.prev_obs = obs
        info['trajectory'] = np.array(trajectory).reshape(-1,obs.shape[-1])

        info['splined_action'] = action[:idx+1,:2].reshape(-1,2)
        if idx + 5 > action.shape[0]:
            info['splined_action'] = action[:idx+1,:2].reshape(-1,2)
        else:
            info['splined_action'] = action[:idx+5,:2].reshape(-1,2)
        
        if self.action_reg:
            action_norm = sum(np.linalg.norm(action[:-1,:2] - action[1:,:2], axis=1))
            reward -= self.reg_coeff * action_norm
        
        return obs, reward, done, info
    
    def reset(self, tblock_pos = None):    
        self.task.eval = self.eval
        
        if tblock_pos is not None:
            self.task.eval_pose = tblock_pos
        while True:
            res = super().reset()
            if self.physics.data.ncon == 0:
                break
            
        obs = self.get_obs(res)
        self.prev_obs = obs
        return obs
    
    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def get_frames(self):
        return self.frames
    
    def set_state(self, state):
        rod_state = np.array([-0.3,-0.3])
        tblock_state = state[0, :2]*self.task._desk_size
        theta = np.arctan2(state[0, 3], state[0, 2])
        self._task.set_rod_joint_pos(self.physics, rod_state[:2])    
        self._task._tblock.set_pose(self.physics, position=(tblock_state[0], tblock_state[1], 1.11), quaternion=zyx_euler_to_mjquat([theta, 0, 0]))
        self._task._tblock.set_velocity(self.physics, velocity=np.array([0.0, 0.0, 0.0]), angular_velocity=np.array([0.0, 0.0, 0.0]))
        
        if self.obstacle:
            try:
                for i in range(self._task.obstacle_num):
                    obstacle_state = state[0, 4 + 2*i:4 + 2*(i+1)]
                    
                    self._task._obstacle[i].set_pose(self.physics, position=(obstacle_state[0], obstacle_state[1], 1.11), quaternion=zyx_euler_to_mjquat([0, 0, 0]))
                    self._task._obstacle[i].set_velocity(self.physics, velocity=np.array([0.0, 0.0, 0.0]), angular_velocity=np.array([0.0, 0.0, 0.0]))
            except:
                pass
                
        res = super().step([0, 0])
            
        self.prev_obs = self.get_obs(res)

    def obs2state(self, obs):
        return obs[:, :4]
    
    def get_obs(self, res):
        obs = np.concatenate([
            res.observation['tblock_se2'][:,:2]/self.task._desk_size,
            np.cos(res.observation['tblock_se2'][:,-1]).reshape(1,-1),
            np.sin(res.observation['tblock_se2'][:,-1]).reshape(1,-1)
            ],
            axis=1
        )
        if self.obstacle:
            block_pos = self._observation_updater.get_observation()['obstacle_pos']/self.task._desk_size
            obs = np.concatenate((obs,block_pos), axis=1)
            
        return obs
    
    def get_info(self):
        return self.action_dim, self.obs_dim