import gym
from gym import spaces
import numpy as np
from collections import defaultdict, deque
import dill
from pymunk.vec2d import Vec2d
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
import torch
from DIVO.utils.util import cubic_spline_ant_action, cubic_spline_swimmer_action

def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x,axis=0),n,axis=0)

def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype
    )

def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f'Unsupported space type {type(space)}')

def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])

def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result

def aggregate(data, method='max'):
    if method == 'max':
        # equivalent to any
        return np.max(data)
    elif method == 'min':
        # equivalent to all
        return np.min(data)
    elif method == 'mean':
        return np.mean(data)
    elif method == 'sum':
        return np.sum(data)
    elif method == 'last':
        return data[-1]
    else:
        raise NotImplementedError()

def stack_last_n_obs(all_obs, n_steps):
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    result = np.zeros((n_steps,) + all_obs[-1].shape, 
        dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        # pad
        result[:start_idx] = result[start_idx]
    
    if len(result.shape) == 5:
        result = result[0]
    return result


class MultiStepWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            n_obs_steps, 
            n_action_steps, 
            max_steps=None,
            reward_agg_method='last',
            action_dim=[32,],
            obs_dim=[29,],
            spline_action=False,
            is_swimmer=False
        ):
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        if spline_action:
            if is_swimmer:
                self.len_traj = 25
            else:
                self.len_traj = 20
        else:
            self.len_traj = n_action_steps
        self.exclude_current_positions_from_observation = env._exclude_current_positions_from_observation
        self._motion_pred = env._motion_pred
        self.spline_action = spline_action
        self.is_swimmer = is_swimmer

        self.obs = deque(maxlen=n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=n_obs_steps+1))
    
    def reset(self):
        """Resets the environment using kwargs."""
        obs = super().reset()[0]
        self.obs = deque([obs], maxlen=self.n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.terminated = list()
        self.info = defaultdict(lambda : deque(maxlen=self.n_obs_steps+1))
        self.current_step = 0

        self.frames = []
        self.frames.append(super().render())

        obs = self._get_obs(self.n_obs_steps)
        return obs

    def step(self, action, motion=None):
        """
        actions: (n_action_steps,) + action_shape
        """
        action = action.reshape(self.n_action_steps, -1)
        # if motion is not None:
        #     assert len(action) == len(motion)

        if self.spline_action:
            if self.is_swimmer:
                action = cubic_spline_swimmer_action(action, 25)
            else:
                action = cubic_spline_ant_action(action, 20)

        trajectory = []
        self.frames = []
        self.reward = list()
        self.done = list()
        self.terminated = list()
        for idx, act in enumerate(action):
            if len(self.terminated) > 0 and (self.terminated[-1]):
                # termination
                break
            
            if (self._motion_pred) and (motion is not None):
                observation, reward, terminated, done, info = super().step(np.concatenate([act, motion[idx]]))
            else:
                observation, reward, terminated, done, info = super().step(act)


            self.obs.append(observation)
            self.reward.append(reward)
            if (self.max_steps is not None) \
                and (self.current_step >= self.max_steps):
                # truncation
                done = True
                terminated =True
            self.terminated.append(terminated)
            self.done.append(done)
            self._add_info(info)

            self.frames.append(super().render())
            if self.exclude_current_positions_from_observation:
                trajectory.append(np.concatenate((info['xy_position'],observation)))

            else:
                trajectory.append(observation)
            self.current_step += 1

        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method)
        terminated = aggregate(self.terminated, 'max')
        done = aggregate(self.done, 'last')
        info = dict_take_last_n(self.info, self.n_obs_steps)
        info['trajectory'] = np.array(trajectory)
        if self.spline_action:
            info['splined_action'] = action

        return observation, reward, terminated, info

    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.obs) > 0)
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs(
                    [obs[key] for obs in self.obs],
                    n_steps
                )
            return result
        else:
            raise RuntimeError('Unsupported space type')

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)
    
    def get_rewards(self):
        return self.reward
    
    def get_attr(self, name):
        return getattr(self, name)
    
    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result
    
    def obs2state(self, obs):
        if self.env._obstacle:
            return obs[:, :-2*self.env._num_obstacles] # (1, 27)
        else:
            return obs[:, :] # (1, 27)
        # return obs[:, :-2*self.env._num_obstacles]
    
    def get_info(self):
        return self.action_dim, self.obs_dim
    
class EasyMultiStepWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            n_obs_steps, 
            n_control_points,
            max_episode_steps=None,
            reward_agg_method='max',
            len_traj=20,
            action_scale=8,
            action_reg=False,
            reg_coeff=None,
            generate_dataset=False
        ):
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, n_control_points)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_control_points = n_control_points
        self.reward_agg_method = reward_agg_method
        self.len_traj = len_traj
        self.action_scale = action_scale
        self.action_reg = action_reg
        self.reg_coeff = reg_coeff
        self.generate_dataset = generate_dataset

        self.obs = deque(maxlen=n_obs_steps+1)
        self.full_obs = dict()
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=n_obs_steps+1))
        self.eval = False
    
    def reset(self, **kwargs):
        """Resets the environment using kwargs."""
        self.set_eval(self.eval)
        obs = super().reset(**kwargs)

        self.obs = deque([obs], maxlen=self.n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=self.n_obs_steps+1))

        obs = self._get_obs(self.n_obs_steps)
        return obs

    def step(self, action):
        """
        action: cubic_spline_point X 2
        """
        if isinstance(action, dict):
            motion = action['motion']
            action = action['action']
        else:
            motion = None
        action = action.reshape(self.n_control_points, -1)
        t = torch.linspace(0,1,action.shape[0])
        x = torch.from_numpy(action[:, :2])
        coeffs = natural_cubic_spline_coeffs(t, x)
        spline = NaturalCubicSpline(coeffs)
        point = torch.linspace(0, 1, self.len_traj)
        out = spline.evaluate(point)
        cubic_spline_pos = np.zeros((self.len_traj, 2))
        cubic_spline_pos[:,1] = out[:,1]
        cubic_spline_pos[:,0] = out[:,0]

        cubic_spline_pos_ = cubic_spline_pos*self.window_size/self.action_scale + self.block.position
        cubic_spline_pos_ = np.clip(cubic_spline_pos_,0,self.window_size)

        if motion is not None:
            action = np.concatenate([cubic_spline_pos_,motion],axis=1)
        else:
            if action.shape[1] == 6:
                motion = np.repeat(action[0, 2:].reshape(1,-1),self.len_traj,axis=0)
                action = np.concatenate([cubic_spline_pos_,motion],axis=1)
            else:
                action = cubic_spline_pos_

        info['success'] = False
        for idx in range(self.len_traj):
            if idx == 0:
                self.agent.position = Vec2d(*action[0])
                self.agent.velocity = Vec2d(0,0)
                self.env.space.step(0.01)
                observation, reward, done, info_ = super().step(action[0])
                self.full_obs = list()
            else:
                observation, reward, done, info_ = super().step(action[idx])

            self.obs.append(observation)
            self.full_obs.append(observation)
            self.reward.append(reward)
            if (self.max_episode_steps is not None) \
                and (len(self.reward)/self.len_traj >= self.max_episode_steps):
                # truncation
                done = True
            self.done.append(done)
            if info_['success']:
                info['success'] = True

            if not self.generate_dataset:
                if len(self.done) > 0 and self.done[-1]:
                    # termination
                    break
            else:
                if len(self.done) > 0 and self.done[-1] and reward < -9.9:
                    break

        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method)
        if self.action_reg:
            action_norm = sum(np.linalg.norm(cubic_spline_pos[:-1,] - cubic_spline_pos[1:,], axis=1))
            reward -= self.reg_coeff * action_norm
        done = aggregate(self.done, 'max')
        info = dict_take_last_n(self.info, self.n_obs_steps)
        self.full_obs = np.array(self.full_obs)
        info['trajectory'] = self.full_obs
        info['splined_action'] = action[:,:2].reshape(-1,2)

        return observation, reward, done, info

    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.obs) > 0)
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs(
                    [obs[key] for obs in self.obs],
                    n_steps
                )
            return result
        else:
            raise RuntimeError('Unsupported space type')

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)

    def _add_obs(self, obs):
        self.full_obs.append(obs)
    
    def get_rewards(self):
        return self.reward
    
    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)
    
    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result
