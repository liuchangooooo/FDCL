import torch
from typing import NamedTuple
import torch.nn as nn
import numpy as np
from DIVO.common.pytorch_util import dict_apply
import tqdm
import pathlib
import wandb
import wandb.sdk.data_types.video as wv

class ReplayBufferSamples(NamedTuple):
    observations: np.array
    next_observations: np.array
    actions: np.array
    rewards: np.array
    dones: np.array

class DictReplayBuffer:
    def __init__(self, size, action_dim, obs_entry_info):
        self.obs_buf = {}
        self.next_obs_buf = {}
        self.non_masked_obs_buf = {}
        for e in obs_entry_info:
            self.obs_buf[e[0]] = np.zeros((size, *e[1]), dtype=np.float32)
        for e in obs_entry_info:
            self.next_obs_buf[e[0]] = np.zeros((size, *e[1]), dtype=np.float32)

        self.act_buf = np.zeros((size, *action_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def add(self, obs, next_obs, act, rew, done):
        if isinstance(rew, np.ndarray):
            n_env = int(rew.shape[0])
        else:
            n_env = 1

        for k in list(self.obs_buf.keys()):
            self.obs_buf[k][self.ptr:self.ptr+n_env] = obs[k]

        for k in list(self.next_obs_buf.keys()):
            self.next_obs_buf[k][self.ptr:self.ptr+n_env] = next_obs[k]

        self.act_buf[self.ptr:self.ptr+n_env] = act

        if n_env == 1:
            self.rew_buf[self.ptr] = rew
            self.done_buf[self.ptr] = done
        else:
            self.rew_buf[self.ptr:self.ptr+n_env] = rew.reshape(-1,1)
            self.done_buf[self.ptr:self.ptr+n_env] = done.reshape(-1,1)

        self.ptr = (self.ptr + n_env) % self.max_size
        self.size = min(self.size + n_env, self.max_size)

    def sample(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        observations = {k: self.obs_buf[k][idxs] for k in self.obs_buf.keys()}
        next_observations = {k: self.next_obs_buf[k][idxs] for k in self.next_obs_buf.keys()}
        batch = AttrDict(observations=observations,
                         next_observations=next_observations,
                         actions=self.act_buf[idxs],
                         rewards=self.rew_buf[idxs],
                         dones=self.done_buf[idxs])
        return batch
    
def ccw(A,B,C):
    return ((C[...,1]-A[...,1]) * (B[...,0]-A[...,0]) > (B[...,1]-A[...,1]) * (C[...,0]-A[...,0])).cpu().detach().numpy()

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ((ccw(A,C,D) != ccw(B,C,D)) * (ccw(A,B,C) != ccw(A,B,D))).sum()

# Source-tag encoding for skill rollouts (Task 3 / Requirement 4.9).
SOURCE_NORMAL = 0
SOURCE_DIVERSITY_PAIRED = 1
_SOURCE_STR_TO_INT = {"normal": SOURCE_NORMAL, "diversity_paired": SOURCE_DIVERSITY_PAIRED}


def _encode_source(source):
    if source is None:
        return SOURCE_NORMAL
    if isinstance(source, str):
        try:
            return _SOURCE_STR_TO_INT[source]
        except KeyError:
            raise ValueError(f"Unknown replay source tag: {source}")
    return int(source)


def reward_for_critic(skill_id, reward_task, reward_total, deploy_skill_id=0):
    """Route the critic training reward by skill (Task 4 / Requirement 4.6).

    Deployment skill ``w_0`` (``skill_id == deploy_skill_id``) is trained on the
    pure task reward so ``Q(obs, w_0, .)`` stays a clean task value; probe skills
    use ``reward_total = reward_task + beta_div * reward_div``.

    Accepts numpy arrays (shape ``[B, 1]`` or ``[B]``) or torch tensors and
    returns the same type/shape.
    """
    if isinstance(skill_id, torch.Tensor):
        mask = (skill_id == int(deploy_skill_id))
        return torch.where(mask, reward_task, reward_total)
    skill_id = np.asarray(skill_id)
    return np.where(skill_id == int(deploy_skill_id), np.asarray(reward_task), np.asarray(reward_total))


class StateDictReplayBuffer:
    def __init__(self, size, obs_dim, action_dim, z_dim=None, full_obs=None,
                 obs_entry_info=None, track_skill=False):
        self.z = False
        self.full_obs = False
        # Skill bookkeeping (Task 3): per-transition skill_id, source tag, and the
        # three reward columns (task / div / total). Off by default so strict-DIVO
        # training and existing callers are unchanged.
        self.track_skill = bool(track_skill)
        if obs_entry_info == None:
            self.obs_dict = False
            self.obs_buf = np.zeros((size, *obs_dim), dtype=np.float32)
            self.next_obs_buf = np.zeros((size, *obs_dim), dtype=np.float32)
        else:
            self.obs_dict = True
            self.obs_buf = {}
            self.next_obs_buf = {}
            for e in obs_entry_info:
                self.obs_buf[e[0]] = np.zeros((size, *e[1:]), dtype=np.float32)
            for e in obs_entry_info:
                self.next_obs_buf[e[0]] = np.zeros((size, *e[1:]), dtype=np.float32)

        self.act_buf = np.zeros((size, *action_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        if z_dim != None:
            self.z_buf = np.zeros((size,*z_dim), dtype=np.float32)
            self.z = True
        if full_obs != None:
            self.full_obs_buf = np.zeros((size,*full_obs), dtype=np.float32)
            self.full_obs = True
        if self.track_skill:
            self.skill_id_buf = np.zeros((size, 1), dtype=np.int64)
            self.source_buf = np.zeros((size, 1), dtype=np.int64)
            self.reward_task_buf = np.zeros((size, 1), dtype=np.float32)
            self.reward_div_buf = np.zeros((size, 1), dtype=np.float32)
            self.reward_total_buf = np.zeros((size, 1), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def add(self, obs, next_obs, act, rew, done, z=None, full_obs=None,
            skill_id=None, source=None, reward_task=None, reward_div=None,
            reward_total=None):
        if ((not self.z) and isinstance(z,(np.ndarray, np.generic))) or (self.z and (not isinstance(z,(np.ndarray, np.generic)))):
            raise ValueError('Replaybuffer latent error')

        if not self.obs_dict:
            self.obs_buf[self.ptr:self.ptr+1] = obs
            self.next_obs_buf[self.ptr:self.ptr+1] = next_obs
        else:
            for k in list(self.obs_buf.keys()):
                self.obs_buf[k][self.ptr:self.ptr+1] = obs[k]
            for k in list(self.next_obs_buf.keys()):
                self.next_obs_buf[k][self.ptr:self.ptr+1] = next_obs[k]

        self.act_buf[self.ptr:self.ptr+1] = act

        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        if self.z:
            self.z_buf[self.ptr:self.ptr+1] = z
        
        if self.full_obs:
            self.full_obs_buf[self.ptr:self.ptr+1] = full_obs

        if self.track_skill:
            # skill_id defaults to 0 (deployment skill w_0).
            self.skill_id_buf[self.ptr] = 0 if skill_id is None else int(skill_id)
            self.source_buf[self.ptr] = _encode_source(source)
            # reward_task defaults to the plain reward; reward_div defaults to 0;
            # reward_total defaults to reward_task + reward_div. w_0 transitions
            # carry reward_div=0 so reward_total == reward_task (design 组件 4).
            r_task = float(rew) if reward_task is None else float(reward_task)
            r_div = 0.0 if reward_div is None else float(reward_div)
            r_total = (r_task + r_div) if reward_total is None else float(reward_total)
            self.reward_task_buf[self.ptr] = r_task
            self.reward_div_buf[self.ptr] = r_div
            self.reward_total_buf[self.ptr] = r_total

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=32, idxs=None):
        if idxs is None:
            idxs = np.random.randint(0, self.size, size=batch_size)

        if not self.obs_dict:
            observations = self.obs_buf[idxs]
            next_observations = self.next_obs_buf[idxs]
        else:
            observations = {k: self.obs_buf[k][idxs] for k in self.obs_buf.keys()}
            next_observations = {k: self.next_obs_buf[k][idxs] for k in self.next_obs_buf.keys()}

        # Base transition fields; optional channels are appended so z / full_obs /
        # skill can coexist (old single-flag callers see identical keys).
        batch = AttrDict(observations=observations,
                         next_observations=next_observations,
                         actions=self.act_buf[idxs],
                         rewards=self.rew_buf[idxs],
                         dones=self.done_buf[idxs])
        if self.z:
            batch['z'] = self.z_buf[idxs]
        if self.full_obs:
            batch['full_obs'] = self.full_obs_buf[idxs]
        if self.track_skill:
            batch['idxs'] = idxs
            batch['skill_id'] = self.skill_id_buf[idxs]
            batch['source'] = self.source_buf[idxs]
            batch['reward_task'] = self.reward_task_buf[idxs]
            batch['reward_div'] = self.reward_div_buf[idxs]
            batch['reward_total'] = self.reward_total_buf[idxs]

        return batch

    def sample_stratified(self, batch_size=32, w0_min_ratio=0.5,
                          paired_sample_ratio=0.3, rng=None):
        """Source-stratified sampling (Task 4 / Requirements 4.2, 4.3).

        Controls the *final batch* composition (not just the rollout mix):
          - ``P(w_0) >= w0_min_ratio``  (protect the deployment skill),
          - ``P(diversity_paired) <= paired_sample_ratio``  (avoid paired data
            dominating the training distribution).
        Falls back to plain ``sample`` when skill tracking is off. Pools are
        drawn with replacement (matching the base buffer) and degrade gracefully
        when a pool is empty (e.g. early training with only w_0).
        Returns the usual batch plus ``batch_ratios`` (effective w0/probe/paired).
        """
        if not self.track_skill or self.size == 0:
            return self.sample(batch_size)
        rng = rng if rng is not None else np.random

        n = self.size
        all_idx = np.arange(n)
        skill = self.skill_id_buf[:n, 0]
        source = self.source_buf[:n, 0]
        w0_pool = all_idx[skill == 0]
        probe_pool = all_idx[skill != 0]
        paired_pool = probe_pool[source[probe_pool] == SOURCE_DIVERSITY_PAIRED] if probe_pool.size else probe_pool
        probe_normal_pool = probe_pool[source[probe_pool] == SOURCE_NORMAL] if probe_pool.size else probe_pool

        n_w0 = int(round(float(w0_min_ratio) * batch_size))
        n_w0 = max(0, min(n_w0, batch_size))
        n_probe = batch_size - n_w0
        n_paired = min(int(round(float(paired_sample_ratio) * batch_size)), n_probe)
        n_probe_normal = n_probe - n_paired

        def _draw(pool, k, *fallbacks):
            if k <= 0:
                return np.empty(0, dtype=np.int64)
            chosen = pool
            for fb in (pool,) + fallbacks + (all_idx,):
                if len(chosen) > 0:
                    break
                chosen = fb
            return rng.choice(chosen, size=k, replace=True).astype(np.int64)

        chunks = [
            _draw(w0_pool, n_w0, all_idx),
            _draw(paired_pool, n_paired, probe_normal_pool, probe_pool),
            _draw(probe_normal_pool, n_probe_normal, probe_pool),
        ]
        idxs = np.concatenate([c for c in chunks if len(c) > 0]) if any(len(c) for c in chunks) else np.empty(0, np.int64)
        if len(idxs) < batch_size:
            pad = rng.choice(all_idx, size=batch_size - len(idxs), replace=True).astype(np.int64)
            idxs = np.concatenate([idxs, pad])
        rng.shuffle(idxs)

        batch = self.sample(idxs=idxs)
        sk = self.skill_id_buf[idxs, 0]
        so = self.source_buf[idxs, 0]
        batch['batch_ratios'] = {
            'w0': float(np.mean(sk == 0)),
            'probe': float(np.mean(sk != 0)),
            'paired': float(np.mean(so == SOURCE_DIVERSITY_PAIRED)),
        }
        return batch

    def state_dict(self):
        state = {
            'z': self.z,
            'full_obs': self.full_obs,
            'obs_dict': self.obs_dict,
            'ptr': self.ptr,
            'size': self.size,
            'max_size': self.max_size,
            'act_buf': self.act_buf.copy(),
            'rew_buf': self.rew_buf.copy(),
            'done_buf': self.done_buf.copy(),
        }
        if self.obs_dict:
            state['obs_buf'] = {k: v.copy() for k, v in self.obs_buf.items()}
            state['next_obs_buf'] = {k: v.copy() for k, v in self.next_obs_buf.items()}
        else:
            state['obs_buf'] = self.obs_buf.copy()
            state['next_obs_buf'] = self.next_obs_buf.copy()
        if self.z:
            state['z_buf'] = self.z_buf.copy()
        if self.full_obs:
            state['full_obs_buf'] = self.full_obs_buf.copy()
        state['track_skill'] = self.track_skill
        if self.track_skill:
            state['skill_id_buf'] = self.skill_id_buf.copy()
            state['source_buf'] = self.source_buf.copy()
            state['reward_task_buf'] = self.reward_task_buf.copy()
            state['reward_div_buf'] = self.reward_div_buf.copy()
            state['reward_total_buf'] = self.reward_total_buf.copy()
        return state

    def load_state_dict(self, state_dict):
        self.z = bool(state_dict['z'])
        self.full_obs = bool(state_dict['full_obs'])
        self.obs_dict = bool(state_dict['obs_dict'])
        self.ptr = int(state_dict['ptr'])
        self.size = int(state_dict['size'])
        self.max_size = int(state_dict['max_size'])

        if self.obs_dict:
            self.obs_buf = {k: np.array(v, copy=True) for k, v in state_dict['obs_buf'].items()}
            self.next_obs_buf = {k: np.array(v, copy=True) for k, v in state_dict['next_obs_buf'].items()}
        else:
            self.obs_buf = np.array(state_dict['obs_buf'], copy=True)
            self.next_obs_buf = np.array(state_dict['next_obs_buf'], copy=True)

        self.act_buf = np.array(state_dict['act_buf'], copy=True)
        self.rew_buf = np.array(state_dict['rew_buf'], copy=True)
        self.done_buf = np.array(state_dict['done_buf'], copy=True)

        if self.z and 'z_buf' in state_dict:
            self.z_buf = np.array(state_dict['z_buf'], copy=True)
        if self.full_obs and 'full_obs_buf' in state_dict:
            self.full_obs_buf = np.array(state_dict['full_obs_buf'], copy=True)

        # Skill bookkeeping: absent in pre-skill (mode A) checkpoints -> stays off.
        self.track_skill = bool(state_dict.get('track_skill', False))
        if self.track_skill and 'skill_id_buf' in state_dict:
            self.skill_id_buf = np.array(state_dict['skill_id_buf'], copy=True)
            self.source_buf = np.array(state_dict['source_buf'], copy=True)
            self.reward_task_buf = np.array(state_dict['reward_task_buf'], copy=True)
            self.reward_div_buf = np.array(state_dict['reward_div_buf'], copy=True)
            self.reward_total_buf = np.array(state_dict['reward_total_buf'], copy=True)
    
class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d

class RandomProcess(object):
    def reset_states(self):
        pass

class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma

class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

    def state_dict(self):
        return {
            'theta': self.theta,
            'mu': self.mu,
            'sigma': self.sigma,
            'dt': self.dt,
            'x0': None if self.x0 is None else np.array(self.x0, copy=True),
            'size': self.size,
            'sigma_min': self.sigma_min,
            'n_steps': self.n_steps,
            'x_prev': np.array(self.x_prev, copy=True),
            'm': self.m,
            'c': self.c,
        }

    def load_state_dict(self, state_dict):
        self.theta = state_dict['theta']
        self.mu = state_dict['mu']
        self.sigma = state_dict['sigma']
        self.dt = state_dict['dt']
        self.x0 = None if state_dict['x0'] is None else np.array(state_dict['x0'], copy=True)
        self.size = state_dict['size']
        self.sigma_min = state_dict['sigma_min']
        self.n_steps = state_dict['n_steps']
        self.x_prev = np.array(state_dict['x_prev'], copy=True)
        self.m = state_dict['m']
        self.c = state_dict['c']

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

class ParameterModule(nn.Module):
    def __init__(
            self,
            init_value
    ):
        super().__init__()

        self.param = torch.nn.Parameter(init_value)

def jacobian_decoder_jvp_parallel(func, inputs, v=None, create_graph=True):
    batch_size, z_dim = inputs.size()
    if v is None:
        v = torch.eye(z_dim).unsqueeze(0).repeat(batch_size, 1, 1).view(-1, z_dim).to(inputs)
    inputs = inputs.repeat(1, z_dim).view(-1, z_dim)
    jac = (
        torch.autograd.functional.jvp(
            func, inputs, v=v, create_graph=create_graph
        )[1].view(batch_size, z_dim, -1).permute(0, 2, 1)
    )
    return jac

def relaxed_distortion_measure(func, z, eta=0.2, create_graph=True):
    '''
    func: decoder that maps "latent value z" to "data", where z.size() == (batch_size, latent_dim)
    '''
    bs = len(z)
    z_perm = z[torch.randperm(bs)]
    alpha = (torch.rand(bs) * (1 + 2*eta) - eta).unsqueeze(1).to(z)
    z_augmented = alpha*z + (1-alpha)*z_perm
    v = torch.randn(z.size()).to(z)
    Jv = torch.autograd.functional.jvp(
        func, z_augmented, v=v, create_graph=create_graph)[1]
    TrG = torch.sum(Jv.view(bs, -1)**2, dim=1).mean()
    JTJv = (torch.autograd.functional.vjp(
        func, z_augmented, v=Jv, create_graph=create_graph)[1]).view(bs, -1)
    TrG2 = torch.sum(JTJv**2, dim=1).mean()
    return TrG2/TrG**2

def compute_distance(traj1, traj2):
    assert traj1.shape == traj2.shape
    len_traj = traj1.shape[0]
    traj1_pos = (traj1[:,2:4]+1)*256
    traj1_angle = (traj1[:,-1]+1)*np.pi
    cos1 = np.cos(traj1_angle)
    sin1 = np.sin(traj1_angle)
    cos2 = np.cos(traj2_angle)
    sin2 = np.sin(traj2_angle)
    traj2_pos = (traj2[:,2:4]+1)*256
    traj2_angle = (traj2[:,-1]+1)*np.pi
    R1 = np.zeros((len_traj,3,3))
    R2 = np.zeros((len_traj,3,3))
    R1[:,-1,-1] = 1
    R2[:,-1,-1] = 1
    R1[:,:2,-1] = traj1_pos
    R2[:,:2,-1] = traj2_pos
    R1[:,0,0] = cos1
    R1[:,1,1] = cos1
    R1[:,0,1] = -sin1
    R1[:,1,0] = sin1
    R2[:,0,0] = cos2
    R2[:,1,1] = cos2
    R2[:,0,1] = -sin2
    R2[:,1,0] = sin2

    p = np.linalg.norm((R1 - R2)[:,:2, -1],axis=1)/(128/np.pi)
    w = np.arccos((np.linalg.inv(R1)@R2)[:,0,0])

    return (p+w).mean()
