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

class StateDictReplayBuffer:
    def __init__(self, size, obs_dim, action_dim, z_dim=None, full_obs=None, obs_entry_info=None):
        self.z = False
        self.full_obs = False
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
        self.ptr, self.size, self.max_size = 0, 0, size

    def add(self, obs, next_obs, act, rew, done, z=None, full_obs=None):
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

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        if not self.obs_dict:
            observations = self.obs_buf[idxs]
            next_observations = self.next_obs_buf[idxs]
        else:
            observations = {k: self.obs_buf[k][idxs] for k in self.obs_buf.keys()}
            next_observations = {k: self.next_obs_buf[k][idxs] for k in self.next_obs_buf.keys()}

        if self.z:
            batch = AttrDict(observations=observations,
                            next_observations=next_observations,
                            actions=self.act_buf[idxs],
                            rewards=self.rew_buf[idxs],
                            dones=self.done_buf[idxs],
                            z=self.z_buf[idxs])
        elif self.full_obs:
            batch = AttrDict(observations=observations,
                            next_observations=next_observations,
                            actions=self.act_buf[idxs],
                            rewards=self.rew_buf[idxs],
                            dones=self.done_buf[idxs],
                            full_obs=self.full_obs_buf[idxs])
        elif self.z and self.full_obs:
            batch = AttrDict(observations=observations,
                            next_observations=next_observations,
                            actions=self.act_buf[idxs],
                            rewards=self.rew_buf[idxs],
                            dones=self.done_buf[idxs],
                            z=self.z_buf[idxs],
                            full_obs=self.full_obs_buf[idxs])
        else:
            batch = AttrDict(observations=observations,
                            next_observations=next_observations,
                            actions=self.act_buf[idxs],
                            rewards=self.rew_buf[idxs],
                            dones=self.done_buf[idxs])

        return batch
    
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