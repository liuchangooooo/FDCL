import torch
from DIVO.sampler.base import Base
from DIVO.nets.sampler_components import conditional_ode_integrator
from DIVO.utils.util import MMDCalculator
from tqdm import tqdm
import copy
import numpy as np

class FlowMatching(Base):
    def __init__(
        self, 
        velocity_field,
        prob_path='OT',
        sigma_1=0.01,
        z_dim=1,
        core_cond=None,
        *args,
        **kwargs
        ):
        super(FlowMatching, self).__init__()
        self.velocity_field = velocity_field
        self.prob_path = prob_path
        self.sigma_1 = sigma_1
        self.z_dim = z_dim

        self.mmd_func = MMDCalculator(type='L2_traj')

        if core_cond is None:
            self.core_cond = torch.zeros(1, z_dim)
        else:
            self.core_cond = core_cond

        self.register_buffer('mean', torch.zeros(1, z_dim))
        self.register_buffer('std', torch.ones(1, z_dim))
        
    def normalize(self, dl, device):
        zs = []
        for _, z in dl:
            zs.append(z)
        zs = torch.cat(zs, dim=0)
        self.mean = zs.mean(dim=0, keepdim=True).to(device)
        self.std = zs.std(dim=0, keepdim=True).to(device)

    def forward(self, t, z, cond, device):
        if isinstance(cond, list):
            cond = torch.cat(cond, dim=0)

        q = cond.to(device,dtype=torch.float32)
        z = z.to(device,dtype=torch.float32)
        t = t.to(device,dtype=torch.float32)
        
        return self.velocity_field(t, z, q)
    
    def velocity(self, t, z ,cond, device, guidance=None):
        if guidance is None:
            return self(t, z, cond, device)
        else:
            v0 = self(t, z, [self.core_cond]*len(cond), device)
            vc = self(t, z, cond, device)
            return v0 + guidance*(vc - v0)
    
    def sample(
            self, 
            cond, 
            device='cuda:0',
            method='euler',
            guidance=None,
            output_traj=False,
            **kwargs
        ):
        rand_init = torch.randn(len(cond), self.z_dim).to(device)
        
        def func(t, z, cond, device):
            return self.velocity(t, z, cond, device, guidance=guidance)
        
        ode_results = conditional_ode_integrator(
            func, 
            t0=0, 
            t1=1, 
            rand_init=rand_init, 
            cond=cond, 
            method=method, 
            device=device, 
            output_traj=output_traj,
            **kwargs
        )
        if output_traj:
            gen_z = ode_results[0]
            traj = ode_results[1]
        else:
            gen_z = ode_results 
        
        gen_z = self.std*gen_z + self.mean
        
        if output_traj:
            traj = self.std.unsqueeze(1)*traj + self.mean.unsqueeze(1)
            return traj

        else:
            return gen_z
        
    def Gaussian_t_xbar_x1(self, t, z1, **kwargs):
        if self.prob_path == 'OT':
            mu_t = t*z1
            sigma_t = torch.ones_like(t) - (1-self.sigma_1) * t
        elif self.prob_path == 'VE':
            mu_t = z1
            sigma_t = 3*(1-t)
        elif self.prob_path == 'VP':
            alpha_1mt = torch.exp(-5*(1-t)**2)
            mu_t = alpha_1mt*z1 
            sigma_t = torch.sqrt((1-alpha_1mt**2))       
        elif self.prob_path == 'VP2':
            alpha_1mt = torch.exp(-0.5*(1-t)**2)
            mu_t = alpha_1mt*z1 
            sigma_t = torch.sqrt((1-alpha_1mt**2))       
        return mu_t, sigma_t
    
    def sample_from_Gaussian_t_xbar_x1(self, t, z1, **kwargs):
        mu, sigma = self.Gaussian_t_xbar_x1(t, z1, **kwargs)
        samples = sigma*torch.randn_like(z1) + mu
        return samples
    
    def u_t_xbarx1(self, t, z, z1, **kwargs):
        u = (z1 - (1-self.sigma_1)*z)/(1 - (1-self.sigma_1)*t)
        return u 
    
    def compute_loss(self, cond, z, *args, **kwargs):
        bs = len(z)
        
        t = torch.rand(bs, 1).to(z)
        z1 = z
        z1 = (z1-self.mean)/self.std
        z = self.sample_from_Gaussian_t_xbar_x1(t, z1)
        u_t = self.u_t_xbarx1(t, z, z1)
        v_t = self(t, z, cond, device=z.device)
        
        loss = ((u_t.detach() - v_t)**2).mean()

        return loss
        
    def eval_step(self, val_loader, device):
        dict_for_evals = val_loader.dataset.dict_for_evals
        mmd_list = []
        for key, item in tqdm(dict_for_evals.items()):
            samples = self.sample(
                100*[key], 
                dt=0.1, 
                sample_z=False, 
                guidance=1, 
                device=device)
            mmd = self.mmd_func(item.to(device), samples)
            mmd_list.append(copy.copy(torch.tensor([mmd], dtype=torch.float32)))
        mmd_torch = torch.cat(mmd_list)
        
        return {
            'mmd_avg_': mmd_torch.mean(),
            'mmd_std_': mmd_torch.std(),
            'mmd_ste_': mmd_torch.std()/np.sqrt(len(mmd_torch)),
        }
