from typing import Union
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import math
from torch.nn.modules.batchnorm import _BatchNorm
import copy
from DIVO.utils.LieGroup_torch import skew, exp_so3

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)
    
class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int],
            **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        
        x = sample
        h = []
        for idx, (resnet, resnet2) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x

class ConditionalResidualBlockFcn(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,):
        super().__init__()

        self.blocks = nn.ModuleList([
            nn.Linear(in_channels, out_channels),
            nn.Linear(out_channels, out_channels),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.GELU(),
            nn.Linear(cond_dim, cond_channels),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Linear(in_channels, out_channels) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out
    
class ConditionalUnetFcn(nn.Module):
    def __init__(self, 
        input_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.GELU(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlockFcn(mid_dim, mid_dim, cond_dim),
            ConditionalResidualBlockFcn(mid_dim, mid_dim, cond_dim),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlockFcn(dim_in, dim_out, cond_dim),
                ConditionalResidualBlockFcn(dim_out, dim_out, cond_dim),
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlockFcn(dim_out*2, dim_in, cond_dim),
                ConditionalResidualBlockFcn(dim_in, dim_in, cond_dim),
            ]))
        
        final_fcn = nn.Sequential(
            nn.Linear(start_dim, start_dim),
            nn.GELU(),
            nn.Linear(start_dim, input_dim),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_fcn = final_fcn

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int],
            **kwargs):
        """
        x: (B,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,input_dim)
        """

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        
        x = sample
        h = []
        for idx, (resnet, resnet2) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)

        x = self.final_fcn(x)

        return x
    
class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        max_value=0.9999
    ):
        """
        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        Args:
            inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
            power (float): Exponential factor of EMA warmup. Default: 2/3.
            min_value (float): The minimum EMA decay rate. Default: 0.
        """

        self.averaged_model = model
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        self.decay = self.get_decay(self.optimization_step)

        # old_all_dataptrs = set()
        # for param in new_model.parameters():
        #     data_ptr = param.data_ptr()
        #     if data_ptr != 0:
        #         old_all_dataptrs.add(data_ptr)

        all_dataptrs = set()
        for module, ema_module in zip(new_model.modules(), self.averaged_model.modules()):            
            for param, ema_param in zip(module.parameters(recurse=False), ema_module.parameters(recurse=False)):
                # iterative over immediate parameters only.
                if isinstance(param, dict):
                    raise RuntimeError('Dict parameter not supported')
                
                # data_ptr = param.data_ptr()
                # if data_ptr != 0:
                #     all_dataptrs.add(data_ptr)

                if isinstance(module, _BatchNorm):
                    # skip batchnorms
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                elif not param.requires_grad:
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay)

        # verify that iterating over module and then parameters is identical to parameters recursively.
        # assert old_all_dataptrs == all_dataptrs
        self.optimization_step += 1

def conditional_ode_integrator(model, t0, t1, rand_init, cond, method, device, output_traj=False, **kwargs):
    bs = len(rand_init)
    dt = kwargs['dt']
    T = t1 - t0
    traj = [copy.deepcopy(rand_init.unsqueeze(1))]
    if method == 'euler':
        z = rand_init
        t = torch.tensor(t0, dtype=torch.float32).unsqueeze(0).repeat(bs, 1).to(device)
        for _ in range(int(T/dt)):
            zdot = copy.copy(model(t, z, cond, device).detach())
            z += zdot*dt
            t += torch.tensor(dt, dtype=torch.float32).to(device)
            traj.append(copy.deepcopy(z.unsqueeze(1)))
        if output_traj:
            return z, torch.cat(traj, dim=1)
        else:
            return z
    elif method == 'se3-euler':
        SE3traj = rand_init # bs x len x 4 x 4 
        t = torch.tensor(t0, dtype=torch.float32).unsqueeze(0).repeat(bs, 1).to(device)
        for _ in range(int(T/dt)):
            se3trajdot = copy.copy(model(t, SE3traj, cond, device).detach()) # bs x len x 6 
            so3trajdot = se3trajdot[:, :, :3] # bs x len x 3 
            p3trajdot = se3trajdot[:, :, 3:] # bs x len x 3 
            
            deltaSO3traj = exp_so3(skew(so3trajdot.view(-1, 3))*dt).view(bs, -1, 3, 3)
            SE3traj[:, :, :3, :3] = SE3traj[:, :, :3, :3]@deltaSO3traj
            SE3traj[:, :, :3, 3] += p3trajdot*dt
            t += torch.tensor(dt, dtype=torch.float32).to(device)
        return SE3traj