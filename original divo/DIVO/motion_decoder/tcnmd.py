import torch
import torch.nn as nn
from DIVO.motion_decoder.base import Base
from DIVO.motion_decoder.tcn import _TCNModule

class TCNMotionDecoder(Base):
    def __init__(
        self,
        in_chan,
        out_chan,
        hidden_chan,
        kernel_size,
        dilation_base,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        len_traj = kwargs['len_traj']
        self.tcn = _TCNModule(
            input_size=in_chan,
            input_chunk_length = len_traj+1,
            kernel_size=kernel_size,
            num_filters=hidden_chan,
            num_layers=None, # chosen automatically to ensure full dilation coverage
            dilation_base=dilation_base,
            weight_norm=False,
            target_size=out_chan,
            target_length=len_traj+1,
            dropout=0)

    
    def forward(self, z):
        bs = z.shape[0]
        z = z.view(bs, self.len_traj, -1)
        z_final = z[:, -1, :]
        z = torch.cat([z, z_final.unsqueeze(1)], dim=1) # we need information upto t+1 to predict t
        x = self.tcn(z)
        return x[:, 1:, :] # we need information upto t+1 to predict t