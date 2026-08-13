import torch
import torch.nn as nn

class LatentDetPolicy(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        obs2state,
        *args,
        **kwargs
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder 
        self.obs2state = obs2state
        
    def predict_action(self, obs):
        z = self.encoder(obs)
        state = self.obs2state(obs)
        action = self.decoder(torch.cat([state, z], dim=1))
        return action
    
    def reset(self):
        pass
    
    