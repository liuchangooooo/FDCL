import torch.nn as nn

class Base(nn.Module):  
    def fit(self, data):
        raise NotImplementedError()
    
    def compute_loss(self, obs, z):
        raise NotImplementedError()

    def sample(self, obs):
        pass


    