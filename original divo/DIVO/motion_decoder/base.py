import torch
import torch.nn as nn

class Base(nn.Module):    

    def __init__(self, **kwargs):
        super().__init__()
        self.len_traj = kwargs['len_traj']
        self.loss_criterion = kwargs['loss_criterion'] if 'loss_criterion' in kwargs else 'mse'
        
    def compute_loss(self, obs, z, motion):
        pred_motion = self.se2eq_pred(obs, z)

        if self.loss_criterion == 'mse':
            loss = nn.functional.mse_loss(pred_motion, motion)
        elif self.loss_criterion == 'feature_mse':
            state_dim = motion.shape[-1]
            flattened_motion = motion.view(-1, state_dim) # (batch_size * len_traj, state_dim)
            flattened_pred_motion = pred_motion.view(-1, state_dim) # (batch_size * len_traj, state_dim)
            tblock_feature_points = self.get_tblock_feature_points(flattened_motion) # (batch_size * len_traj, num_tblock_feature_points (=4) , 2)
            pred_tblock_feature_points = self.get_tblock_feature_points(flattened_pred_motion) # (batch_size * len_traj, num_tblock_feature_points (=4) , 2)
            loss = nn.functional.mse_loss(tblock_feature_points, pred_tblock_feature_points)
        return loss

    def sample(self, obs, z):

        pred_motion = self.se2eq_pred(obs, z)

        return pred_motion

    def forward(self, z):
        """
        inputs action sequence observed at initial frame 
        input shape: (batch_size, action_dim * len_traj)
        outputs unnormalized state sequence prediction observed at initial frame 
        output shpe: (batch_size, len_traj, state_dim)
        """
        pass

    def se2eq_pred(self, obs, z):
        """
        Compute SE(2) equivariant action sequence
        obs: (batch_size, state_dim)
        z: (batch_size, action_dim * len_traj)
        output: (batch_size, len_traj, state_dim)
        """
        batch_size = obs.shape[0]
        obs = obs
        init_pos = obs[:, :2] # (batch_size, 2)
        init_cos = obs[:, 2]
        init_sin = obs[:, 3]
        init_rot_matrix = torch.stack([init_cos, -init_sin, init_sin, init_cos], dim=-1).view(batch_size, 2, 2).unsqueeze(1) # (batch_size, 1, 2, 2)
        init_rot_inv_matrix = torch.stack([init_cos, init_sin, -init_sin, init_cos], dim=-1).view(batch_size, 2, 2).unsqueeze(1) # (batch_size, 1, 2, 2)
        z = z.reshape(batch_size, self.len_traj, -1) # (batch_size, len_traj, action_dim)
        z = z - init_pos.unsqueeze(1) # (batch_size, len_traj, action_dim)
        z = torch.matmul(init_rot_inv_matrix, z.unsqueeze(-1)).squeeze(-1) # (batch_size, len_traj, action_dim)
        pred_motion = self.forward(z.reshape(batch_size, -1)) # (batch_size, len_traj, state_dim)
        # Rotate the predicted position
        pred_pos = pred_motion[:, :, :2]
        pred_pos = torch.matmul(init_rot_matrix, pred_pos.unsqueeze(-1)).squeeze(-1)
        # Add the initial position back
        pred_pos = pred_pos + init_pos.unsqueeze(1)
        # Rotate the predicted orientation
        pred_ori = pred_motion[:,:,2:] /torch.norm(pred_motion[:,:,2:],dim=-1,keepdim=True) # (batch_size, len_traj, 2)
        pred_ori = torch.matmul(init_rot_matrix, pred_ori.unsqueeze(-1)).squeeze(-1)
        output = torch.cat([pred_pos, pred_ori], dim=-1)
        output = output
        return output

    def get_tblock_feature_points(self, obs):
        """
        Calculate the feature points of the tblock given the center position and orientation. 
        Four corners are considered as the feature points.
        input: (batch_size, state_dim)
        output: (batch_size, num_tblock_feature_points (=4) , 2)
        """
        center = obs[:, :2]
        cos = obs[:, 2]
        sin = obs[:, 3]
        rot_matrix = torch.stack([cos, -sin, sin, cos], dim=-1).view(-1, 2, 2).unsqueeze(1) # (batch_size, 1, 2, 2)
        rel_tblock_feature_points = torch.tensor([[0.2, 0.06], [-0.2, 0.06], [0.06, -0.34], [-0.06, -0.34]]).unsqueeze(0).repeat(center.shape[0], 1, 1).to(obs.device) # (batch_size, 4, 2)
        rot_rel_tblock_feature_points = torch.matmul(rot_matrix, rel_tblock_feature_points.unsqueeze(-1)).squeeze(-1) # (batch_size, 4, 2)
        tblock_feature_points = center.unsqueeze(1) + rot_rel_tblock_feature_points
        return tblock_feature_points