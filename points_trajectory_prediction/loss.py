import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def chamfer_distance(p1, p2):
    '''
        inputs: [batch_size, num_points, feature_dim]
    '''
    diff = p1[:, :, None, :] - p2[:, None, :, :]
    dist = torch.sum(diff ** 2, dim=-1)

    min_dist_p1_to_p2 = torch.min(dist, dim=2)[0]  # [batch_size, num_points]
    min_dist_p2_to_p1 = torch.min(dist, dim=1)[0]  # [batch_size, num_points]

    chamfer_dist = torch.mean(min_dist_p1_to_p2, dim=1) + torch.mean(min_dist_p2_to_p1, dim=1)
    total_chamfer_dist = torch.mean(chamfer_dist)

    return total_chamfer_dist

def exponential_weighted_distance(pred, target, weight_factor=1.0):
    diff = pred - target
    squared_diff = torch.square(diff)
    euclidean_distances = torch.sqrt(torch.sum(squared_diff, dim=-1))

    weighted_distances = torch.exp(weight_factor * euclidean_distances) - 1

    total_weighted_distance = torch.mean(weighted_distances)
    
    return total_weighted_distance

def cosine_similarity(pred, target):
    cos_sim = F.cosine_similarity(pred.view(pred.size(0), -1), target.view(target.size(0), -1), dim=1)
    # similarity_score = (cos_sim * 50) + 50
    similarity_score = cos_sim
    return similarity_score.mean()


class Loss(nn.Module):
    def __init__(self, weight_cd=0, weight_mse=1.0, weight_mae=0.0, weight_exp=1.0, weight_traj=10.0, decay_factor=0.95):
        super(Loss, self).__init__()
        self.weight_cd = weight_cd
        self.weight_mse = weight_mse
        self.weight_mae = weight_mae
        self.weight_exp = weight_exp
        self.weight_traj = weight_traj
        self.decay_factor = decay_factor
        self.mse_loss = nn.MSELoss(reduction='none')        # keep dim, 算轨迹误差
        self.cls_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()


    def forward(self, pred, target, mask, z_mu, z_logvar):
        traj_loss = self.mse_loss(pred, target)  # (batch_size, num_points, num_frames, 3)

        reconstruction_loss = traj_loss.sum() / mask.numel()   

        kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())

        total_loss = reconstruction_loss + 0.01 * kl_loss

        return reconstruction_loss,  0.01 * kl_loss, total_loss