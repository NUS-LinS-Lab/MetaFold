import sys
sys.path.append('points_trajectory_prediction')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Transformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.init import xavier_uniform_, xavier_normal_
from multihead_point_transformer_pytorch import MultiheadPointTransformerLayer
from .pointnet2 import PointNet2
import inspect

from pointnet2_sem_seg_msg import FeatureEncoder
from simple_mlp import PointnetMLP, TransformerMLP, PositionalEncodingMLP
import ipdb
import time
# import dgl
# from equivariant_attention.modules import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res, GMaxPooling, GAvgPooling
# from equivariant_attention.fibers import Fiber
# from utils.data_utils import connect_fully, build_knn_graph

class TrajectoryTransformer(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_encoder_layers, num_decoder_layers, num_points, num_frames, point_dim, device, dropout=0.1, max_seq_length=5000):
        super(TrajectoryTransformer, self).__init__()
        assert hidden_dim % nhead == 0, "hidden_dim must be divisible by nhead"

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nhead = nhead
        self.max_seq_length = max_seq_length
        self.num_points = num_points
        self.num_frames = num_frames
        self.point_dim = point_dim
        self.device = device
        model = PointNet2(in_dim=3, hidden_dim=128, out_dim=128)
        print(model.__class__.__name__)

        for base in model.__class__.__bases__:
            print(base.__name__)

        print(inspect.getmodule(model.__class__)) 
        self.obj_feature_net = PointNet2(in_dim=3, hidden_dim=128, out_dim=128)
        # self.obj_feature_net = FeatureEncoder()
        # self.obj_feature_net = PointnetMLP()
        # self.test_transformer = TransformerMLP()
        # self.input_projection = nn.Linear(3, 128)
        # self.output_projection = nn.Linear(128, 63) # hidden_dim=128
        self.output_projection = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_frames*point_dim)
        )    

        self.sem_mlp = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
  
        self.pcd_dim = 128
        self.sem_dim = 128
        self.latent_dim = 128
        self.cat_size = self.pcd_dim +self.sem_dim + num_frames*point_dim
        self.hidden_size = 256
        self.decode_size = self.latent_dim + self.pcd_dim +self.sem_dim
        self.cls = nn.Parameter(torch.randn(1, self.cat_size))
        self.mlp_mu = nn.Linear(self.hidden_size, self.latent_dim)
        self.mlp_logvar = nn.Linear(self.hidden_size, self.latent_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=nhead, dropout=dropout)
        self.cvae_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        
        decoder_layer = nn.TransformerEncoderLayer(d_model=self.decode_size, nhead=nhead, dropout=dropout)        # Use Transformer Encoder as CVAE decoder
        self.cvae_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_decoder_layers)

        self.encoder_mlp = nn.Linear(self.cat_size, self.hidden_size)

        self.decoder_mlp = nn.Sequential(
            nn.Linear(self.decode_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_frames*point_dim)
        )


    def reparameterize(self, mu, logvar, max_std=1.0):
        std = torch.exp(0.5 * logvar)
        std_clipped = torch.clamp(std, max=max_std)
        eps = torch.randn_like(std_clipped)
        return mu + eps * std_clipped
    
    def forward(self, pcd, language_embedding, gt_traj=None, mode='train'):
        B, N, _ = pcd.shape
        # language_embedding = language_embedding.unsqueeze(1)
        # print(language_embedding.shape)
        _, N_l, _ = language_embedding.shape
        M = gt_traj.shape[2] if gt_traj is not None else 0

        cls_token = self.cls.expand(B, -1, -1)
        
        pcd_feat = self.obj_feature_net(pcd)        # [B, N, 128]
        
        sem_feat = self.sem_mlp(language_embedding)
        sem_feat = sem_feat.expand(-1, pcd.shape[1], -1)


        if mode == 'train':
            gt_traj_flat = gt_traj.reshape(B, N, -1)
            encoder_input = torch.cat([pcd_feat, sem_feat, gt_traj_flat], dim=2)  # [B, N, cat_size=128+128+3*M]
            encoder_input = torch.cat([cls_token, encoder_input], dim=1)        # [B, 1+N, cat_size=128+128+3*M]
            encoder_input = self.encoder_mlp(encoder_input)                     # [B, 1+N, hidden_size=512]
            encoder_output = self.cvae_encoder(encoder_input)
            # print('encoder_input: ', encoder_input.shape)
            # print('encoder_output: ', encoder_output.shape)
            

            z_cls_output = encoder_output[:, 0]  # [B, hidden_size]
            # print('z_cls_output: ', z_cls_output.shape)
            z_mu = self.mlp_mu(z_cls_output)  # [B, latent_dim]
            # print('z_mu: ', z_mu.shape)
            z_logvar = self.mlp_logvar(z_cls_output)  # [B, latent_dim]
            # print('z_logvar: ', z_logvar.shWape)

            z = self.reparameterize(z_mu, z_logvar)

        else:
            z_mu = torch.zeros(B, self.latent_dim).to(pcd.device)
            z_logvar = torch.zeros(B, self.latent_dim).to(pcd.device)
            z = torch.randn(B, self.latent_dim).to(pcd.device)

        z_expand = z.unsqueeze(1).expand(B, N, -1)  # [B, N, latent_dim]
        decoder_input = torch.cat([z_expand, pcd_feat, sem_feat], dim=2)  # [B, N, latent_dim + 128 + 128]

        decoder_output = self.cvae_decoder(decoder_input)
        pred_traj_off = self.decoder_mlp(decoder_output)  # 输出轨迹 [B, N, 3*M]

        return pred_traj_off.reshape(pcd.shape[0], pcd.shape[1], 21, 3), z_mu, z_logvar
