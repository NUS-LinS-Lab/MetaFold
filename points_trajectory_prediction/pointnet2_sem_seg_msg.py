import torch.nn as nn
import torch.nn.functional as F
import torch
from pointnet2_utils import PointNetSetAbstraction,PointNetSetAbstractionMsg,PointNetFeaturePropagation


class FeatureEncoder(nn.Module):
    def __init__(self):
        super(FeatureEncoder, self).__init__()

        # Set Abstraction layers
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 3, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])

        # Feature Propagation layers
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        # print('shallow pointnet')
        # self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 9 + 3, [32, 32, 64], False)
        # self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        # self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        # self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        # self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        # self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        # self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        # self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        # print("scale wider Pointnet")
        # # Set Abstraction layers
        # self.sa1 = PointNetSetAbstractionMsg(1024, [0.5, 1], [16, 32], 6, [[16, 16, 32], [32, 32, 64]])
        # self.sa2 = PointNetSetAbstractionMsg(256, [1, 2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        # self.sa3 = PointNetSetAbstractionMsg(64, [2, 4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        # self.sa4 = PointNetSetAbstractionMsg(16, [4, 8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])

        # # Feature Propagation layers
        # self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        # self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        # self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        # self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        

        self.conv0 = nn.Conv1d(128, 128, 1)
        self.bn0 = nn.BatchNorm1d(128)
        self.drop0 = nn.Dropout(0.5)
        # self.conv2 = nn.Conv1d(128, 128, 1)
        # self.conv3 = nn.Conv1d(128, 128, 1)
        # self.conv3 = nn.Conv1d(128, 63, 1)
        # self.conv4 = nn.Conv1d(128, 63, 1)

    def forward(self, xyz):
        '''
            input: [batch, point, feature]
            output: [batch, point, feature]
        '''
        xyz = xyz.permute(0, 2, 1)
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        # print(l0_xyz.shape, l0_points.shape)

        # Set Abstraction layers
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Feature Propagation layers
        # l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        # l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        x = self.drop0(F.relu(self.bn0(self.conv0(l0_points))))
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # Return the encoded features at different levels
        return x.permute(0, 2, 1) #, l1_points, l2_points, l3_points, l4_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == '__main__':
    model = FeatureEncoder()
    xyz = torch.rand(6, 9, 2048)  # Example input   : batch_size, feature_dim, point_num
    features = model(xyz)
    for i, feature in enumerate(features):
        print(f"Encoded features {i}: ", feature.shape)
'''
Encoded features 0:  torch.Size([6, 128, 2048])     # batch_size, feature_dim, point_num
Encoded features 1:  torch.Size([6, 128, 1024])
Encoded features 2:  torch.Size([6, 256, 256])
Encoded features 3:  torch.Size([6, 256, 64])
Encoded features 4:  torch.Size([6, 1024, 16])
'''