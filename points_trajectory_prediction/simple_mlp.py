import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

class PositionalEncodingMLP(nn.Module):
    def __init__(self):
        super(PositionalEncodingMLP, self).__init__()

        self.fc1 = nn.Linear(3, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(64, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm1d(512)

    def forward(self, x):
        # [batch_size, num_points, 3]
        n, p, _ = x.shape

        x = x.view(-1, 3)  # [batch_size * num_points, 3]
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = x.view(n, p, 512)

        return x

class PointnetMLP(nn.Module):
    def __init__(self):
        super(PointnetMLP, self).__init__()

        self.fc1 = nn.Linear(3, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(128, 63)
        self.bn3 = nn.BatchNorm1d(63)

    def forward(self, x):
        n, p, _ = x.shape

        x = x.view(-1, 3)  # [batch_size * num_points, 3]
        x = F.relu(self.bn1(self.fc1(x)))  
        x = self.dropout1(x)
        x = x.view(n, p, -1)  

        x = x.view(-1, 64)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = x.view(n, p, -1)

        x = x.view(-1, 128)
        x = F.relu(self.bn3(self.fc3(x)))
        x = x.view(n, p, -1)

        return x

class TransformerMLP(nn.Module):
    def __init__(self):
        super(TransformerMLP, self).__init__()

        self.fc1 = nn.Linear(128, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(512, 63)
        self.bn3 = nn.BatchNorm1d(63)

    def forward(self, x):
        # [batch_size, num_points, 128]
        n, p, _ = x.shape
        # ipdb.set_trace()

        x = x.view(-1, 128)  # [batch_size * num_points, 128]
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = x.view(n, p, 21, 3)

        return x


