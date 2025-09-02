import torch
from torch.utils.data import Dataset
import numpy as np
import os
from data_preprocessing.dataset_utils import read_point_cloud_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import re
from utils.category import clothing_encoding, clothing_decoding
from utils.data_utils import description_encoding


class PointCloudTrajectoryTransform:
    def __init__(self):
        pass  

    def random_rotate(self, points, trajectory):
        """Randomly rotate the point cloud and trajectory along the Z-axis."""
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = torch.tensor([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        return torch.matmul(points, rotation_matrix), torch.matmul(trajectory, rotation_matrix)

    def random_scale(self, points, trajectory, scale_range=(0.8, 1.2)):
        """Randomly scale the point cloud and trajectory."""
        scale = np.random.uniform(*scale_range)
        return points * scale, trajectory * scale

    def random_translate(self, points, trajectory, translate_range=(-0.2, 0.2)):
        """Randomly translate the point cloud and trajectory."""
        translation = torch.tensor(np.random.uniform(*translate_range, size=(1, 3)), dtype=torch.float32)
        return points + translation, trajectory + translation
    
    def __call__(self, sample):
        points, trajectory = sample['point_cloud'], sample['trajectory']
        points, trajectory = self.random_rotate(points, trajectory)
        points, trajectory = self.random_scale(points, trajectory)
        points, trajectory = self.random_translate(points, trajectory)
        return {'point_cloud': points, 'trajectory': trajectory}



class PointCloudTrajectoryDataset(Dataset):
    """
    A custom Dataset for loading point cloud data and corresponding trajectories,
    keeping everything as samples.
    """
    def __init__(self, data_dir, split='train', transform=None):
        """
        Initialize the dataset.

        Parameters:
        - data_dir: Directory where the data files are stored.
        - split: 'train' for training data or 'eval' for evaluation data.
        - transform: Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        # Load the full dataset
        full_samples = read_point_cloud_dataset(file_name=self.data_dir)   
             
        # Determine split size
        if len(full_samples) >= 5:
            split_size = len(full_samples) // 5  # 20% for eval, 80% for train
            self.samples = []
            if self.split == 'train':
                for i in range(len(full_samples)):
                    match = re.search(r'(\d+)_action', full_samples[i]['name'])     
                    if match and int(match.group(1)) % 5 != 0:  
                        self.samples.append(full_samples[i])
                # self.samples = full_samples[:-split_size]
            elif self.split == 'eval':
                for i in range(len(full_samples)):
                    match = re.search(r'(\d+)_action', full_samples[i]['name'])
                    if (not match) or int(match.group(1)) % 5 == 0:
                        self.samples.append(full_samples[i])
                        # print(full_samples[i]['name'])
                # self.samples = full_samples[-split_size:]
            elif self.split == 'all':
                self.samples = full_samples[:]
            else:
                raise ValueError("Invalid split. Expected 'train' or 'eval' or 'all'.")
        else:
            if self.split == 'train':
                if len(full_samples) > 1:
                    self.samples = full_samples[:-1]
                else:
                    self.samples = full_samples[:]
            elif self.split == 'eval':
                self.samples = full_samples[-1:]
            elif self.split == 'all':
                self.samples = full_samples[:]
            else:
                raise ValueError("Invalid split. Expected 'train' or 'eval' or 'all'.")

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        Fetch a single sample from the dataset.

        Parameters:
        - index: Index of the sample to be fetched.

        Returns:
        - sample: A dictionary containing the point cloud and its corresponding trajectory.
        """
        sample = self.samples[index]
        
        # Apply transform if specified
        if self.transform:
            sample = self.transform(sample)
        
        return sample


if __name__ == '__main__':
    # data_dir = '/data2/chanchen/projects/points-traj-prediction/datasets/point_cloud_samples.h5'
    data_dir = '/data2/chaonan/points-traj-prediction/data/all_normalized_data.h5'


    eval_dataset = PointCloudTrajectoryDataset(data_dir, split='eval')
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=2, shuffle=False)

    for batch_idx, batch in enumerate(eval_loader):
        pass

