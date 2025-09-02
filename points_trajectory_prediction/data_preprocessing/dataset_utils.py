import h5py
import numpy as np
import torch

def read_point_cloud_dataset(file_name='point_cloud_samples.h5'):
    """
    Read point cloud samples and their trajectories from an HDF5 file,
    convert them to torch.Tensors, and store them in a list of dictionaries.

    Parameters:
    - file_name: The name of the HDF5 file from which to read the data.
    """
    samples = []

    with h5py.File(file_name, 'r') as hf:
        for sample_name in hf.keys():
            # print(sample_name)
            grp = hf[sample_name]
            
            # Convert NumPy arrays directly to torch.Tensors
            points = torch.tensor(grp['points'][:], dtype=torch.float32)
            trajectories = torch.tensor(grp['trajectories'][:], dtype=torch.float32)
            description = grp['description'][()].decode('utf-8')
            description_embed = torch.tensor(grp['description_embed'][:], dtype=torch.float32)
            
            # Add each sample as a dictionary to the list
            samples.append({'point_cloud': points, 'trajectory': trajectories, 'name': sample_name, 'description': description, 'description_embed': description_embed})

    return samples

if __name__ == '__main__':
    # samples = read_point_cloud_dataset(file_name='/data2/chanchen/projects/points-traj-prediction/datasets/point_cloud_samples.h5')
    # samples = read_point_cloud_dataset(file_name='/data2/chaonan/points-traj-prediction/data/PL_data.h5')
    # samples = read_point_cloud_dataset(file_name='/data2/chaonan/points-traj-prediction/data/TNLC_normalized_isaac_data.h5')
    samples = read_point_cloud_dataset(file_name='/data2/chaonan/points-traj-prediction/data/all_normalized_data.h5')

    # Verify the data
    print(f"Total number of samples: {len(samples)}")
    print(f"Shape of the point cloud in the first sample: {samples[0]['point_cloud'].shape}")
    print(f"Shape of the trajectory in the first sample: {samples[0]['trajectory'].shape}")
