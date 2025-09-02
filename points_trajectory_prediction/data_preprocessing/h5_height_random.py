import h5py
import numpy as np

def process_h5_file(h5_file_path):
    with h5py.File(h5_file_path, 'r') as h5_file:
        point_cloud_data = np.array(h5_file['pcd_traj'])  
        
        print(f"Original shape of point cloud data: {point_cloud_data.shape}")
        

        first_frame = point_cloud_data[0] 
        for i in range(1, point_cloud_data.shape[0]): 
            delta_y = point_cloud_data[i][:, 1] - first_frame[:, 1] 
            
            scaled_delta_y = delta_y * (2 / 3)
            
            point_cloud_data[i][:, 1] = first_frame[:, 1] + scaled_delta_y
        
        return point_cloud_data

h5_file_path = '/data2/chaonan/cloth_traj_data/TCSC/TCSC_083_action2/pcd_traj.h5'
processed_data = process_h5_file(h5_file_path)

print(f"Processed shape of point cloud data: {processed_data.shape}")