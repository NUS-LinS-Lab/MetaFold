import numpy as np

class RandomRotate(object):
    """
    Randomly rotate the point cloud around the z-axis.
    """
    def __call__(self, sample):
        point_cloud, trajectory = sample['point_cloud'], sample['trajectory']
        
        rotation_angle = np.random.uniform() * 2 * np.pi
        cos_val = np.cos(rotation_angle)
        sin_val = np.sin(rotation_angle)
        
        rotation_matrix = np.array([[cos_val, -sin_val, 0],
                                    [sin_val, cos_val, 0],
                                    [0, 0, 1]])
        
        rotated_point_cloud = np.dot(point_cloud, rotation_matrix)
        
        # Assuming the trajectory is a sequence of positions, apply the same rotation.
        rotated_trajectory = np.dot(trajectory, rotation_matrix)
        
        return {'point_cloud': rotated_point_cloud, 'trajectory': rotated_trajectory}

class RandomScale(object):
    """
    Randomly scale the point cloud.
    """
    def __call__(self, sample):
        point_cloud, trajectory = sample['point_cloud'], sample['trajectory']
        
        scale = np.random.uniform(0.8, 1.2)
        scaled_point_cloud = point_cloud * scale
        scaled_trajectory = trajectory * scale  # Assuming the trajectory scales with the point cloud
        
        return {'point_cloud': scaled_point_cloud, 'trajectory': scaled_trajectory}
