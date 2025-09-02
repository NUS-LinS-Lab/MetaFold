import numpy as np

def create_random_rotation_matrix():
    angle_x = np.random.uniform(0, 2 * np.pi)
    angle_y = np.random.uniform(0, 2 * np.pi)
    angle_z = np.random.uniform(0, 2 * np.pi)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

def create_random_rotation_matrix_x():
    angle_x = np.random.uniform(0, 2 * np.pi)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    return Rx

def create_random_rotation_matrix_y():
    angle_y = np.random.uniform(0, 2 * np.pi) 
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    return Ry

def create_random_rotation_matrix_z():
    angle_z = np.random.uniform(0, 2 * np.pi)
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    return Rz



def mirror_along_yz_plane(point_cloud_traj):
    """
    Mirror the point cloud along the yz-plane.
    
    :param point_cloud: numpy array of shape (num_points, num_frames, 3) representing the point cloud trajectory.
    :return: mirrored_point_cloud: numpy array with points mirrored along the yz-plane.
    """
    mirrored_point_cloud_traj = point_cloud_traj.copy()
    mirrored_point_cloud_traj[:, :, 0] = -point_cloud_traj[:, :, 0]  # Mirror x-coordinate
    return mirrored_point_cloud_traj

def rotate_point_cloud(point_cloud, R):
    rotated_point_cloud = np.dot(point_cloud, R.T)
    return rotated_point_cloud

def rotate_point_cloud_y(data, angle=np.pi):
    cos_val = np.cos(angle)
    sin_val = np.sin(angle)
    
    # Define the rotation matrix for y-axis
    rotation_matrix = np.array([
        [cos_val, 0, sin_val],
        [0, 1, 0],
        [-sin_val, 0, cos_val]
    ])
    
    # Apply the rotation
    rotated_data = np.dot(data, rotation_matrix)
    return rotated_data


if __name__ == "__main__":
    point_cloud = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    R = create_random_rotation_matrix()
    rotated_point_cloud = rotate_point_cloud(point_cloud, R)

    print(point_cloud)
    print(rotated_point_cloud)