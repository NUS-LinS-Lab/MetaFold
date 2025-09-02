import numpy as np

def normalize_point_cloud_trajectory(trajectory):         
    # assert 0        # Don't do it the whole trajectory
    """
    Normalize a batch of point cloud trajectories
    :param batch_data: [num_points, traj_steps, dim]
    :return: normalized_data
    """
    # Step 1: Extract the first frame
    first_frame = trajectory[:, 0, :]  # [num_points, 3]

    # Step 2: Compute the centroid of the first frame
    first_frame_centroid = first_frame.mean(axis=0, keepdims=True)  # [1, 3]

    # Step 3: Center the first frame
    first_frame_centered = first_frame - first_frame_centroid  # [num_points, 3]

    # Step 4: Compute the max distance from the origin in the first frame
    distances = np.sqrt(np.sum(first_frame_centered ** 2, axis=-1))  # [num_points]
    max_distance = np.max(distances)  # scalar, max distance in the first frame

    # Step 5: Scale the trajectory based on the first frame's max distance
    scaling_factor = max_distance  # scalar

    # Step 6: Apply the scaling and centering to the entire trajectory
    normalized_trajectory = (trajectory - first_frame_centroid) / scaling_factor  # [num_points, traj_steps, 3]

    return normalized_trajectory