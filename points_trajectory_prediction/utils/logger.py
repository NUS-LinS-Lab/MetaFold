import logging
import matplotlib.pyplot as plt
import numpy as np

def setup_logger(name, log_file, level=logging.INFO):
    """Set up and get a logger.

    Parameters:
    - name: The name of the logger.
    - log_file: The path to store the log file.
    - level: The level of logging.

    Returns:
    - A configured logger.
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler for logging to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Console handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_point_cloud_trajectories(points, targets, mask, outputs, save_path):
    fig = plt.figure(figsize=(20, 10))

    selected_indices = np.random.choice(points.shape[0], 5, replace=False)

    colormap = plt.cm.Blues

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', s=1, alpha=0.5, label='Point Cloud')

    for idx in selected_indices:
        valid_targets = targets[idx][mask[idx].astype(bool)]
        num_points = valid_targets.shape[0]
        color_gradient = colormap(np.linspace(0.2, 1, num_points))  

        for j in range(num_points - 1):
            ax1.plot(valid_targets[j:j+2, 0], valid_targets[j:j+2, 1], valid_targets[j:j+2, 2],
                     color=color_gradient[j], linewidth=2)
    
    ax1.set_xlabel('X Axis')
    ax1.set_ylabel('Y Axis')
    ax1.set_zlabel('Z Axis')
    ax1.set_xlim([min(-1, min(points[:, 0])), max(1, max(points[:, 0]))])
    ax1.set_ylim([min(-1, min(points[:, 1])), max(1, max(points[:, 1]))])
    ax1.set_zlim([min(-1, min(points[:, 2])), max(1, max(points[:, 2]))])
    ax1.title.set_text('Point Cloud and Target Trajectories')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', s=1, alpha=0.5, label='Point Cloud')
    
    for idx in selected_indices:
        valid_outputs = outputs[idx][mask[idx].astype(bool)]
        num_points = valid_outputs.shape[0]
        color_gradient = colormap(np.linspace(0.2, 1, num_points))  

        for j in range(num_points - 1):
            ax2.plot(valid_outputs[j:j+2, 0], valid_outputs[j:j+2, 1], valid_outputs[j:j+2, 2],
                     color=color_gradient[j], linewidth=2)
    
    ax2.set_xlabel('X Axis')
    ax2.set_ylabel('Y Axis')
    ax2.set_zlabel('Z Axis')
    ax2.set_xlim([min(-1, min(points[:, 0])), max(1, max(points[:, 0]))])
    ax2.set_ylim([min(-1, min(points[:, 1])), max(1, max(points[:, 1]))])
    ax2.set_zlim([min(-1, min(points[:, 2])), max(1, max(points[:, 2]))])
    ax2.title.set_text('Point Cloud and Output Trajectories')

    plt.savefig(save_path)
    plt.close(fig)


def visualize_pred_point_cloud(points, targets, mask, outputs, save_path):
    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(outputs[:, 0, 0], outputs[:, 0, 1], outputs[:, 0, 2], color='blue', s=1, alpha=0.5, label='Point Cloud')
    ax1.set_xlabel('X Axis')
    ax1.set_ylabel('Y Axis')
    ax1.set_zlabel('Z Axis')
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1, 1])
    ax1.set_zlim([-1, 1]) 
    ax1.legend(loc='upper left')
    ax1.title.set_text('Predict Point Cloud 1st frame')

    valid_frame_indices = np.where(mask.any(axis=0))[0]
    last_valid_frame = valid_frame_indices[-1] if len(valid_frame_indices) > 0 else 20
    mid_frame = last_valid_frame // 2

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(outputs[:, mid_frame, 0], outputs[:, mid_frame, 1], outputs[:, mid_frame, 2], color='blue', s=1, alpha=0.5, label='Point Cloud')
    ax2.set_xlabel('X Axis')
    ax2.set_ylabel('Y Axis')
    ax2.set_zlabel('Z Axis')
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.set_zlim([-1, 1])
    ax2.legend(loc='upper left')
    ax2.title.set_text(f'Predict Point Cloud {mid_frame + 1}th frame')

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(outputs[:, last_valid_frame, 0], outputs[:, last_valid_frame, 1], outputs[:, last_valid_frame, 2], color='blue', s=1, alpha=0.5, label='Point Cloud')
    ax3.set_xlabel('X Axis')
    ax3.set_ylabel('Y Axis')
    ax3.set_zlabel('Z Axis')
    ax3.set_xlim([-1, 1])
    ax3.set_ylim([-1, 1])
    ax3.set_zlim([-1, 1])
    ax3.legend(loc='upper left')
    ax3.title.set_text(f'Predict Point Cloud {last_valid_frame + 1}th frame')

    plt.savefig(save_path)
    plt.close(fig)  