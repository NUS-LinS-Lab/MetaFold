import matplotlib.pyplot as plt
import numpy as np
import os
import h5py

def visualize_point_cloud_trajectories(points, trajectories):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', s=1, alpha=0.5, label='Initial Point Cloud')
    final_positions = trajectories[:, -1, :]
    ax.scatter(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], color='green', s=1, alpha=0.5, label='Final Point Cloud Position')
    colors = ['red', 'gold', 'purple', 'orange', 'cyan']
    for i, idx in enumerate(np.random.choice(points.shape[0], 5, replace=False)):
        ax.plot(trajectories[idx, :, 0], trajectories[idx, :, 1], trajectories[idx, :, 2], color=colors[i], linewidth=2, label=f'Trajectory {i+1}')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-6, 0])
    ax.set_zlim([-3, 3]) 
    # ax.view_init(elev=0, azim=90)
    ax.legend()
    plt.show()
    plt.savefig('./vis_pc.png')

data_path = 'data/data.h5'
with h5py.File(data_path, 'r') as data_h5:
    data = data_h5['DLNS_Dress001']
    points = data['points'][:]
    trajectories = data['trajectories'][:]
    visualize_point_cloud_trajectories(points, trajectories)

viser_enabled = False
if viser_enabled:
    import viser
    server = viser.ViserServer()
    server.add_point_cloud(
        "/init state",
        points=points,
        point_size=0.03,
        point_shape='circle',
        colors=np.array([64,224,208]),
    )

    server.add_point_cloud(
        "/middle state",
        points=trajectories[:, 10, :],
        point_size=0.03,
        point_shape='circle',
        colors=np.array([135,206,250]),
    )


    server.add_point_cloud(
        "/final state",
        points=trajectories[:, -1, :],
        point_size=0.03,
        point_shape='circle',
        colors=np.array([255,127,80]),
    )

    import time 
    while True:
        time.sleep(0.01)