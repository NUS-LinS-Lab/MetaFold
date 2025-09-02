import os 
import pickle 
import numpy as np
import viser
import matplotlib
import matplotlib.cm as cm
import warnings
import trimesh 
from scipy.spatial.transform import Rotation as R

warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

def normalize(x):
    '''
    Normalize the input vector. If the magnitude of the vector is zero, a small value is added to prevent division by zero.

    Parameters:
    - x (np.ndarray): Input vector to be normalized.

    Returns:
    - np.ndarray: Normalized vector.
    '''
    if len(x.shape) == 1:
        mag = np.linalg.norm(x)
        if mag == 0:
            mag = mag + 1e-10
        return x / mag
    else: 
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        return x / norms
    
def sample_transform_w_normals(new_palm_center, new_face_vector, sample_roll, ori_face_vector=np.array([1.0, 0.0, 0.0])):
    '''
    Compute the transformation matrix from the original palm pose to a new palm pose.
    
    Parameters:
    - new_palm_center (np.ndarray): The point of the palm center [x, y, z].
    - new_face_vector (np.ndarray): The direction vector representing the new palm facing direction.
    - sample_roll (float): The roll angle in range [0, 2*pi).
    - ori_face_vector (np.ndarray): The original direction vector representing the palm facing direction. Default is [1.0, 0.0, 0.0].

    Returns:
    - rst_transform (np.ndarray): A 4x4 transformation matrix.
    '''

    rot_axis = np.cross(ori_face_vector, normalize(new_face_vector))
    rot_axis = rot_axis / (np.linalg.norm(rot_axis) + 1e-16)
    rot_ang = np.arccos(np.clip(np.dot(ori_face_vector, new_face_vector), -1.0, 1.0))

    if rot_ang > 3.1415 or rot_ang < -3.1415: 
        rot_axis = np.array([1.0, 0.0, 0.0]) if not np.isclose(ori_face_vector, np.array([1.0, 0.0, 0.0])).all() else np.array([0.0, 1.0, 0.0])
    
    rot = R.from_rotvec(rot_ang * rot_axis).as_matrix()
    roll_rot = R.from_rotvec(sample_roll * new_face_vector).as_matrix()

    final_rot = roll_rot @ rot
    rst_transform = np.eye(4)
    rst_transform[:3, :3] = final_rot
    rst_transform[:3, 3] = new_palm_center
    return rst_transform

def vis_vector(start_point, vector, length=0.1, cyliner_r=0.003, color=[255, 255, 100, 245]):
    '''
    start_points: np.ndarray, shape=(3,)
    vectors: np.ndarray, shape=(3,)
    length: cylinder length 
    '''
    normalized_vector = normalize(vector)
    end_point = start_point + length * normalized_vector

    # create a mesh for the force
    force_cylinder = trimesh.creation.cylinder(radius=cyliner_r, 
                                               segment=np.array([start_point, end_point]))
    
    # create a mesh for the arrowhead
    cone_transform = sample_transform_w_normals(end_point, normalized_vector, 0, ori_face_vector=np.array([0.0, 0.0, 1.0]))
    arrowhead_cone = trimesh.creation.cone(radius=2*cyliner_r, 
                                           height=4*cyliner_r, 
                                           transform=cone_transform)
    # combine the two meshes into one
    force_mesh = force_cylinder + arrowhead_cone 
    force_mesh.visual.face_colors = color

    return force_mesh



from sklearn.neighbors import KDTree
def point_cloud_nms(pc, hmap, radius=0.2, heatmap_threshold_percentile=90, least_group_size=10, min_heatmap_value=0.5):
    
    '''
    Perform non-maximum suppression on the point cloud based on the heatmap.
    
    Args:
    - pc: np, shape=(n, 3)
    - hmap: np, shape=(n,)
    - radius: float, the radius to search the neighbors
    - heatmap_threshold_percentile: float, the percentile to filter the heatmap
    '''
    normalized_hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min())
    
    heatmap_threshold = np.percentile(normalized_hmap, heatmap_threshold_percentile)
    selected_indices = np.where(normalized_hmap > heatmap_threshold)[0]
    selected_points = pc[selected_indices]
    selected_heatmap_values = normalized_hmap[selected_indices]

    tree = KDTree(selected_points)
    indices = tree.query_radius(selected_points, r=radius)

    keep_mask = np.zeros(len(selected_points), dtype=bool)

    for i, ind in enumerate(indices):
        if selected_heatmap_values[i] >= selected_heatmap_values[ind].max() and len(ind) >= least_group_size and selected_heatmap_values[i] > min_heatmap_value:
            keep_mask[i] = True

    filtered_points = selected_points[keep_mask]
    filtered_points = np.unique(filtered_points, axis=0)
    keep_idx_in_sel = np.where(keep_mask)[0]
    keep_idx_in_ori = selected_indices[keep_idx_in_sel]
    

    return filtered_points, keep_idx_in_ori



def update_gt_heatmap_pc(pc, hmap) -> None:
    """
    Draw the ground truth heatmap and point cloud.
    
    Args:
    - pc: The point cloud. np, shape [N, 3].
    - hmap: The heatmap. np, shape [N].
    """
    normalized_hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min())
    hmap_colored = colormap_gt(normalized_hmap)  
    hmap_rgb = hmap_colored[:, :3]
    hmap_rgb_uint8 = (hmap_rgb * 255).astype('uint8')
    
    server.add_point_cloud(
        "/pc_hmap_gt",
        points=pc,
        point_size=0.03,
        point_shape='circle',
        colors=hmap_rgb_uint8,
    )
    
def update_gt_force(pc, contact_point_id, force_map) -> None:
    '''
    Draw the ground truth force arrows.
    
    Args:
    - pc: The point cloud. np, shape [N, 3].
    - contact_point_id: The contact point id. np, shape [4].
    - force_map: The force map. np, shape [4, 3].
    '''
    force_mesh_list = []
    for idx, pid in enumerate(contact_point_id):
        if pid != 2048:
            contact_point = pc[pid]
            force_vector = force_map[idx]
            force_mesh_list.append(vis_vector(contact_point, force_vector, length=0.2, cyliner_r=0.02, color=[255, 255, 0, 255]))
    server.add_mesh_trimesh("gt_force_mesh", 
                            trimesh.Scene(force_mesh_list).dump(concatenate=True))
    
    
def update_pred_force(pc, contact_point_id, force_map) -> None:
    '''
    Draw the ground truth force arrows.
    
    Args:
    - pc: The point cloud. np, shape [N, 3].
    - contact_point_id: The contact point id. np, shape [m].
    - force_map: The force map. np, shape [N, 3].
    '''
    force_mesh_list = []
    for idx, pid in enumerate(contact_point_id):
        if pid != 2048:
            contact_point = pc[pid] + np.array([-2.5, 0.0, 0])
            force_vector = force_map[pid]
            force_mesh_list.append(vis_vector(contact_point, force_vector, length=0.2, cyliner_r=0.02, color=[255, 255, 0, 255]))
    server.add_mesh_trimesh("pred_force_mesh", 
                            trimesh.Scene(force_mesh_list).dump(concatenate=True))
    
    
def update_pred_heatmap_pc(pc, hmap) -> None:
    """
    Draw the predicted heatmap and point cloud.
    
    Args:
    - pc: The point cloud. np, shape [N, 3].
    - hmap: The heatmap. np, shape [N].
    """
    normalized_hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min())
    hmap_colored = colormap_pred(normalized_hmap)  
    hmap_rgb = hmap_colored[:, :3]
    hmap_rgb_uint8 = (hmap_rgb * 255).astype('uint8')
    server.add_point_cloud(
        "/pc_hmap_pred",
        points=pc + np.array([-2.5, 0.0, 0]),
        point_size=0.03,
        point_shape='circle',
        colors=hmap_rgb_uint8,
    )
    markdown_blurb = server.add_gui_markdown(
        content=f"data_id = {data_id}",
    )

@gui_reset_scene.on_click
def _(_) -> None:
    """Reset the scene when the reset button is clicked."""
    global data_id
    data_id += 1 

    data_point_cloud = data["point_cloud"][data_id]
    data_point_normal = data["point_normal"][data_id]
    data_pred_contact_points_heatmap = data["pred_contact_points_heatmap"][data_id].squeeze(1)
    data_pred_contact_force_map = data["pred_contact_force_map"][data_id]
    data_gt_contact_point_id = data["gt_contact_point_id"][data_id]
    data_gt_contact_points_heatmap = data["gt_contact_points_heatmap"][data_id]
    data_gt_contact_force_map = data["gt_contact_force_map"][data_id]
    
    filtered_points, keep_idx = point_cloud_nms(data_point_cloud, data_pred_contact_points_heatmap)
    
    update_gt_heatmap_pc(data_point_cloud, data_gt_contact_points_heatmap)
    update_gt_force(data_point_cloud, data_gt_contact_point_id, data_gt_contact_force_map)
    update_pred_heatmap_pc(data_point_cloud, data_pred_contact_points_heatmap)
    update_pred_force(data_point_cloud, keep_idx, data_pred_contact_force_map)
    
    
    
if __name__ == '__main__':
    data_path = "clothes_test_data.pkl"
    data = pickle.load(open(data_path, "rb"))
    '''
    --------- Data keys ---------
    print(data.keys())
    dict_keys(['cloth_name', 'point_cloud', 'point_normal', 'pred_contact_points_heatmap', 'pred_contact_force_map', 'gt_contact_point_id', 'gt_contact_points_heatmap', 'gt_contact_force_map'])

    --------- Data shape ---------
    for k, v in data.items():
        if isinstance(v, list):
            print(k, len(v))
        elif isinstance(v, np.ndarray):
            print(k, v.shape)
            
    cloth_name 640
    point_cloud (640, 2048, 3)
    point_normal (640, 2048, 3)
    pred_contact_points_heatmap (640, 2048, 1)
    pred_contact_force_map (640, 2048, 3)
    gt_contact_point_id (640, 4)
    gt_contact_points_heatmap (640, 2048)
    gt_contact_force_map (640, 4, 3)
    '''

    data_len = len(data["cloth_name"])

    server = viser.ViserServer()
    gui_reset_scene = server.add_gui_button("Next data")


    colormap_gt = cm.get_cmap('viridis')
    colormap_pred = cm.get_cmap('plasma')

    data_id = -1

    import time 
    while True:
        time.sleep(0.01)
        


