import torch
import sys
sys.path.append(".")
from networks import ManiFM
from dataset import RigidBodyDataset, ClothDataset, MPMDataset
import numpy as np
import random
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator
import ipdb
# from utils.visualize import update_pred_force, update_pred_heatmap_pc
import viser
import matplotlib
import matplotlib.cm as cm
import warnings
import trimesh 
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import KDTree


from safetensors.torch import load_model
from pathlib import Path
import trimesh
import viser
from utils.pred_utils import (
    load_hand_pointcloud_and_normals,
    load_object_point_cloud_and_normal_rigid,
    load_object_point_cloud_and_normal_clothes,
    load_object_point_cloud_and_normal_mpm,
    vis_pc_heatmap,
    vis_vector,
    point_cloud_nms,
    pointcloud_motion_to_wrench,
)
from utils.data_utils import from_wrench_to_contact_force


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

def update_pred_force(server, pc, contact_point_id, force_map) -> None:
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
    
    
def update_pred_heatmap_pc(server, pc, hmap) -> None:
    """
    Draw the predicted heatmap and point cloud.
    
    Args:
    - pc: The point cloud. np, shape [N, 3].
    - hmap: The heatmap. np, shape [N].
    """
    colormap_pred = cm.get_cmap('plasma')
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

def load_dataset_list(cfg):
    '''
    remember to add new args if update the config file'''
    dataset_list = []
    info_list = []
    if OmegaConf.is_list(cfg.dir.data_dir.rigid_body):
        # try:
        if len(cfg.dir.data_dir.rigid_body) > 0:
            rigid_body_dataset = RigidBodyDataset(pkl_file_directory=cfg.dir.data_dir.rigid_body, load_ratio=cfg.dir.load_ratio, generate_new_wrench=True, use_gaussian_map_ratio=cfg.train.use_gaussian_map_ratio, augment_part_seg_indicator_ratio=cfg.train.augment_part_seg_indicator_ratio, use_region=cfg.dir.use_region, use_physics=cfg.dir.use_physics, num_object=cfg.dir.num_object, num_palm_pose=cfg.dir.num_palm_pose, num_motion=cfg.dir.num_motion, use_scale=cfg.dir.use_scale,  remove_pc_num=cfg.dir.remove_pc_num, remove_pc_prob=cfg.dir.remove_pc_prob, noisy_upper_level=cfg.dir.noisy_upper_level, random_flip_normal_upper_level=cfg.dir.random_flip_normal_upper_level)
            dataset_list.append(rigid_body_dataset)
            info_list.append(f"[info] Rigid body dataset size: {len(rigid_body_dataset)}")
        # except:
        #     info_list.append(f"Missing rigid_body dataset")
    if OmegaConf.is_dict(cfg.dir.data_dir.rigid_body):
        try:
            if len(cfg.dir.data_dir.rigid_body.force_closure.path) > 0:
                force_closure_dataset = RigidBodyDataset(pkl_file_directory=cfg.dir.data_dir.rigid_body.force_closure.path, load_ratio=cfg.dir.load_ratio, generate_new_wrench=False, use_gaussian_map_ratio=cfg.train.use_gaussian_map_ratio, augment_part_seg_indicator_ratio=cfg.train.augment_part_seg_indicator_ratio, use_region=cfg.dir.use_region, use_physics=cfg.dir.use_physics, num_object=cfg.dir.data_dir.rigid_body.force_closure.num_object, num_palm_pose=cfg.dir.data_dir.rigid_body.force_closure.num_palm_pose, num_motion=cfg.dir.data_dir.rigid_body.force_closure.num_motion, use_scale=cfg.dir.use_scale,  remove_pc_num=cfg.dir.remove_pc_num, remove_pc_prob=cfg.dir.remove_pc_prob, noisy_upper_level=cfg.dir.noisy_upper_level, random_flip_normal_upper_level=cfg.dir.random_flip_normal_upper_level)
                dataset_list.append(force_closure_dataset)
                info_list.append(f"[info] Rigid body/Force closure dataset size: {len(force_closure_dataset)}")
        except:
            info_list.append(f"Missing force_closure dataset")
            
        try:
            if len(cfg.dir.data_dir.rigid_body.leap_hand.path) > 0:
                leap_hand_dataset = RigidBodyDataset(pkl_file_directory=cfg.dir.data_dir.rigid_body.leap_hand.path, load_ratio=cfg.dir.load_ratio, generate_new_wrench=True, use_gaussian_map_ratio=cfg.train.use_gaussian_map_ratio, augment_part_seg_indicator_ratio=cfg.train.augment_part_seg_indicator_ratio, use_region=cfg.dir.use_region, use_physics=cfg.dir.use_physics, num_object=cfg.dir.data_dir.rigid_body.leap_hand.num_object, num_palm_pose=cfg.dir.data_dir.rigid_body.leap_hand.num_palm_pose, num_motion=cfg.dir.data_dir.rigid_body.leap_hand.num_motion, use_scale=cfg.dir.use_scale, remove_pc_num=cfg.dir.remove_pc_num, remove_pc_prob=cfg.dir.remove_pc_prob, noisy_upper_level=cfg.dir.noisy_upper_level, random_flip_normal_upper_level=cfg.dir.random_flip_normal_upper_level)
                dataset_list.append(leap_hand_dataset)
                info_list.append(f"[info] Rigid body/Leap Hand dataset size: {len(leap_hand_dataset)}")
        except:
            info_list.append(f"Missing leap_hand dataset")
        
        try:
            if len(cfg.dir.data_dir.rigid_body.kinova3f_hand.path) > 0:
                kinova3f_hand_dataset = RigidBodyDataset(pkl_file_directory=cfg.dir.data_dir.rigid_body.kinova3f_hand.path, load_ratio=cfg.dir.load_ratio, generate_new_wrench=True, use_gaussian_map_ratio=cfg.train.use_gaussian_map_ratio, augment_part_seg_indicator_ratio=cfg.train.augment_part_seg_indicator_ratio, use_region=cfg.dir.use_region, use_physics=cfg.dir.use_physics, num_object=cfg.dir.data_dir.rigid_body.kinova3f_hand.num_object, num_palm_pose=cfg.dir.data_dir.rigid_body.kinova3f_hand.num_palm_pose, num_motion=cfg.dir.data_dir.rigid_body.kinova3f_hand.num_motion, use_scale=cfg.dir.use_scale,  remove_pc_num=cfg.dir.remove_pc_num, remove_pc_prob=cfg.dir.remove_pc_prob, noisy_upper_level=cfg.dir.noisy_upper_level, random_flip_normal_upper_level=cfg.dir.random_flip_normal_upper_level)
                dataset_list.append(kinova3f_hand_dataset)
                info_list.append(f"[info] Rigid body/Kinova3f hand dataset size: {len(kinova3f_hand_dataset)}")
        except:
            info_list.append(f"Missing kinova3f_hand dataset")
            
        try:
            if len(cfg.dir.data_dir.rigid_body.panda_hand.path) > 0:
                panda_hand_dataset = RigidBodyDataset(pkl_file_directory=cfg.dir.data_dir.rigid_body.panda_hand.path, load_ratio=cfg.dir.load_ratio, generate_new_wrench=True, use_gaussian_map_ratio=cfg.train.use_gaussian_map_ratio, augment_part_seg_indicator_ratio=cfg.train.augment_part_seg_indicator_ratio, use_region=cfg.dir.use_region, use_physics=cfg.dir.use_physics, num_object=cfg.dir.data_dir.rigid_body.panda_hand.num_object, num_palm_pose=cfg.dir.data_dir.rigid_body.panda_hand.num_palm_pose, num_motion=cfg.dir.data_dir.rigid_body.panda_hand.num_motion, use_scale=cfg.dir.use_scale, remove_pc_num=cfg.dir.remove_pc_num, remove_pc_prob=cfg.dir.remove_pc_prob, noisy_upper_level=cfg.dir.noisy_upper_level, random_flip_normal_upper_level=cfg.dir.random_flip_normal_upper_level)
                dataset_list.append(panda_hand_dataset)
                info_list.append(f"[info] Rigid body/Panda hand dataset size: {len(panda_hand_dataset)}")
        except:
            info_list.append(f"Missing panda_hand dataset")
    
    # try:
    if len(cfg.dir.data_dir.cloth) > 0:
        cloth_dataset = ClothDataset(cloth_file_directory=cfg.dir.data_dir.cloth, use_gaussian_map_ratio=cfg.train.use_gaussian_map_ratio, augment_part_seg_indicator_ratio=cfg.train.augment_part_seg_indicator_ratio, use_region=cfg.dir.use_region, use_physics=cfg.dir.use_physics, use_scale=cfg.dir.use_scale, noisy_upper_level=cfg.dir.noisy_upper_level, random_flip_normal_upper_level=cfg.dir.random_flip_normal_upper_level)
        dataset_list.append(cloth_dataset)
        info_list.append(f"[info] Cloth dataset size: {len(cloth_dataset)}")
    # except:
    #     info_list.append(f"Missing cloth dataset")
        
    # try:
    if len(cfg.dir.data_dir.mpm) > 0:
        mpm_dataset = MPMDataset(pkl_file_directory=cfg.dir.data_dir.mpm, load_ratio=1.0, use_gaussian_map_ratio=cfg.train.use_gaussian_map_ratio, augment_part_seg_indicator_ratio=cfg.train.augment_part_seg_indicator_ratio, use_region=cfg.dir.use_region, use_physics=cfg.dir.use_physics, use_scale=cfg.dir.use_scale, noisy_upper_level=cfg.dir.noisy_upper_level, random_flip_normal_upper_level=cfg.dir.random_flip_normal_upper_level)
        dataset_list.append(mpm_dataset)
        info_list.append(f"[info] Mpm dataset size: {len(mpm_dataset)}")
    # except:
    #     info_list.append(f"Missing mpm dataset")

    # try:
    #     if len(cfg.dir.data_dir.rope) > 0:
    #         rope_dataset = MPMDataset(pkl_file_directory=cfg.dir.data_dir.rope, load_ratio=1.0, use_gaussian_map_ratio=cfg.train.use_gaussian_map_ratio, augment_part_seg_indicator_ratio=cfg.train.augment_part_seg_indicator_ratio, use_region=cfg.dir.use_region, use_physics=cfg.dir.use_physics, use_scale=cfg.dir.use_scale, noisy_upper_level=cfg.dir.noisy_upper_level, random_flip_normal_upper_level=cfg.dir.random_flip_normal_upper_level)
    #         dataset_list.append(rope_dataset)
    #         info_list.append(f"[info] Rope dataset size: {len(rope_dataset)}")
    # except:
    #     info_list.append(f"Missing Rope dataset")
    return dataset_list, info_list

def test():
    cfg = OmegaConf.load("configs/test.json")

    accelerator = Accelerator(step_scheduler_with_optimizer=False)
    device = accelerator.device

    model = ManiFM(cfg.model, device)
    accelerator.load_state("logs/log_diffcloth_gt/models/model_0")
    model.eval()

    dataset_list, info_list = load_dataset_list(cfg)
    for info in info_list:
        print(info) 

    assert len(dataset_list) > 0, "dataset_list is empty"
    test_dataset = ConcatDataset(dataset_list)
    test_loader = DataLoader(test_dataset, batch_size=cfg.test.batch_size, shuffle=False) 

    predictions = []
    actuals = []

    model = model.to(device)
    with torch.no_grad():  # Disable gradient computation
        for data in test_loader:
            print(data.keys())
            ipdb.set_trace()
            # inputs= 
            output = model(data["input_hand_point_cloud"].to(device), data["input_object_point_cloud"].to(device), data["mix_heatmap"].unsqueeze(-1).to(device))
            
            print(output.keys())
            print(output['contacts_object'].shape, output['forces_object'].shape)
            # Convert output probabilities to predicted class (max probability)
            preds = torch.argmax(output, dim=1)
            predictions.extend(preds.view(-1).cpu().numpy())
            actuals.extend(labels.view(-1).cpu().numpy())

def mytest():
    cfg = OmegaConf.load("configs/test.json")

    gpu_number = 0
    checkpoint_path = "checkpoints/200k_204epc_model.safetensors"

    device = torch.device(f"cuda:{gpu_number}")
    model = ManiFM(cfg.model, device).cuda(gpu_number)
    load_model(model, str(checkpoint_path))
    model.eval()

    dataset_list, info_list = load_dataset_list(cfg)
    for info in info_list:
        print(info) 

    assert len(dataset_list) > 0, "dataset_list is empty"
    test_dataset = ConcatDataset(dataset_list)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) 

    server = viser.ViserServer()

    predictions = []
    actuals = []

    model = model.to(device)
    with torch.no_grad():  # Disable gradient computation
        for data in test_loader:
            print(data.keys())
            print(data["input_object_point_cloud"].shape)
            ipdb.set_trace()
            # inputs= 
            output = model(data["input_hand_point_cloud"].to(device), data["input_object_point_cloud"].to(device), data["mix_heatmap"].unsqueeze(-1).to(device))
            
            print(output.keys())
            print(output['contacts_object'].shape, output['forces_object'].shape)
            # Convert output probabilities to predicted class (max probability)
            # preds = torch.argmax(output, dim=1)
            # predictions.extend(preds.view(-1).cpu().numpy())
            # actuals.extend(labels.view(-1).cpu().numpy())

            data_point_cloud = data["input_object_point_cloud"][0, :, :3].cpu().numpy()
            data_pred_contact_points_heatmap = output['contacts_object'][0, :, 0].cpu().numpy()
            data_pred_contact_force_map = output['forces_object'][0, :, :].cpu().numpy()
            data_gt_contact_points_heatmap = data["mix_heatmap"]

            filtered_points, keep_idx = point_cloud_nms(data_point_cloud, data_pred_contact_points_heatmap)

            # update_gt_heatmap_pc(data_point_cloud, )
            # update_gt_force(data_point_cloud, data_gt_contact_point_id, data_gt_contact_force_map)
            update_pred_heatmap_pc(server, data_point_cloud, data_pred_contact_points_heatmap)
            update_pred_force(server, data_point_cloud, keep_idx, data_pred_contact_force_map)


def main():
    # test()
    mytest()

if __name__ == "__main__":
    main()
