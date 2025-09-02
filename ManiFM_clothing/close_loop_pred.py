import torch
import torch.nn as nn
import os
from .networks import ManiFM
import numpy as np
import random
from omegaconf import OmegaConf
from safetensors.torch import load_model, save_file
from pathlib import Path
import trimesh
import viser
import json
import argparse
from .utils.pred_utils import (
    load_hand_pointcloud_and_normals,
    load_object_point_cloud_and_normal_rigid,
    load_object_point_cloud_and_normal_clothes,
    load_object_point_cloud_and_normal_mpm,
    vis_pc_heatmap,
    vis_vector,
    point_cloud_nms,
    pointcloud_motion_to_wrench,
    load_pcd_traj,
    compute_normals
)
from .utils.data_utils import from_wrench_to_contact_force
from collections import defaultdict


def set_seed(num=666):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed_all(num)


demo_configs = {
    "microwave": {
        "object_type": "rigid",
    },
    "rope": {
        "object_type": "mpm",
    },
    "Tshirt": {
        "object_type": "clothes",
    },
    "plasticine": {
        "object_type": "mpm",
    },
}

POINT_THRESHOLD = 0.5
FORCE_THRESHOLD = 1.0

def ManiFM_model(pcd1,pcd2,mask=None):
    config_file_path = "ManiFM_clothing/configs/test.json"
    cfg = OmegaConf.load(config_file_path)
    set_seed(cfg.train.seed)
    
    demo_name = "cup_mask"
    gpu_number = 0
    checkpoint_path = "ManiFM_clothing/checkpoints/200k_204epc_model.safetensors"
    pred_time = 160
    
    pp_file_path = Path("ManiFM_clothing/")
    dataset_directory_path = Path.joinpath(pp_file_path, "dataset")
    test_demo_data = np.load(os.path.join(dataset_directory_path, f"test_demo_data/{demo_name}.pkl"), allow_pickle=True)
    object_type = "clothes" 
        
    device = torch.device(f"cuda:{gpu_number}")
    model = ManiFM(cfg.model, device).cuda(gpu_number)
    checkpoint = torch.load(
        'ManiFM_clothing/checkpoints/checkpoint0313_24.pth',
        weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    # load_model(model, str(checkpoint_path))

    
    model.eval()

    scale = 1.0
    normal = np.array(compute_normals(pcd1))
    motion1 = pcd2 - pcd1
    input_object_point_cloud, offset_, norm_offset_ = (
        load_object_point_cloud_and_normal_clothes(
            pcd1,
            normal,
            point_motion=motion1,
            scale=scale
        )
    )
    with torch.no_grad():
        """return"""
        input_hand_point_cloud = load_hand_pointcloud_and_normals(test_demo_data["robot_name"])
        predicted_contact_points_heatmap, contact_forces = model.infer(
            input_hand_point_cloud.unsqueeze(0).repeat(pred_time, 1, 1,).cuda(gpu_number),
            input_object_point_cloud.unsqueeze(0).repeat(pred_time, 1, 1,).cuda(gpu_number),
        )
        pred_contact_points_heatmap = predicted_contact_points_heatmap.detach().cpu().numpy()  # shape=(b, 2048, 1)
        pred_contact_force_map = contact_forces.detach().cpu().numpy()  # shape=(b, 2048, 3)

    data_point_cloud = input_object_point_cloud[:, :3].detach().cpu().numpy() # shape=(2048, 3)
    # server = viser.ViserServer(port=8024)
    data_list = []
    to_show_cnt = 0
    for i in range(pred_time):
        if (pred_contact_points_heatmap[i, :, 0].max() - pred_contact_points_heatmap[i, :, 0].min())<=1e-6:
            continue
        normalized_hmap = (pred_contact_points_heatmap[i, :, 0] - pred_contact_points_heatmap[i, :, 0].min()) / (pred_contact_points_heatmap[i, :, 0].max() - pred_contact_points_heatmap[i, :, 0].min())
        
        if "Hand" in test_demo_data["robot_name"]:
            filter_radius = 0.02 / test_demo_data["object_scale"]
        elif "Arm" in test_demo_data["robot_name"]:
            filter_radius = 0.2

        if mask is not None:
            pred_contact_points_heatmap[i, :, 0] *= mask


        filtered_points, keep_idx = point_cloud_nms(
            data_point_cloud,
            pred_contact_points_heatmap[i, :, 0],
            filter_radius,
        )
        

        
        force_mesh_list = []
        for idx, pid in enumerate(keep_idx):
            if pid != 2048: 
                contact_point = data_point_cloud[pid]
                force_vector = pred_contact_force_map[i, pid]
                force_mesh_list.append(vis_vector(contact_point, force_vector, length=0.3, cyliner_r=0.03, color=[255, 255, 20, 255]))
                data = {
                    "prediction_index": i,
                    "contact_point": pcd1[pid].tolist(),
                    "force_vector": force_vector.tolist(),
                }

                data_list.append(data)

    data = {
        "prediction_index": -1,
        "contact_point": [0.0,0.0,0.0],
        "force_vector": [0.0,0.0,0.0],
    }

    data_list.append(data)                
    def vector_distance(v1, v2):
        return np.linalg.norm(np.array(v1) - np.array(v2))

    def are_vectors_close(v1, v2, threshold):
        return vector_distance(v1, v2) < threshold

    similar_groups = defaultdict(list)
    for entry in data_list:
        contact_point = entry["contact_point"]
        force_vector = entry["force_vector"]
        matched = False
        for group_key, group_items in similar_groups.items():
            ref_point = group_items[0]["contact_point"]
            ref_force = group_items[0]["force_vector"]
            if are_vectors_close(contact_point, ref_point, POINT_THRESHOLD) and are_vectors_close(force_vector, ref_force, FORCE_THRESHOLD):
                if np.linalg.norm(np.array(force_vector)) > 0.1:
                    similar_groups[group_key].append(entry)
                    similar_groups[group_key][0]["contact_point"] = np.array(similar_groups[group_key][0]["contact_point"])+(np.array(contact_point) - np.array(ref_point)) / (len(similar_groups[group_key]))
                    similar_groups[group_key][0]["force_vector"] = np.array(similar_groups[group_key][0]["force_vector"])+(np.array(force_vector) - np.array(ref_force)) / (len(similar_groups[group_key]))
                    matched = True
                    break
            
        
        if not matched:
            similar_groups[(tuple(contact_point), tuple(force_vector))].append(entry)
            similar_groups[(tuple(contact_point), tuple(force_vector))].append(entry)
            
    sorted_groups = sorted(similar_groups.values(), key=len, reverse=True)
    if len(sorted_groups) <= 0:
        similar_groups = defaultdict(list)
        for entry in data_list:
            contact_point = entry["contact_point"]
            force_vector = entry["force_vector"]
            matched = False
            for group_key, group_items in similar_groups.items():
                ref_point = group_items[0]["contact_point"]
                ref_force = group_items[0]["force_vector"]
                if are_vectors_close(contact_point, ref_point, POINT_THRESHOLD) and are_vectors_close(force_vector, ref_force, FORCE_THRESHOLD):
                    if np.linalg.norm(np.array(force_vector)) > 0.0:
                        similar_groups[group_key].append(entry)
                        similar_groups[group_key][0]["contact_point"] = np.array(similar_groups[group_key][0]["contact_point"])+(np.array(contact_point) - np.array(ref_point)) / (len(similar_groups[group_key]))
                        similar_groups[group_key][0]["force_vector"] = np.array(similar_groups[group_key][0]["force_vector"])+(np.array(force_vector) - np.array(ref_force)) / (len(similar_groups[group_key]))
                        matched = True
                        break
                
            
            if not matched:
                similar_groups[(tuple(contact_point), tuple(force_vector))].append(entry)
                similar_groups[(tuple(contact_point), tuple(force_vector))].append(entry)
    sorted_groups = sorted(similar_groups.values(), key=len, reverse=True)
    if len(sorted_groups) >= 2:
        top_two_groups = sorted_groups[:2]
    elif len(sorted_groups) == 1:
        top_two_groups = [sorted_groups[0], sorted_groups[0]] 
    else:
        top_two_groups = [data]  
        

    contact_points = []
    force_vectors = []
    for group in top_two_groups:
        print(len(group))
        avg_contact_point = np.mean([entry["contact_point"] for entry in group], axis=0)
        avg_force_vector = min(group, key=lambda entry: np.linalg.norm(np.array(entry["contact_point"])-avg_contact_point))
        contact_points.append(avg_contact_point)
        force_vectors.append(np.array(avg_force_vector["force_vector"]))
   
    return contact_points,force_vectors
    
if __name__ == "__main__":
    pcd_path = 'data/chenhn_data/eval_ep121/cloth0/output.txt'

    pcd_list, normal_list, motion_list, scale = load_pcd_traj(pcd_path)

    print(ManiFM_model(pcd_list[5],pcd_list[10]))
    