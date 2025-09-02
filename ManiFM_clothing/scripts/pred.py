import torch
import torch.nn as nn
import os
import sys
sys.path.append(".")
from networks import ManiFM
import numpy as np
import random
from omegaconf import OmegaConf
from safetensors.torch import load_model, save_file
from pathlib import Path
import trimesh
import viser
import json
import argparse
from utils.pred_utils import (
    load_hand_pointcloud_and_normals,
    load_object_point_cloud_and_normal_rigid,
    load_object_point_cloud_and_normal_clothes,
    load_object_point_cloud_and_normal_mpm,
    vis_pc_heatmap,
    vis_vector,
    point_cloud_nms,
    pointcloud_motion_to_wrench,
    load_pcd_traj,
)
from utils.data_utils import from_wrench_to_contact_force


def generate_force_heatmap(pcd_t, pid_list, force_list, sigma=1.0, scale_factor=1.0):

    num_points = pcd_t.shape[0]
    heatmap = np.zeros((num_points, 3))

    for i, pid in enumerate(pid_list):
        distances = np.linalg.norm(pcd_t - pcd_t[pid], axis=1)
        
        gaussian_weights = np.exp(-(distances**2) / (2 * sigma**2))

        heatmap[:, 0] += force_list[i][0] * gaussian_weights * scale_factor  
        heatmap[:, 1] += force_list[i][1] * gaussian_weights * scale_factor  
        heatmap[:, 2] += force_list[i][2] * gaussian_weights * scale_factor  
    
    
    return heatmap


def generate_heatmap(pcd_t, pid_list, force_list, sigma=1.0, scale_factor=1.0):

    num_points = pcd_t.shape[0]
    heatmap = np.zeros(num_points) 

    for i, pid in enumerate(pid_list):
        force_magnitude = np.linalg.norm(force_list[i]) * scale_factor
        
        distances = np.linalg.norm(pcd_t - pcd_t[pid], axis=1)
        gaussian_weights = np.exp(-(distances**2) / (2 * sigma**2))
        
        heatmap += force_magnitude * gaussian_weights
    
    heatmap = heatmap / np.max(heatmap)
    
    return heatmap

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



if __name__ == "__main__":
    config_file_path = "configs/test.json"
    cfg = OmegaConf.load(config_file_path)
    set_seed(cfg.train.seed)
    
    demo_name = "cup_mask"
    gpu_number = 0
    checkpoint_path = "checkpoints/200k_204epc_model.safetensors"
    pred_time = 160
    
    
    pp_file_path = Path(__file__).parent.parent
    dataset_directory_path = Path.joinpath(pp_file_path, "dataset")
    test_demo_data = np.load(os.path.join(dataset_directory_path, f"test_demo_data/{demo_name}.pkl"), allow_pickle=True)
    object_type = "clothes"
    
    
    if object_type == "rigid":
        input_object_point_cloud, offset_, norm_offset_ = (
            load_object_point_cloud_and_normal_rigid(
                test_demo_data["down_sampled_begin_points"],
                test_demo_data["down_sampled_begin_normals"],
                point_motion=test_demo_data["point_motion"],
                scale=test_demo_data["object_scale"],
            )
        )
    elif object_type == "clothes":
        parser = argparse.ArgumentParser(description="Process some parameters.")
        parser.add_argument('--t', type=float, required=True, help='Value of parameter t')
        parser.add_argument('--c', type=int, required=True, help='Value of parameter c')
        args = parser.parse_args()

    elif object_type == "mpm":
        input_object_point_cloud, offset_, norm_offset_ = (
            load_object_point_cloud_and_normal_mpm(
                test_demo_data["down_sampled_begin_points"],
                test_demo_data["down_sampled_begin_normals"],
                point_motion=test_demo_data["point_motion"],
                scale=test_demo_data["object_scale"],
            )
        )
        
        
    device = torch.device(f"cuda:{gpu_number}")
    model = ManiFM(cfg.model, device).cuda(gpu_number)
    checkpoint = torch.load('checkpoint3.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    # load_model(model, str(checkpoint_path))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss_fn_heatmap = nn.MSELoss()
    loss_fn_force = nn.MSELoss()
    server = viser.ViserServer(port=8024)
    to_show_cnt0 = 0
    e_pcd_list = []
    e_normal_list = []
    e_motion_list = []
    e_scale = []
    data_point_clouds = {}
    data_gt_hms = {}
    data_gt_fs = {}
    batch = 15
    input_hand_point_cloud0 = load_hand_pointcloud_and_normals(test_demo_data["robot_name"])
    pass_list = [6,7,8,11,18,19,20]
    for epoch in range(0):
        for cloth_id in range(27):
            if cloth_id in pass_list:
                continue
            if epoch == 0:
                input_object_point_clouds = []
                gt_heatmaps = []
                gt_forces_list = []

                if cloth_id<9:
                    pcd_path = 'data/chenhn_data/eval_ep121_random/cloth'+str(int(cloth_id))+'/mesh/point_cloud.txt'
                else :
                    pcd_path = 'data/chenhn_data/eval_ep120/cloth'+str(int(cloth_id-9))+'/mesh/point_cloud.txt'
                pcd_list, normal_list, motion_list, scale = load_pcd_traj(pcd_path)

                for t in range(batch): 
                    if cloth_id<9:
                        with open("data/chenhn_data/eval_ep121_random/cloth"+str(int(cloth_id))+"/mesh/output_point_f"+str(t//5)+"-"+str(t%5)+".json", "r") as f:
                            data_list = json.load(f)
                    else: 
                        with open("data/chenhn_data/eval_ep120/cloth"+str(int(cloth_id-9))+"/mesh/output_point_f"+str(t//5)+"-"+str(t%5)+".json", "r") as f:
                            data_list = json.load(f)
                    pcd_t = pcd_list[t]
                    normal_t = normal_list[t]
                    pcd_next = pcd_list[t+5]
                    motion = pcd_next - pcd_t 
                    pid_list = []
                    force_list = []
                    for entry in data_list:
                        pid_list.append(np.argmin(np.linalg.norm(pcd_t - entry["contact_point"], axis=1)))
                        force_list.append(np.array(entry["force_vector"]))
                    
                    input_object_point_cloud0, offset_0, norm_offset_0 = (
                        load_object_point_cloud_and_normal_clothes(
                            pcd_t,
                            normal_t,
                            point_motion=motion,
                            scale=scale,
                        )
                    )
                    input_object_point_clouds.append(input_object_point_cloud0)

                    gt_heatmap_np = generate_heatmap(pcd_t, pid_list, force_list, sigma=0.1, scale_factor=0.7) 
                    gt_heatmap = torch.from_numpy(gt_heatmap_np).float()
                    gt_heatmaps.append(gt_heatmap.unsqueeze(-1))
                    
                    
                    gt_forces_np = generate_force_heatmap(pcd_t, pid_list, force_list, sigma=0.05, scale_factor=1.0)
                    gt_forces = torch.from_numpy(gt_forces_np).float()  
                    gt_forces_list.append(gt_forces)
                    
                input_object_point_clouds_t = torch.stack(input_object_point_clouds, dim=0)
                gt_heatmaps_t = torch.stack(gt_heatmaps, dim=0)
                gt_forces_list_t = torch.stack(gt_forces_list, dim=0)
                data_point_clouds[cloth_id] = input_object_point_clouds_t
                data_gt_hms[cloth_id] = gt_heatmaps_t
                data_gt_fs[cloth_id] = gt_forces_list_t
                

            else:
                input_object_point_clouds_t = data_point_clouds[cloth_id]
                gt_heatmaps_t = data_gt_hms[cloth_id]
                gt_forces_list_t = data_gt_fs[cloth_id]




            optimizer.zero_grad()  
            
            results = model(
                input_hand_point_cloud0.unsqueeze(0).repeat(batch, 1, 1,).cuda(gpu_number),
                input_object_point_clouds_t.cuda(gpu_number),
                gt_heatmaps_t.cuda(gpu_number)  
            )

            predicted_contact_points_heatmap = results['contacts_object'] 
            contact_forces = results['forces_object'] 

            loss_heatmap = loss_fn_heatmap(predicted_contact_points_heatmap, gt_heatmaps_t.cuda(gpu_number))
            loss_force = loss_fn_force(contact_forces, gt_forces_list_t.cuda(gpu_number))
            total_loss = loss_heatmap + loss_force

            total_loss.backward()

            optimizer.step()


    model.eval()
    pcd_path = 'data/chenhn_data/eval_ep120/cloth'+str(int(args.c))+'/output.txt'
    with open('output.json', 'w') as f:
        pass
    pcd_list, normal_list, motion_list, scale = load_pcd_traj(pcd_path)
    
    t = int(args.t)

    pcd1 = pcd_list[t]
    normal = normal_list[t]
    motion1 = pcd_list[t+5] - pcd_list[t]
    input_object_point_cloud, offset_, norm_offset_ = (
        load_object_point_cloud_and_normal_clothes(
            pcd1,
            normal,
            point_motion=motion1,
            scale=scale,
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
            
        filtered_points, keep_idx = point_cloud_nms(
            data_point_cloud,
            pred_contact_points_heatmap[i, :, 0],
            filter_radius,
        )
        
        if len(keep_idx) == 0 or len(keep_idx) > 4 or np.average(normalized_hmap) > 1.0 or np.percentile(normalized_hmap, 50) > 1.0: 
            print(len(keep_idx), np.average(normalized_hmap))
            
            continue
        
        if object_type == "rigid": 
            # if the object is rigid, we can optimize the force and see whether the contact points and forces can produce the target point motion
            try: 
                wrench = pointcloud_motion_to_wrench(test_demo_data["down_sampled_begin_points"], test_demo_data["point_motion"])
                prob, f_global_array = from_wrench_to_contact_force(
                    test_demo_data["down_sampled_begin_points"], 
                    test_demo_data["down_sampled_begin_normals"],
                    keep_idx, 0.1 * wrench / (np.linalg.norm(wrench) + 1e-8) )
                if prob > 0.1:  # fail to solve 
                    continue                    
                else:
                    force_mesh_list = []
                    for idx, pid in enumerate(keep_idx):
                        if pid != 2048:
                            contact_point = data_point_cloud[pid]
                            force_vector = f_global_array[idx]
                            force_mesh_list.append(vis_vector(contact_point, force_vector, length=0.3, cyliner_r=0.03, color=[255, 255, 20, 255]))

                    to_show_cnt += 1
            except Exception as e:
                print(f"Fail to solve the force for {i}-th prediction, exception: {e}")
                continue
        else: 
            force_mesh_list = []
            for idx, pid in enumerate(keep_idx):
                if pid != 2048: 
                    # pid = random.randint(0,2047)
                    contact_point = data_point_cloud[pid]
                    force_vector = pred_contact_force_map[i, pid]
                    force_mesh_list.append(vis_vector(contact_point, force_vector, length=0.3, cyliner_r=0.03, color=[255, 255, 20, 255]))
                    data = {
                        "prediction_index": i,
                        "contact_point": pcd_list[0][pid].tolist(),
                        "force_vector": (pcd_list[10][pid] - pcd_list[5][pid]).tolist(),
                        "motion0": (pcd_list[5][pid] - pcd_list[0][pid]).tolist(),
                        "motion1": (pcd_list[10][pid] - pcd_list[5][pid]).tolist(),
                        "motion2": (pcd_list[15][pid] - pcd_list[10][pid]).tolist(),
                        "motion3": (pcd_list[20][pid] - pcd_list[15][pid]).tolist()
                    }

                    data_list.append(data)

            to_show_cnt += 1
    with open('output.json', 'a') as f:
        json.dump(data_list, f, indent=4)
        f.write('\n')



    