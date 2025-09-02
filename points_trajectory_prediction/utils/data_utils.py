import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import h5py
import utils
# import dgl
from sklearn.neighbors import NearestNeighbors

def get_random_float():
    idx = torch.rand(1).cuda()
    if utils.is_dist_avail_and_initialized():
        dist.broadcast(idx, src=0)
    return idx.item()


def get_random_index(total_len):
    idx = torch.randint(0, total_len, (1, )).cuda()
    # print("Rank {}: {} before broadcasting".format(dist.get_rank(), idx), force=True)
    if utils.is_dist_avail_and_initialized():
        dist.broadcast(idx, src=0)
    # print("Rank {}: {} after broadcasting".format(dist.get_rank(), idx), force=True)
    return int(idx.item())


def get_random_number_between_a_and_b(a, b):
    rank = dist.get_rank()
    idx = torch.randint(a, b, (1, )).cuda()
    # print("Rank {}: {} before broadcasting".format(dist.get_rank(), idx), force=True)
    if utils.is_dist_avail_and_initialized():
        dist.broadcast(idx, src=0)
    # print("Rank {}: {} after broadcasting".format(dist.get_rank(), idx), force=True)
    return int(idx.item())


def get_non_overlap_nums_between_0_and_b(b, num):
    rank = dist.get_rank()
    
    idx = torch.randperm(b)[:num].cuda()
    # print("Rank {}: {} before broadcasting".format(dist.get_rank(), idx), force=True)
    if utils.is_dist_avail_and_initialized():
        dist.broadcast(idx, src=0)
    # print("Rank {}: {} after broadcasting".format(dist.get_rank(), idx), force=True)
    return idx.cpu().tolist()


def cls_accuracy(predictions, targets, threshold=0.5):
    # Convert predictions to binary classification format
    pred_cls_binary = (predictions < threshold).float()
    true_cls_binary = (targets == 0).float()

    # Find the index of the first '0' in targets (gt)
    gt_first_zero_idx = (true_cls_binary.cumsum(dim=1) == 1).float().argmax(dim=1)
    
    # If not found, set the index to the last position
    gt_first_zero_idx[gt_first_zero_idx == 0] = targets.size(1) - 1
    
    # Find the index of the first < threshold in predictions (pred)
    pred_first_below_threshold_idx = (pred_cls_binary.cumsum(dim=1) == 1).float().argmax(dim=1)
    
    # If not found, set the index to the last position
    pred_first_below_threshold_idx[pred_first_below_threshold_idx == 0] = predictions.size(1) - 1

    # Check if the index is the same for both pred and gt
    correct_predictions = (gt_first_zero_idx == pred_first_below_threshold_idx)

    # Compute the accuracy
    accuracy = correct_predictions.float().mean().item()
    
    return accuracy


def llm_process(llm_model, tokenizer, descriptions):
    llm_inputs = tokenizer(descriptions, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        llm_outputs = llm_model(**llm_inputs)

    hidden_states = llm_outputs.hidden_states

    average_features = average_tokens(llm_inputs['attention_mask'], hidden_states[-1])

    return average_features


def llm_embedding(llm_model, tokenizer, descriptions):
    # descriptions = [description_encoding(name)["description"] for name in names]

    description_embeddings = llm_process(llm_model, tokenizer, descriptions)


    return description_embeddings       # [batch_size, feature_dim]


def average_tokens(mask, feature):      
    mask = mask.unsqueeze(-1)

    x_masked = feature * mask
    valid_lengths = mask.sum(dim=1)

    valid_lengths = torch.where(valid_lengths == 0, torch.ones_like(valid_lengths), valid_lengths)
    averaged_features = x_masked.sum(dim=1) / valid_lengths

    return averaged_features



def description_encoding(input_name, is_mirror=False):
    # Define the mappings for clothing attributes
    type_mapping = {'D': 'Dress', 'T': 'Tops', 'P': 'Pants', 'S': 'Skirt'}
    property_mapping = {'S': 'Short', 'L': 'Long', 'H': 'Hooded', 'C': 'Collar', 'N': 'No-Collar'}
    sleeve_mapping = {'G': 'Gallus', 'T': 'Tube', 'L': 'Long-Sleeve', 'S': 'Short-Sleeve', 'N':'No-Sleeve'}
    extra_mapping = {'S': ' ', 'C': 'FrontClose', 'O': 'FrontOpen'}

    # Define the folding methods based on clothing types
    sleeveless_folding = ['DLG', 'DLNS', 'DLT', 'DSNS', 'SL', 'SS', 'TCNC', 'TCNO', 'THNC', 'TNNC']
    short_sleeve_folding = ['DLSS', 'DSSS', 'TCSC', 'TNSC']
    long_sleeve_folding = ['DLLS', 'DSLS', 'TCLC', 'TCLO', 'THLC', 'THLO', 'TNLC']
    pants_folding = ['PL', 'PS']

    # Extract information from the input string
    parts = input_name.split('_')
    cloth_code = parts[0]
    # cloth_type_name = parts[1]
    action = int(parts[-1].replace('action', ''))

    # Determine the type of clothing
    cloth_type = type_mapping.get(cloth_code[0], 'Unknown')
    description_parts = [cloth_type]

    # Determine the properties of the clothing
    if len(cloth_code) > 1:
        description_parts.append(property_mapping.get(cloth_code[1], ''))

    if len(cloth_code) > 2:
        description_parts.append(sleeve_mapping.get(cloth_code[2], ''))

    if len(cloth_code) > 3:
        description_parts.append(extra_mapping.get(cloth_code[3], ''))

    # Generate the main description
    description = ', '.join(filter(None, description_parts))

    # Determine folding method based on the code
    if cloth_code in sleeveless_folding:
        fold_description = "fold bottom-up"
    elif cloth_code in short_sleeve_folding:
        fold_description = ["fold left sleeve", "fold right sleeve", "fold bottom-up"][action]
    elif cloth_code in long_sleeve_folding:
        fold_description = ["fold left sleeve", "fold right sleeve", "fold bottom-up"][action]
    elif cloth_code in pants_folding:
        fold_description = ["fold right pant leg", "Drag right pant leg", "fold bottom-up"][action]
    else:
        fold_description = "unknown folding method"

    if is_mirror:
        fold_description = fold_description.replace("left", "temp").replace("right", "left").replace("temp", "right")

    # Combine description with folding method
    final_description = f"{description}, {fold_description}"

    # Create result dictionary
    result = {
        'stage': action,
        'description': final_description
    }
    return result


def normalize_point_cloud_trajectory(trajectory):
    """
    Normalize a batch of point cloud trajectories
    :param trajectory: Tensor of shape [batch_size, num_points, traj_steps, dim]
    :return: normalized_trajectory: Tensor of the same shape, normalized
    """
    # Compute the centroid of the entire trajectory (all points, all frames) for each batch
    centroid = trajectory.mean(dim=(1, 2), keepdim=True)  # [batch_size, 1, 1, dim]
    
    # Center the trajectory
    trajectory_centered = trajectory - centroid  # [batch_size, num_points, traj_steps, dim]
    
    # Compute the max distance from the origin of the centered trajectory for each batch
    # Step 1: Calculate the Euclidean distance for each point
    distances = torch.sqrt((trajectory_centered ** 2).sum(dim=-1))  # [batch_size, num_points, traj_steps]
    
    # Step 2: Find the maximum distance across all points and trajectory steps
    max_distance, _ = distances.max(dim=1, keepdim=True)  # [batch_size, 1, traj_steps]
    max_distance, _ = max_distance.max(dim=2, keepdim=True)  # [batch_size, 1, 1]
    
    # Scale the trajectory
    normalized_trajectory = trajectory_centered / max_distance.unsqueeze(-1)  # [batch_size, num_points, traj_steps, dim]
    
    return normalized_trajectory




def random_rotation_matrix(fixed, degree):
    # angles = (torch.rand(3) - 0.5) * torch.pi / 2
    if fixed:
        anglex = degree[0]
        angley = degree[1]
        anglez = degree[2]
        angles = torch.tensor([anglex, angley, anglez])
    else:
        anglex = (get_random_float() - 0.5) * 2 * degree[0] 
        angley = (get_random_float() - 0.5) * 2 * degree[1]
        anglez = (get_random_float() - 0.5) * 2 * degree[2]
        angles = torch.tensor([anglex, angley, anglez])
        

    cos_x, sin_x = torch.cos(angles[0]), torch.sin(angles[0])
    cos_y, sin_y = torch.cos(angles[1]), torch.sin(angles[1])
    cos_z, sin_z = torch.cos(angles[2]), torch.sin(angles[2])

    rotation_x = torch.tensor([
        [1, 0, 0],
        [0, cos_x, -sin_x],
        [0, sin_x, cos_x]
    ])
    rotation_y = torch.tensor([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]
    ])
    rotation_z = torch.tensor([
        [cos_z, -sin_z, 0],
        [sin_z, cos_z, 0],
        [0, 0, 1]
    ])
    rotation_matrix = rotation_z @ rotation_y @ rotation_x
    
    return rotation_matrix

def apply_rotation(points, rotation_matrix):
    return points @ rotation_matrix.T

def points_occlusion(points, trajectory, occlusion_ratio_range=(0, 1/6), target_num_points=2048):
    batch_size, num_points, _ = points.shape
    
    reshaped_points = []
    adjusted_trajectory = []
    
    for i in range(batch_size):
        min_coords = points[i].min(dim=0)[0]
        max_coords = points[i].max(dim=0)[0]
        max_length = (max_coords - min_coords).max().item()
        
        occlusion_diameter = torch.rand(1).item() * (occlusion_ratio_range[1] - occlusion_ratio_range[0]) + occlusion_ratio_range[0]
        occlusion_radius = (occlusion_diameter * max_length) / 2
        
        center_point = points[i][torch.randint(0, num_points, (1,)).item(), :]  
        
        distances = torch.norm(points[i] - center_point, dim=1)
        
        occlusion_indices = distances < occlusion_radius
        
        remaining_points = points[i][~occlusion_indices] 
        remaining_trajectory = trajectory[i][~occlusion_indices]
        
        num_remaining_points = remaining_points.shape[0]
        
        if num_remaining_points < target_num_points:
            additional_indices = torch.randint(0, num_remaining_points, (target_num_points - num_remaining_points,))
            additional_points = remaining_points[additional_indices]
            additional_trajectory = remaining_trajectory[additional_indices]
            
            reshaped_points_batch = torch.cat([remaining_points, additional_points], dim=0)
            adjusted_trajectory_batch = torch.cat([remaining_trajectory, additional_trajectory], dim=0)
        else: 
            selected_indices = torch.randperm(num_remaining_points)[:target_num_points]
            reshaped_points_batch = remaining_points[selected_indices]
            adjusted_trajectory_batch = remaining_trajectory[selected_indices]
        
        reshaped_points.append(reshaped_points_batch)
        adjusted_trajectory.append(adjusted_trajectory_batch)
    
    reshaped_points = torch.stack(reshaped_points)
    adjusted_trajectory = torch.stack(adjusted_trajectory)
    
    return reshaped_points, adjusted_trajectory


if __name__ == '__main__':
    # data_path = "/data2/chaonan/points-traj-prediction/data/data.h5"
    data_path = "/data2/chaonan/points-traj-prediction/data/point_cloud_samples.h5"
    cnt = 0
    with h5py.File(data_path, 'r+') as file:
        for group_name in file:
            if cnt % 50 == 0:
                print('Now processing: ', cnt)
            cnt += 1
            group = file[group_name]
            if 'points' in group:
                points_data = group['points'][:]
                enhanced_points_data = add_feature(points_data)
                if 'points' in group:
                    del group['points']
                group.create_dataset('points', data=enhanced_points_data)
