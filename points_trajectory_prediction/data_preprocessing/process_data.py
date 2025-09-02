import os
import h5py
import numpy as np
from normalize import normalize_point_cloud_trajectory
from rotation import create_random_rotation_matrix, create_random_rotation_matrix_y, rotate_point_cloud, rotate_point_cloud_y, mirror_along_yz_plane
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.category import clothing_encoding, clothing_decoding
from utils.data_utils import description_encoding, llm_embedding

from transformers import AutoTokenizer, LlamaForCausalLM
import torch


output_file = 'data/all_data.h5'
output_group = 'data'

root_dirs = ['../cloth_traj_data']
dir_subnames = [[
'DLG', 'DLNS', 'DLT', 'DSNS', 'SL', 'SS', 'TCNC', 'TCNO', 'THNC', 'TNNC',       # No-sleeve old 0
'PL', 'PS',     # Pants old 0
'DLLS', 'DSLS', 'TCLC', 'TCLO', 'THLC', 'THLO', 'TNLC', # Long sleeve 2
'DLSS', 'DSSS', 'TNSC', 'TCSC',     # Short sleeve 2
]] # all data

print(os.getcwd())

cloth_cnt = 0
# data_cnt = 0
Rotation_enabled = False

llm_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(llm_model_id, use_fast=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

llm_model = LlamaForCausalLM.from_pretrained(llm_model_id, output_hidden_states=True)
llm_model.resize_token_embeddings(len(tokenizer))

with h5py.File(output_file, 'w') as output_h5:
    for dir_subname, root_dir in zip(dir_subnames, root_dirs):
        for subname in dir_subname:
            origin_dir = os.path.join(root_dir, subname)
            for folder_name in os.listdir(origin_dir):
                folder_path = os.path.join(origin_dir, folder_name)

                if subname not in ['DLG', 'DLNS', 'DLT', 'DSNS', 'SL', 'SS', 'TCNC', 'TCNO', 'THNC', 'TNNC', 'PL', 'PS']:
                    if folder_name[-1] != '2':
                        continue
                else:
                    if folder_name[-1] != '0':
                        continue

                cloth_cnt += 1
                # if cnt > 1:
                #     break
                print('Now processing ', folder_path)
                if os.path.isdir(folder_path):
                    # base_group_name = folder_name

                    # group_name = subname + '_' + folder_name      # DLNS_DLNS_... (in case the cloth name differ)
                    # group = output_h5.create_group(group_name)

                    group_name_L = f"{subname}_L_old_{folder_name}"  # Original data (L)
                    group_L = output_h5.create_group(group_name_L)
                    if subname not in ['DLG', 'DLNS', 'DLT', 'DSNS', 'SL', 'SS', 'TCNC', 'TCNO', 'THNC', 'TNNC']:
                        group_name_R = f"{subname}_R_old_{folder_name}"  # Mirrored data (R)
                        group_R = output_h5.create_group(group_name_R)

                    # group_name2 = 'tmp'
                    # group2 = output_h5.create_group(group_name2)
                
                    for file_name in os.listdir(folder_path):
                        if file_name.endswith('.h5'):
                            file_path = os.path.join(folder_path, file_name)
                            with h5py.File(file_path, 'r') as input_h5:
                                for dataset_name in input_h5.keys():
                                    assert dataset_name == 'pcd_traj'
                                    input_dataset = input_h5[dataset_name]
                                    input_data = input_dataset[:] 

                                    traj_length = len(input_data)
                                    target_frames = 21
                                    frame_interval = traj_length // (target_frames - 1)
                                    frame_indices = np.arange(0, traj_length, frame_interval)
                                    frame_indices = np.append(frame_indices, traj_length - 1)
                                    input_data = input_data[frame_indices]

                                    target_num_points = 2048
                                    # print(input_data[0].shape)
                                    if len(input_data[0]) >= target_num_points:    # downsample
                                        sampled_indices = np.random.choice(len(input_data[0]), target_num_points, replace=False)
                                        sampled_pcd_traj = input_data[:, sampled_indices, :]
                                    else:       # upsample
                                        sampled_indices = np.random.choice(len(input_data[0]), target_num_points - len(input_data[0]), replace=True)
                                        sampled_pcd_traj = np.concatenate((input_data, input_data[:, sampled_indices, :]), axis=1)         

                                    sampled_pcd_traj = np.transpose(sampled_pcd_traj, (1, 0, 2))
                                    normalized_pcd_traj = normalize_point_cloud_trajectory(sampled_pcd_traj)

                                    if subname in ['PL', 'PS']:
                                        rotated_pcd_traj = rotate_point_cloud_y(normalized_pcd_traj, np.pi)
                                    else:
                                        rotated_pcd_traj = normalized_pcd_traj      # no rotation


                                    description_L = description_encoding(folder_name, is_mirror=False)
                                    # description_embed_L = llm_embedding(llm_model, tokenizer, description_L['description'])
                                    description_embed_L = torch.zeros(1, 4096)
                                    print('     > description_L:', description_L['description'], '.    description_embed_L: ', description_embed_L)                                    

                                    group_L.create_dataset('points', data=rotated_pcd_traj[:, 0, :])
                                    group_L.create_dataset('trajectories', data=rotated_pcd_traj)
                                    group_L.create_dataset('description', data=description_L['description'])
                                    group_L.create_dataset('description_embed', data=description_embed_L)

                                    if subname not in ['DLG', 'DLNS', 'DLT', 'DSNS', 'SL', 'SS', 'TCNC', 'TCNO', 'THNC', 'TNNC']:
                                        mirrored_pcd_traj = mirror_along_yz_plane(rotated_pcd_traj)
                                        # print(rotated_pcd_traj[:3, :3, :])
                                        # print(mirrored_pcd_traj[:3, :3, :])
                                        
                                        description_R = description_encoding(folder_name, is_mirror=True)
                                        # description_embed_R = llm_embedding(llm_model, tokenizer, description_R['description'])
                                        description_embed_R = torch.zeros(1, 4096)
                                        print('     > description_R:', description_R['description'], '.    description_embed_R: ', description_embed_R)                                    

                                        group_R.create_dataset('points', data=mirrored_pcd_traj[:, 0, :])  # First frame point cloud
                                        group_R.create_dataset('trajectories', data=mirrored_pcd_traj)
                                        group_R.create_dataset('description', data=description_R['description'])
                                        group_R.create_dataset('description_embed', data=description_embed_R)



    print(output_h5.keys())
    print("h5keys num: ", len(list(output_h5.keys())))

print("Merge and preprocess success in {}".format(output_file))
print("Total cloth&action_num: ", cloth_cnt)
