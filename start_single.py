import numpy as np
import torch
import os
import sys
import json
sys.path.append(os.getcwd())

from points_trajectory_prediction.trajectory_predictor import TrajectoryTransformer
from safetensors.torch import load_model,save_file
from points_trajectory_prediction.utils.data_utils import description_encoding, llm_embedding
from points_trajectory_prediction.utils.llm_utils import randomize_description_embed, description_matching
import math
import open3d as o3d





def closed_loop(env, model, cloth_config, cloth_name):
    device = torch.device("cuda")

    threshold = cloth_config["threshold"]  
    step = cloth_config["frames_per_step"]
    description_embed_list = cloth_config["description_embed_list"]
    env.world_pause()

    print("get_now_pcd_from_camera")
    now_real_pcd = env.get_now_pcd_from_camera()   
    output_pcd=[]
    output_pcd.append(now_real_pcd)
    cloth_inital_pcd = now_real_pcd
    last_pred_pcd = now_real_pcd  # for the first time
    last_real_pcd = now_real_pcd
    for i in range(cloth_config["stage_num"]):
        description_embed = description_embed_list[i].to(torch.float32).to(device)
        for _ in range(cloth_config["step_turn"][i]):
            times = 0
            stage_terminal = False
            first_contact = np.array([0, 0, 0])
            first_time = True
            begin = True
            print("DESCRIPTION",i)



            while True:
                if times >= math.ceil(cloth_config["frame_per_stage"][i]/(step)) or stage_terminal:
                    break
                

             
                target_num_points = 2048       
                pts = np.array(now_real_pcd)

                if pts.shape[0] > target_num_points:    # downsample
                    sampled_indices = np.random.choice(pts.shape[0], target_num_points, replace=False)
                    pts = pts[sampled_indices, :]
                elif pts.shape[0] < target_num_points:
                    sampled_indices = np.random.choice(pts.shape[0], target_num_points - pts.shape[0], replace=True)
                    pts = np.concatenate((pts, pts[sampled_indices, :]), axis=0)         

                if cloth_config["type"] in ["Pants"]:
                    pts[:,0]*=-1
                    pts[:,2]*=-1

                aligned_pcd = torch.from_numpy(pts).to(torch.float32)      # No alignment
                aligned_pcd = aligned_pcd.to(device)

                pcd_for_predict, transform_scale, transform_offset = normalize_point_cloud_trajectory(aligned_pcd.unsqueeze(0).unsqueeze(2))

                cpu_transform_scale = transform_scale.cpu().numpy()
                cpu_transform_offset = transform_offset.cpu().numpy()
                
                outputs_traj = predict(model, pcd_for_predict[:,:,0,:], description_embed, "").detach().cpu().numpy()

                aligned_pcd = pcd_for_predict[0,:,0,:].detach().cpu().numpy()

                pred_next_pcds = outputs_traj[0, :, :(step), :]


                masked_pred_next_pcds=pred_next_pcds[:, -1, :]
                hmask = None
                step_control = cloth_config["force_per_stage"][i]
                if (cloth_config["type"] in ["Short-sleeve","Long-sleeve","No-sleeve"] or (cloth_config["type"] in ["Pants"] and i==1)) and first_time is True and step_control==2:
                    extent = np.ptp(aligned_pcd[:, 2])  # Peak-to-peak range (max - min)
                    mask = aligned_pcd[:, 2] >=  (0.5 *extent+np.min(aligned_pcd[:, 2]))
                    hmask = np.zeros((aligned_pcd.shape[0]))
                    hmask[mask]=1
                    print(hmask.shape)
                if (cloth_config["type"] in ["Short-sleeve","Long-sleeve"] or (cloth_config["type"] in ["Pants"] and i==1)) and first_time is False and step_control==2:
                    extent = np.ptp(aligned_pcd[:, 2])  # Peak-to-peak range (max - min)
                    mask = first_contact
                    hmask = np.zeros((aligned_pcd.shape[0]))
                    hmask[mask]=1
                    print(hmask.shape)

                contacts, forces = ManiFM_model(aligned_pcd, masked_pred_next_pcds,hmask)
                

                if first_time:
                    first_time = False
                    first_contact = np.array([], dtype=int) 
                    for contact in contacts:
                        distances = np.linalg.norm(aligned_pcd - contact, axis=1)
                        indices = np.where(distances < 0.2)[0]
                        first_contact = np.concatenate((first_contact, indices))
                    print("LEN",len(first_contact))# Because real robot can only grasp the same point
                    
                contacts = np.array(contacts) * cpu_transform_scale[0, 0, 0] + cpu_transform_offset[0, 0]
                forces = np.array(forces) * cpu_transform_scale[0, 0, 0] 
                

                if cloth_config["type"] in ["Pants"]:
                    contacts[:,0]*=-1
                    contacts[:,2]*=-1
                    forces[:,0]*=-1
                    forces[:,2]*=-1

                env.control_to_next_step(begin,step_control,contacts, forces)   
                begin = False 
                last_pred_pcd = pred_next_pcds[:, -1, :]     
                last_real_pcd = now_real_pcd
                now_real_pcd = env.get_now_pcd_from_camera()       

                times += 1
            env.wait_step()
            now_real_pcd = env.get_now_pcd_from_camera() 
            output_pcd.append(now_real_pcd)
    
    # output meshs
    index = 0
    output_path = os.path.join("./isaac_sim/output", cloth_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created folder: {output_path}")
    base_name = "mesh"
    while True:
        folder_name = os.path.join(output_path, f"{base_name}{index}")
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Created folder: {folder_name}")
            break
        index += 1
  
    np.savetxt(folder_name+'/fin_pcd.txt', now_real_pcd)
    np.savetxt(folder_name+'/initial_pcd.txt', cloth_inital_pcd)
    for i,pcd in enumerate(output_pcd):
        np.savetxt(folder_name+'/step'+str(i)+'_pcd.txt', pcd)


if __name__=="__main__":


    device = torch.device('cuda:0')
    
    model = TrajectoryTransformer(
        input_dim=128,      # or the appropriate input dimension
        hidden_dim=256,     # make sure this matches the hidden dimension from the checkpoint
        output_dim=128,     # or the appropriate output dimension
        nhead=4,            # use the correct number of attention heads
        num_encoder_layers=4,  # set to 4 to match the checkpoint
        num_decoder_layers=4,  # adjust if necessary
        num_points=2048, 
        num_frames=21, 
        point_dim=3,
        device=device
    ).to(device)
        
    # input
    #--------------------------------------------------------------------------------------------
    embed_dict = torch.load('./data/description_embeddings_mirrored.pt')
    model.load_state_dict(torch.load('./data/model_1113_199.pth', map_location=device))

    descriptions = [
        "Please fold Short-Sleeve top from the left sleeve",
        "Please fold Short-Sleeve top from the right sleeve",
        "Please fold Short-Sleeve top from the bottom-up",
        # "fold the short pants from the left",
        # "fold the short pants bottom-up",
        # "fold the no-sleeve bottom-up"
    ]
    cloth_config_path = "./data/Cloth-Simulation/Configs/Short-sleeve.json"
    cloth_root = "./data/Cloth-Simulation/Assets/cloth_eval_data_all"
    cloth_name = "DSSS/DSSS_Dress385_action0"




    with open(cloth_config_path, 'r') as file:
        cloth_arg_config = json.load(file)

    if False:
        file_path = './data/chenhn_data/points-traj-prediction/description_embed_list.pt'
        description_embed_list = pre_process(model, descriptions)
        np_list = [description.cpu().numpy() for description in description_embed_list]

        torch.save(description_embed_list,file_path)
    
    else :

        description_embed_list = []

        
        for description in descriptions:
            description, description_embed = description_matching(description, embed_dict)
            description_embed_list.append(description_embed)
        
    

    from isaac_sim.physxDemo.cloth import cloth_main,normalize_point_cloud_trajectory,predict
    from isaac_sim.physxDemo.cloth import ClothDemoEnv
    from ManiFM_clothing.close_loop_pred import ManiFM_model

    
    
    cloth_config = {           
        "type": cloth_arg_config["type"],
        "scale": cloth_arg_config["scale"],
        "stage_num": cloth_arg_config["stage_num"],                           # total stage(action) num
        "frame_per_stage": cloth_arg_config["frame_per_stage"],               # Frame number in each stage(action)
        "force_per_stage": cloth_arg_config["force_per_stage"],               # Force number in each stage(action)
        "step_turn": cloth_arg_config["step_turn"],                           # Currently useless 
        "frames_per_step": cloth_arg_config["frames_per_step"],               # Frame number in each step(for close loop prediction)  
        "threshold": cloth_arg_config["threshold"],                           # Currently useless 
        "description_embed_list": description_embed_list,      
    }

    env = cloth_main(cloth_root, cloth_name, cloth_config["scale"])

    # env.stop()
    closed_loop(env, model, cloth_config, cloth_name)
    
    while 1:
        env.step()


