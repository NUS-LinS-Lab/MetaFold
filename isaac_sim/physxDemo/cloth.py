import numpy as np
from isaacsim import SimulationApp
import torch
import sys
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

sys.path.append("isaac_sim")
simulation_app = SimulationApp({"headless": False})
import numpy as np
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.franka import Franka
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.franka import KinematicsSolver
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.prims import delete_prim
from Env.Utils.transforms import euler_angles_to_quat
import torch
from Env.Utils.transforms import quat_diff_rad
from Env.env.BaseEnv import BaseEnv
from Env.Garment.Garment import Garment
from Env.Robot.Franka.MyFranka import MyFranka
from Env.env.Control import Control, AttachmentBlock
from Env.Config.GarmentConfig import GarmentConfig
from Env.Config.FrankaConfig import FrankaConfig
from Env.Config.DeformableConfig import DeformableConfig
from scipy.optimize import minimize
from pxr import UsdGeom, Gf, Sdf, Usd, Vt, UsdPhysics, PhysxSchema
import omni.usd
import math
import json
from Env.Config.GarmentConfig import GarmentConfig
from scipy.spatial.transform import Rotation as R
from omni.isaac.core.utils.stage import get_current_stage
import omni.isaac.core.utils.prims as prims_utils
from shapely.ops import triangulate, unary_union

import importlib.util
import open3d as o3d
from ManiFM_clothing.close_loop_pred import ManiFM_model
import os
import time
from safetensors.torch import load_model,save_file
from points_trajectory_prediction.utils.data_utils import description_encoding, llm_embedding
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModel
from isaac_sim.physxDemo.metrics import get_current_covered_area,extend_point_cloud
from scipy.spatial import ConvexHull
import cv2
class ClothDemoEnv(BaseEnv):
    def __init__(self,garment_config_list:list):
        BaseEnv.__init__(self,garment=True)

        self.tough = True
        self.tough_cnt = 0
        self.last_dis = 0
        self.step_counter = -100
        self.current_prediction_index = 0
        self.num_steps = 120
        self.scale_factor = 0.45
        self.closest_point_index = [-1,-1]
        self.contacts = np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
        self.forces = np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
        self.target_pos = np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
        self.total_force = np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
        self.g = np.array([0.0, 0.0, 0.38])
        self.control_data = []

        self.center = np.array([0.0, 0.0, 0.0])
        self.begin = True
        self.offset_x_output = 0
        self.offset_y_output = 0 
        self.scale_x_output = 1.0
        self.scale_y_output = 1.0
        self.offset_x = 0
        self.offset_y = 0 
        self.scale_x = 1
        self.scale_y = 1
        self.offset_z = 0 
        self.scale_z = 1
        self.mesh_data = []
        self.coverage_trial = np.array([0.0, 0.0, 0.0, 0.0])
        self.rotation=torch.from_numpy(np.array([0.707, 0.707, 0.0, 0.0]))
        self.garment = []
        for garment_config in garment_config_list:
            garment=Garment(self.world,garment_config)
            garment.set_mass(10)
            self.garment.append(garment)

        self.cur_index = -1

        self.rigid_group_path="/World/Collision/Rigid_group"
        self.rigid_group = UsdPhysics.CollisionGroup.Define(self.stage, self.rigid_group_path)
        self.filter_rigid=self.rigid_group.CreateFilteredGroupsRel()
        self.robot_group_path="/World/Collision/robot_group"
        self.robot_group = UsdPhysics.CollisionGroup.Define(self.stage, self.robot_group_path)
        self.filter_robot = self.robot_group.CreateFilteredGroupsRel()
        self.garment_group_path="/World/Collision/Garment_group"
        self.garment_group = UsdPhysics.CollisionGroup.Define(self.stage, self.garment_group_path)
        self.filter_garment = self.garment_group.CreateFilteredGroupsRel()
        self.attach_group_path="/World/attach_group"
        self.attach_group = UsdPhysics.CollisionGroup.Define(self.stage, self.attach_group_path)
        self.filter_attach = self.attach_group.CreateFilteredGroupsRel()
        self.filter_robot.AddTarget(self.garment_group_path)
        self.filter_robot.AddTarget(self.rigid_group_path)
        self.filter_robot.AddTarget(self.attach_group_path)
        self.filter_garment.AddTarget(self.robot_group_path)
        # self.filter_garment.AddTarget(self.rigid_group_path)
        self.filter_garment.AddTarget(self.attach_group_path)
        self.filter_rigid.AddTarget(self.robot_group_path)
        # self.filter_rigid.AddTarget(self.garment_group_path)
        self.filter_rigid.AddTarget(self.attach_group_path)
        self.filter_attach.AddTarget(self.robot_group_path)
        self.filter_attach.AddTarget(self.garment_group_path)
        self.filter_attach.AddTarget(self.rigid_group_path)
        self.collectionAPI_robot = Usd.CollectionAPI.Apply(self.filter_robot.GetPrim(), "colliders")
        # for robot in self.robot:
        #     self.collectionAPI_robot.CreateIncludesRel().AddTarget(robot.get_prim_path())
        self.collectionAPI_garment = Usd.CollectionAPI.Apply(self.filter_garment.GetPrim(), "colliders")
        self.collectionAPI_garment.CreateIncludesRel().AddTarget(f"/World/Garment")
        for garment in self.garment:
            self.collectionAPI_garment.CreateIncludesRel().AddTarget(garment.garment_mesh_prim_path)
            self.collectionAPI_garment.CreateIncludesRel().AddTarget(garment.garment_prim_path)
            self.collectionAPI_garment.CreateIncludesRel().AddTarget(garment.particle_system_path)
        self.collectionAPI_attach = Usd.CollectionAPI.Apply(self.filter_attach.GetPrim(), "colliders")
        self.collectionAPI_attach.CreateIncludesRel().AddTarget("/World/Attachment")
        
        self.collectionAPI_rigid = Usd.CollectionAPI.Apply(self.filter_rigid.GetPrim(), "colliders")
        self.collectionAPI_rigid.CreateIncludesRel().AddTarget("/World/Avatar")
        
        self.attach = []
        self.attach.append(AttachmentBlock(world = self.world,robot=None,init_place=np.array([100.0, 0.0, 0.0]),collision_group=self.collectionAPI_attach))
        self.attach.append(AttachmentBlock(world = self.world,robot=None,init_place=np.array([100.0, 0.0, 0.0]),collision_group=self.collectionAPI_attach))
        self.attach.append(AttachmentBlock(world = self.world,robot=None,init_place=np.array([100.0, 0.0, 0.0]),collision_group=self.collectionAPI_attach,no=1))
        self.attach.append(AttachmentBlock(world = self.world,robot=None,init_place=np.array([100.0, 0.0, 0.0]),collision_group=self.collectionAPI_attach,no=1))
        self.attach.append(AttachmentBlock(world = self.world,robot=None,init_place=np.array([100.0, 0.0, 0.0]),collision_group=self.collectionAPI_attach,no=2))
        self.attach.append(AttachmentBlock(world = self.world,robot=None,init_place=np.array([100.0, 0.0, 0.0]),collision_group=self.collectionAPI_attach,no=2))
        self.attach.append(AttachmentBlock(world = self.world,robot=None,init_place=np.array([100.0, 0.0, 0.0]),collision_group=self.collectionAPI_attach,no=3))
        self.attach.append(AttachmentBlock(world = self.world,robot=None,init_place=np.array([100.0, 0.0, 0.0]),collision_group=self.collectionAPI_attach,no=3))
        for i in [4,5,6,7]:
            self.attach[4].block_prim.post_reset()


    def Transform_to_world(self,tmp):
        return np.array([-(tmp[0]*self.scale_x_output+self.offset_x_output)*self.scale_x+self.offset_x,(tmp[2]*self.scale_y_output+self.offset_y_output)*self.scale_y+self.offset_y,(tmp[1])*self.scale_z])
                
    def Transform_to_local(self,tmp):
        return np.array([(-(tmp[0]-self.offset_x)/self.scale_x-self.offset_x_output)/self.scale_x_output,(tmp[2]/self.scale_z),((tmp[1]-self.offset_y)/self.scale_y-self.offset_y_output)/self.scale_y_output])

 

    def get_now_pcd_from_camera(self):
        
        stage = get_current_stage()
        mesh_prim = UsdGeom.Mesh.Get(stage, "/World/Garment/garment/initial/mesh")
        world_transform = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(0)
        points_attr = mesh_prim.GetPointsAttr()
        points = points_attr.Get()
        pts = []
        world_points = []
        for pos in points:
            tmp = Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2]))
            post = np.array(world_transform.Transform(tmp))
            contact_point = self.Transform_to_local(post)
            pts.append(contact_point)
        
        return np.array(pts)
    
    def control_to_next_step(self,begin,step_control,controls,forces):
        self.contacts = controls
        self.forces = forces
        self.step_counter = 0
        
        finish = [False, False]
        step = 0
        self.begin = begin
        self.step_control = step_control

        if self.begin:
            stage = get_current_stage()
            mesh_prim = UsdGeom.Mesh.Get(stage, "/World/Garment/garment/initial/mesh")

            world_transform = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(0)
            local_transform = UsdGeom.Xformable(mesh_prim).GetLocalTransformation()


            points_attr = mesh_prim.GetPointsAttr()
            points = points_attr.Get()
            points_np = np.array(points)

        
         
        
        
        if self.step_counter==0:
            contacts = self.contacts
            forces = self.forces
            for i in range(self.step_control):
                contact_point = np.array(contacts[i])
                contacts[i] = self.Transform_to_world(contact_point)
                force_vector = np.array(forces[i]) 
                forces[i] = np.array([-self.scale_x*self.scale_x_output*force_vector[0],force_vector[2]*self.scale_y_output*self.scale_y,force_vector[1]*self.scale_z])/2
            
        self.world.pause()
        for i in range(self.step_control):


            if self.begin:
                
                contact_point = np.array(contacts[i])
                force_vector = np.array(forces[i]) 


                world_points = []
                for pos in points_np:
                    tmp = Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2]))
                    world_points.append(np.array(world_transform.Transform(tmp)))
                distances = np.linalg.norm(world_points-contact_point  , axis=1)
                self.closest_point_index[i] = np.argmin(distances)
            
                
                closest_point_index = self.closest_point_index[i]
                pos = points_np[closest_point_index] 
                tmp = Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2]))
                attach_pos = np.array(world_transform.Transform(tmp))


                if self.attach[i] is not None :
                    self.attach[i].detach()
                    self.attach[i] = None

                self.attach[i] = AttachmentBlock(world = self.world,robot=None,init_place=attach_pos,collision_group=self.collectionAPI_attach)
                
                
                if np.linalg.norm(force_vector)>0.01:
                    self.attach[i].attach(self.garment[0])

                
            block_handle=self.attach[i]
            attach_pos=np.array(block_handle.get_position())

            if self.step_counter==0:    
                force_vector = np.array(forces[np.argmin(np.linalg.norm(np.array(contacts-attach_pos) , axis=1))])
                
                self.target_pos[i] = attach_pos + force_vector*2
                self.target_pos[i][2]=min(self.target_pos[i][2],1.2)
                self.total_force[i] = force_vector
                

        self.world.play()

        for j in range(60):
            step+=1

            for i in range(self.step_control):
                
                block_velocity=torch.from_numpy((self.total_force[i]+self.g))
                orientation_ped=torch.zeros_like(block_velocity)
                cmd=torch.cat([block_velocity,orientation_ped],dim=-1)
                self.attach[i].set_velocities(cmd)


                ori_pos = self.attach[i].get_position()
                target_pos = self.target_pos[i]

                
                self.attach[i+2].set_position(ori_pos)
            
                
            self.world.step()
            self.world.step()   
            if j == 0:
                self.world.pause()
                cur_time = time.time()

                self.world.play()


            for i in range(self.step_control):
                if not finish[i]:
                    attach_pos = np.array(self.attach[i].get_position())
                    target =  np.array(self.target_pos[i]) - attach_pos
                    if np.linalg.norm(target) < 0.02:
                        finish[i] = True
                    else:
                        self.total_force[i] = np.linalg.norm(self.total_force[i]) * target / np.linalg.norm(target)
                else :
                    self.total_force[i] = np.array([0,0,0])
            if finish[0] and finish[1]:
                print(" STEP: ",step)
                break        
            self.begin = False
            self.step_counter += 1
        self.world.pause()

    def wait_step(self):
        self.contacts = np.zeros_like(self.contacts)
        self.forces = np.zeros_like(self.forces)
        for i in range(self.step_control):
            block_velocity=torch.from_numpy((self.forces[i]))
            orientation_ped=torch.zeros_like(block_velocity)
            cmd=torch.cat([block_velocity,orientation_ped],dim=-1)
            self.attach[i].set_velocities(cmd)

            self.attach[i+2].set_position(np.array([100.0, 0.0, 0.0]))
        print("WAIT")

        
        self.world.play()
        for i in range(50):
            self.world.step(render=True)
        self.world.pause()
        for i in range(2):
            if self.attach[i] is not None :
                self.attach[i].detach()
        self.world.play()
        for i in range(50):
            self.world.step(render=True)
        self.world.pause()

    def world_pause(self):
        self.world.pause()

    def world_delete_prim(self,path):
        delete_prim(path)
    def find_nearest_index(self,point):
        point=np.array(point).astype(np.float32)
        dist=np.linalg.norm(self.init_position[:,:3]-point,axis=1)
        return np.argmin(dist)
    def get_keypoint_groups(self,xzy : np.ndarray):
        num = 16144
        pos = xzy.copy()
        if(pos.shape[0]<num):
            pos = extend_point_cloud(pos,num)
        else :
            swapped_pcd = pos.copy()
            swapped_pcd[:, [1, 2]] = swapped_pcd[:, [2, 1]]
            pos = swapped_pcd
        self.initial_coverage = get_current_covered_area(pos)
        self.init_position = xzy.copy()
        self.initial_area_rect = self.calculate_rectangle_ratio(xzy)
        x = xzy[:, 0]
        y = xzy[:, 2]

        cloth_height = float(np.max(y) - np.min(y))
        cloth_width = float(np.max(x) - np.min(x))
        
        max_ys, min_ys = [], []
        num_bins = 40
        x_min, x_max = np.min(x),  np.max(x)
        mid = (x_min + x_max)/2
        lin = np.linspace(mid, x_max, num=num_bins)
        for xleft, xright in zip(lin[:-1], lin[1:]):
            if(len(np.where((xleft < x) & (x < xright))[0])>0):
                max_ys.append(-1 * y[np.where((xleft < x) & (x < xright))].min())
                min_ys.append(-1 * y[np.where((xleft < x) & (x < xright))].max())

        #plot the rate of change of the shirt height wrt x
        diff = np.array(max_ys) - np.array(min_ys)
        roc = diff[1:] - diff[:-1]

        #pad beginning and end
        begin_offset = num_bins//5
        end_offset = num_bins//10
        roc[:begin_offset] = np.max(roc[:begin_offset])
        roc[-end_offset:] = np.max(roc[-end_offset:])
        
        #find where the rate of change in height dips, it corresponds to the x coordinate of the right shoulder
        right_x = (x_max - mid) * (np.argmin(roc)/num_bins) + mid

        #find where the two shoulders are and their respective indices
        xzy_copy = xzy.copy()
        xzy_copy[np.where(np.abs(xzy[:, 0] - right_x) > 0.01), 2] = 10
        right_pickpoint_shoulder = np.argmin(xzy_copy[:, 2])
        right_pickpoint_shoulder_pos = xzy[right_pickpoint_shoulder, :]

        left_shoulder_query = np.array([-right_pickpoint_shoulder_pos[0], right_pickpoint_shoulder_pos[1], right_pickpoint_shoulder_pos[2]])
        left_pickpoint_shoulder = (np.linalg.norm(xzy - left_shoulder_query, axis=1)).argmin()
        left_pickpoint_shoulder_pos = xzy[left_pickpoint_shoulder, :]

        #top left and right points are easy to find
        pickpoint_top_right = np.argmax(x - y)
        pickpoint_top_left = np.argmax(-x - y)

        #to find the bottom right and bottom left points, we need to first make sure that these points are
        #near the bottom of the cloth
        pickpoint_bottom = np.argmax(y)
        diff = xzy[pickpoint_bottom, 2] - xzy[:, 2]
        idx = diff < 0.1
        locations = np.where(diff < 0.1)
        points_near_bottom = xzy[idx, :]
        x_bot = points_near_bottom[:, 0]
        y_bot = points_near_bottom[:, 2]

        #after filtering out far points, we can find the argmax as usual
        pickpoint_bottom_right = locations[0][np.argmax(x_bot + y_bot)]
        pickpoint_bottom_left = locations[0][np.argmax(-x_bot + y_bot)]

        self.bottom_right=pickpoint_bottom_right,
        self.bottom_left=pickpoint_bottom_left,
        self.top_right=pickpoint_top_right,
        self.top_left=pickpoint_top_left,
        self.right_shoulder=right_pickpoint_shoulder,
        self.left_shoulder=left_pickpoint_shoulder,
        

        # get middle point
        middle_point_pos = np.mean(self.init_position, axis=0)
        self.middle_point=self.find_nearest_index(middle_point_pos)

        # get left and right points
        middle_band=np.where(np.abs(self.init_position[:,2]-middle_point_pos[2])<0.1)
        self.left_x=np.min(self.init_position[middle_band,0])
        self.right_x=np.max(self.init_position[middle_band,0])
        self.left_point=self.find_nearest_index([self.left_x,0,-0.3])
        self.right_point=self.find_nearest_index([self.right_x,0,-0.3])

        # get top and bottom points
        x_middle_band=np.where(np.abs(self.init_position[:,0]-self.init_position[self.middle_point,0])<0.1)
        self.top_y=np.min(self.init_position[x_middle_band,2])
        self.bottom_y=np.max(self.init_position[x_middle_band,2])
        self.top_point=self.find_nearest_index([0,0,self.top_y])
        self.bottom_point=self.find_nearest_index([0,0,self.bottom_y])

        self.keypoint=[self.bottom_left,self.bottom_right,self.top_left,self.top_right,self.left_shoulder,self.right_shoulder,self.middle_point,self.left_point,self.right_point,self.top_point,self.bottom_point]
    
    def calculate_rectangle_ratio(self,point_cloud):
        points_2d = point_cloud[:, [0,2]]
        
        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]
        hull_points = np.array(hull_points, dtype=np.float32)
        rect = cv2.minAreaRect(hull_points)
        box = cv2.boxPoints(rect)
        width = np.linalg.norm(box[0] - box[1])
        height = np.linalg.norm(box[1] - box[2])
        area_rect = width * height
        
        return area_rect
    



def predict(model, pcd, description_embed, save_path):
    model.eval()
    with torch.no_grad():
        inputs = pcd
        outputs_off, _, _ = model(inputs, description_embed, mode='eval')
        outputs_traj = inputs.unsqueeze(2).repeat(1, 1, 21, 1) + outputs_off
        
        outputs_cpu = outputs_traj.cpu().detach().numpy()
        inputs_cpu = inputs.cpu().detach().numpy()
        targets_cpu = outputs_traj.cpu().detach().numpy()
        targets_cpu_tensor = outputs_traj.cpu()
        mask_cpu = torch.ones_like(targets_cpu_tensor[:, :, :, 0], dtype=torch.bool)
        visualize_point_cloud_trajectories(inputs_cpu[0],targets_cpu[0],mask_cpu[0],outputs_cpu[0], save_path='./vis_pc_eval_real.png')
        visualize_pred_point_cloud(inputs_cpu[0],targets_cpu[0],mask_cpu[0],outputs_cpu[0], save_path='./vis_pc2_eval_real.png')

        outputs_traj_save = outputs_traj.cpu().detach().numpy()
        if os.path.exists(save_path):
            np.savetxt(save_path, outputs_traj_save)

    return outputs_traj

def normalize_point_cloud_trajectory(trajectory):
    """
    Normalize the first frame of a batch of point cloud trajectories, and scale the entire trajectory
    accordingly, keeping the center and scale consistent with the first frame.
    
    :param trajectory: Tensor of shape [batch_size, num_points, traj_steps, dim]
    :return: normalized_trajectory: Tensor of the same shape, normalized based on the first frame
    """
    first_frame = trajectory[:, :, 0, :]  # [batch_size, num_points, dim]

    first_frame_centroid = first_frame.mean(dim=1, keepdim=True)  # [batch_size, 1, dim]
    
    first_frame_centered = first_frame - first_frame_centroid  # [batch_size, num_points, dim]
    
    distances = torch.sqrt((first_frame_centered ** 2).sum(dim=-1))  # [batch_size, num_points]
    max_distance, _ = distances.max(dim=1, keepdim=True)  # [batch_size, 1]
    
    scaling_factor = max_distance.unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1, 1]
    normalized_trajectory = (trajectory - first_frame_centroid.unsqueeze(2)) / scaling_factor  # Apply same scaling across all frames
    
   
    return normalized_trajectory, scaling_factor, first_frame_centroid



def visualize_point_cloud_trajectories(points, targets, mask, outputs, save_path):
    fig = plt.figure(figsize=(20, 10))

    selected_indices = np.random.choice(points.shape[0], 5, replace=False)

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', s=1, alpha=0.5, label='Point Cloud')
    colors = ['red', 'gold', 'purple', 'orange', 'cyan']
    for i, idx in enumerate(selected_indices):
        valid_targets = targets[idx]
        ax1.plot(valid_targets[:, 0], valid_targets[:, 1], valid_targets[:, 2], color=colors[i], linewidth=2, label=f'Target Trajectory {i+1}')
    ax1.set_xlabel('X Axis')
    ax1.set_ylabel('Y Axis')
    ax1.set_zlabel('Z Axis')
    ax1.set_xlim([min(-1, min(points[:, 0])), max(1, max(points[:, 0]))])  
    ax1.set_ylim([min(-1, min(points[:, 1])), max(1, max(points[:, 1]))]) 
    ax1.set_zlim([min(-1, min(points[:, 2])), max(1, max(points[:, 2]))])  
    ax1.legend(loc='upper left')
    ax1.title.set_text('Point Cloud and Target Trajectories')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', s=1, alpha=0.5, label='Point Cloud')
    for i, idx in enumerate(selected_indices):
        valid_outputs = outputs[idx]
        ax2.plot(valid_outputs[:, 0], valid_outputs[:, 1], valid_outputs[:, 2], color=colors[i], linewidth=2, label=f'Output Trajectory {i+1}')
    ax2.set_xlabel('X Axis')
    ax2.set_ylabel('Y Axis')
    ax2.set_zlabel('Z Axis')
    ax2.set_xlim([min(-1, min(points[:, 0])), max(1, max(points[:, 0]))])  
    ax2.set_ylim([min(-1, min(points[:, 1])), max(1, max(points[:, 1]))])  
    ax2.set_zlim([min(-1, min(points[:, 2])), max(1, max(points[:, 2]))])  
    ax2.legend(loc='upper left')
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

    last_valid_frame = 20
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
    masked_pred_next_pcds = outputs[:, mid_frame, :]
    extent = np.ptp(masked_pred_next_pcds[:, 2])  # Peak-to-peak range (max - min)
    mask = masked_pred_next_pcds[:, 2] >= ( 0.0 *extent+np.min(masked_pred_next_pcds[:, 2]))
    masked_pred_next_pcds = masked_pred_next_pcds[mask]
    print(masked_pred_next_pcds.shape[0])
    if masked_pred_next_pcds.shape[0] < 2048:
        sampled_indices = np.random.choice(masked_pred_next_pcds.shape[0], 2048 - masked_pred_next_pcds.shape[0], replace=True)
        masked_pred_next_pcds = np.concatenate((masked_pred_next_pcds, masked_pred_next_pcds[sampled_indices, :]), axis=0) 
                
    ax3.scatter(masked_pred_next_pcds[:,  0], masked_pred_next_pcds[:, 1], masked_pred_next_pcds[:,  2], color='blue', s=1, alpha=0.5, label='Point Cloud')
    
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

def cloth_main(cloth_root,cloth_name, scale = [0.5,0.5,0.5]):
    
    #initial obj
    #python /home/transfer/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/api/omni.kit.asset_converter/asset_usd_converter.py --folder data/chenhn_data/eval_ep121/cloth9

    garment_config_list=[]

    garment_config1=GarmentConfig(usd_path=os.path.join(cloth_root, cloth_name)+"/initial_obj.usd",
                                  visual_material_usd="./data/Cloth-Simulation/Assets/Material/linen_Beige.usd",
                                  scale = np.array(scale),
                                  pos = np.array([0.0,0.0,4.0]),
                                #   ori = np.array([0.7071068, 0.7071068, 0, 0]),
                                  ori = np.array([0.0,0.0,0.707,0.707]),
                                  particle_contact_offset=0.015)

    garment_config_list.append(garment_config1)

    env=ClothDemoEnv(garment_config_list=garment_config_list)
    
    env.reset()
    for i in range(500):
        env.step()

    pts = env.get_now_pcd_from_camera().copy()
    env.get_keypoint_groups(pts)
    
    

    # env.stop()

    return env

def cloth_next(env,cloth_root,cloth_name, scale = [0.5,0.5,0.5]):
    
    #initial obj
    #python /home/transfer/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/api/omni.kit.asset_converter/asset_usd_converter.py --folder data/chenhn_data/eval_ep121/cloth9

    garment_config_list=[]

    garment_config1=GarmentConfig(usd_path=os.path.join(cloth_root, cloth_name)+"/initial_obj.usd",
                                  visual_material_usd="./data/Cloth-Simulation/Assets/Material/linen_Beige.usd",
                                  scale = np.array(scale),
                                  pos = np.array([0.0,0.0,4.0]),
                                #   ori = np.array([0.7071068, 0.7071068, 0, 0]),
                                  ori = np.array([0.0,0.0,0.707,0.707]),
                                  particle_contact_offset=0.01)

    garment_config_list.append(garment_config1)

    env.garment = []
    for garment_config in garment_config_list:
        garment=Garment(env.world,garment_config)
        garment.set_mass(10)
        env.garment.append(garment)

    env.reset()
    for i in range(500):
        env.step()

    pts = env.get_now_pcd_from_camera().copy()
    env.get_keypoint_groups(pts)
    
    


    return env
