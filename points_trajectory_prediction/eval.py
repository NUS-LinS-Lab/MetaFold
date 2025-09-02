import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import PointCloudTrajectoryDataset, PointCloudTrajectoryTransform
from trajectory_predictor import TrajectoryTransformer
from utils.model_utils import init_weights, save_model, get_scheduler, load_model
from utils.config import Config 
from utils.logger import visualize_point_cloud_trajectories, setup_logger, visualize_pred_point_cloud
import numpy as np
import wandb
import os

import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Linear, Module
import torch.multiprocessing as mp

from pointnet2_sem_seg_msg import FeatureEncoder
from loss import Loss, cosine_similarity
from utils.data_utils import points_occlusion, random_rotation_matrix, apply_rotation, cls_accuracy, normalize_point_cloud_trajectory
from utils.llm_utils import randomize_description_embed, description_matching

import shutil
import time
import argparse


def preprocess(inputs, targets):
    start_time = time.perf_counter()
    # start_frame = torch.randint(0, 20, (1,)).item()
    start_frame = 0
    inputs = targets[:, :, start_frame,:]           
    targets = targets[:, :, start_frame:, :]

    num_missing_frames = 21 - targets.shape[2]
    if num_missing_frames > 0:
        last_frame = targets[:, :, -1:, :] 
        padding = last_frame.repeat(1, 1, num_missing_frames, 1)
        targets = torch.cat((targets, padding), dim=2)

    mask = torch.ones_like(targets[:, :, :, 0], dtype=torch.bool)
    if config.enable_padding_mask:
        if num_missing_frames > 0:
            mask[:, :, -num_missing_frames:] = 0
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # print(f"The padding operation took {elapsed_time:.6f} seconds to complete.")
    
    start_time = time.perf_counter()
    if False:
        inputs, targets = points_occlusion(inputs, targets)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # print(f"The occlusion operation took {elapsed_time:.6f} seconds to complete.")
    
    if config.enable_rotation:
        batch_size, num_points, num_frames, _ = targets.shape
        rotation_matrix = random_rotation_matrix(fixed=False, degree=[0, 0, 0]).to(inputs.device)
        # rotation_matrix = random_rotation_matrix(fixed=True, degree=torch.pi/6).to(inputs.device)
        inputs = apply_rotation(inputs, rotation_matrix)
        targets = apply_rotation(targets.view(batch_size, num_points * num_frames, 3), rotation_matrix)
        targets = targets.view(batch_size, num_points, num_frames, 3)

    if config.enable_normalization:
        targets = normalize_point_cloud_trajectory(targets)
        inputs = targets[:, :, 0, :]

    if config.enable_cls:
        cls_point = torch.ones((targets.shape[0], 1, targets.shape[2], 1), device=targets.device)
        cls_point[:, :, -num_missing_frames:] = 0  
        cls_point = cls_point.expand(-1, -1, -1, 3)
        targets_wcls = torch.cat((targets, cls_point), dim=1)

        mask_wcls = torch.cat((mask, torch.ones((mask.shape[0], 1, mask.shape[2]), dtype=torch.bool, device=mask.device)), dim=1)

        cls_point = torch.ones((inputs.shape[0], 1, inputs.shape[2]), device=inputs.device)  # Initialize cls point for input
        inputs_wcls = torch.cat((inputs, cls_point), dim=1)  # Add cls point to the input
    else:
        inputs_wcls = inputs
        targets_wcls = targets
        mask_wcls = mask
        cls_point = torch.ones((targets.shape[0], 1, targets.shape[2], 1), device=targets.device)
        if num_missing_frames > 0:
            cls_point[:, :, -num_missing_frames:, :] = 0

    return inputs_wcls, targets_wcls, mask_wcls, cls_point, mask


def evaluate(config, model, eval_loader, criterion, epoch):
    model.eval()
    total_loss = 0
    total_score = 0
    total_cls = 0
    eval_save = True
    output_list = []
    embed_dict = torch.load('data/description_embeddings_mirrored.pt')

    with torch.no_grad():
        for batch in eval_loader:
            inputs, _, names, description, _ = batch['point_cloud'].to(config.device), batch['trajectory'].to(config.device), batch['name'], batch['description'], batch['description_embed'].to(config.device)
            targets = inputs.unsqueeze(2).repeat(1, 1, 21, 1)
            description, description_embed = description_matching(description, embed_dict)
            description_embed = description_embed.to(config.device)
            print(names)
            # print(description)

            inputs_pro, targets_pro, mask_pro, cls_point, mask = preprocess(inputs, targets)

            # description_embed = llm_embedding(llm_model, tokenizer, inputs_pro, names)

            outputs_off, z_mu, z_logvar = model(inputs_pro, description_embed, None, 'eval')
            outputs_traj = inputs_pro.unsqueeze(2) + outputs_off

            loss = criterion(outputs_traj, targets_pro, mask_pro,z_mu, z_logvar)

            similarity_score = cosine_similarity(outputs_traj * mask.unsqueeze(-1), targets_pro * mask.unsqueeze(-1)) * 50 + 50

            # log loss and lerning rate
            # wandb.log({"train_loss": loss.item(), "learning_rate": optimizer.param_groups[0]['lr']})
            # similarity_score = cosine_similarity(outputs, targets) * 50 + 50
            total_loss += loss
            total_score += similarity_score

            batch_size = inputs_pro.shape[0]
            random_index = np.random.randint(0, batch_size)
            output_list.append((inputs_pro[random_index], targets_pro[random_index], mask_pro[random_index], outputs_traj[random_index]))

            if config.eval_vis and output_list:
                # vis_index = np.random.randint(0, len(output_list))
                inputs_vis, targets_vis, mask_vis, outputs_vis = output_list[-1]
                outputs_cpu = outputs_vis.cpu().detach().numpy()
                inputs_cpu = inputs_vis.cpu().detach().numpy()
                targets_cpu = targets_vis.cpu().detach().numpy()
                mask_cpu = mask_vis.cpu().detach().numpy()
                visualize_point_cloud_trajectories(inputs_cpu,targets_cpu,mask_cpu,outputs_cpu, save_path='./vis_pc_eval_real.png')
                visualize_pred_point_cloud(inputs_cpu,targets_cpu,mask_cpu,outputs_cpu, save_path='./vis_pc2_eval_real.png')
            print('>>>>>>>>>>>>>><<<<<<<<<<<<<<')
            time.sleep(3)

    if eval_save:
        print(os.getcwd())
        output_dir = "outputs_real/eval_ep{}/".format(epoch)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        for i, tup in enumerate(output_list):
            # if epoch % 20 == 0:            # for all data
            if True:                                # for isaac data
                cloth_dir= os.path.join(output_dir, "cloth{}/".format(i))
                if os.path.exists(cloth_dir):
                    shutil.rmtree(cloth_dir)
                os.mkdir(cloth_dir)
                target_file = os.path.join(cloth_dir, "target.txt".format(i))
                output_file = os.path.join(cloth_dir, "output.txt".format(i))
                target = tup[1].cpu().detach().numpy()
                target = target.reshape(target.shape[0], 63)
                output = tup[3].cpu().detach().numpy()
                output = output.reshape(output.shape[0], 63)
                np.savetxt(target_file, target)
                np.savetxt(output_file, output)

    # calculate average loss
    avg_loss = total_loss / len(eval_loader)
    avg_score = total_score / len(eval_loader)
    avg_cls = total_cls / len(eval_loader)
    print(f'Average Evaluation Loss: {avg_loss:.4f}')
    print(f'Average Similarity Score: {avg_score:.4f}')
    # print(f'Average CLS Score: {avg_cls:.4f}')

    wandb.log({"trajectory similarity score": avg_score})
    wandb.log({"trajectory CLS score": avg_cls})

    return avg_loss



if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    # Initialize configuration
    config = Config()
    config.batch_size = 1   # for eval

    # Initialize wandb
    # wandb.init(mode="disabled")
    wandb.init(project='point trajctory prediction', name=config.run_name)

    # config wandb
    wandb.config = {
    "learning_rate": config.learning_rate,
    "epochs": config.epochs,
    "batch_size": config.batch_size,
    }

    # Setup logger
    logger = setup_logger('eval_logger', config.log_path)
    logger.info("Starting the project...")

    # Check device
    logger.info(f"Using device: {config.device}")

    # Load dataset and create train and eval loader
    # transform = PointCloudTrajectoryTransform()
    transform = None
    # eval_dataset = PointCloudTrajectoryDataset('/data2/chaonan/points-traj-prediction/data/all_data_samedirection_mirrored_desemb_0915_allupdated.h5', split='eval')
    eval_dataset = PointCloudTrajectoryDataset('/data2/chaonan/points-traj-prediction/data/TNSC_test_data.h5', split='all')
    
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = TrajectoryTransformer(
        input_dim=config.pcd_output_dim, 
        hidden_dim=config.model_dim, 
        output_dim=config.output_dim, 
        nhead=4, 
        num_encoder_layers=4, 
        num_decoder_layers=config.num_decoder_layers, 
        num_points=config.num_points, 
        num_frames=config.num_frames, 
        point_dim=config.point_dim,
        device=config.device
    ).to(config.device)
    
    epoch = 150


    model = load_model(model, f'/data2/chaonan/points-traj-prediction/logs/cvae/model_150.pth')

    criterion = Loss()
    # Start evaluation
    evaluate(config, model, eval_loader, criterion, epoch=epoch)
