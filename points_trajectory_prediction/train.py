import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import PointCloudTrajectoryDataset, PointCloudTrajectoryTransform
from trajectory_predictor import TrajectoryTransformer
from utils.model_utils import init_weights, save_model, get_scheduler
from utils.config import Config 
from utils.logger import visualize_point_cloud_trajectories, setup_logger, visualize_pred_point_cloud
import numpy as np
import wandb
import os

import utils
from utils import init_distributed_mode

import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Linear, Module
import torch.multiprocessing as mp

from pointnet2_sem_seg_msg import FeatureEncoder
from loss import Loss, cosine_similarity
from utils.data_utils import points_occlusion, random_rotation_matrix, apply_rotation, cls_accuracy, normalize_point_cloud_trajectory, description_encoding, average_tokens, weighted_random_frame
from utils.llm_utils import randomize_description_embed, fixed_description_embed

import shutil
import time
import argparse
import transformers
import torch

from transformers import AutoTokenizer, LlamaForCausalLM


def get_args():
    parser = argparse.ArgumentParser('Points_Trajectory_Prediction', add_help=False)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--test_best', action='store_true',
                        help='Whether test the best model')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--bf16', default=False, action='store_true')
    parser.add_argument('--zero_stage', default=0, type=int,
                        help='ZeRO optimizer stage (default: 0)')
    
    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init

def preprocess(inputs, targets):
    start_time = time.perf_counter()
    # start_frame = torch.randint(0, 20, (1,)).item()
    start_frame = weighted_random_frame(start=0, end=20, bias_towards_start=True, decay_rate=0.85)
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

    if config.enable_height_random:
        random_height_scale = np.random.uniform(0.9, 1.1)
        for i in range(1, targets.shape[2]):  
            delta_y = targets[:, :, i, 1] - inputs[:, :, 1] 
            targets[:, :, i, 1] = inputs[:, :, 1] + delta_y * random_height_scale
    
    start_time = time.perf_counter()
    if config.enable_occlusion:
        inputs, targets = points_occlusion(inputs, targets)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # print(f"The occlusion operation took {elapsed_time:.6f} seconds to complete.")
    
    if config.enable_rotation:
        batch_size, num_points, num_frames, _ = targets.shape
        rotation_matrix = random_rotation_matrix(fixed=False, degree=[torch.pi/6, 0, 0]).to(inputs.device)
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
            inputs, targets, names, _ = batch['point_cloud'].to(config.device), batch['trajectory'].to(config.device), batch['name'], batch['description_embed'].to(config.device)
            # description_embed = randomize_description_embed(names, embed_dict).to(config.device)          # random description embedding, only in training(eval in train/eval dataset)
            description_embed = fixed_description_embed(names, embed_dict).to(config.device) 

            inputs_pro, targets_pro, mask_pro, cls_point, mask = preprocess(inputs, targets)

            # description_embed = llm_embedding(llm_model, tokenizer, inputs_pro, names)

            outputs_off, z_mu, z_logvar = model(inputs_pro, description_embed, None, 'eval')
            outputs_traj = inputs_pro.unsqueeze(2) + outputs_off

            loss = criterion(outputs_traj, targets_pro, mask_pro, z_mu, z_logvar)

            similarity_score = cosine_similarity(outputs_traj * mask.unsqueeze(-1), targets_pro * mask.unsqueeze(-1)) * 50 + 50

            # log loss and lerning rate
            # wandb.log({"train_loss": loss.item(), "learning_rate": optimizer.param_groups[0]['lr']})
            # similarity_score = cosine_similarity(outputs, targets) * 50 + 50
            total_loss += loss
            total_score += similarity_score

            batch_size = inputs_pro.shape[0]
            random_index = np.random.randint(0, batch_size)
            output_list.append((inputs_pro[random_index], targets_pro[random_index], mask_pro[random_index], outputs_traj[random_index]))

    if config.eval_vis and output_list and utils.is_main_process():
        vis_index = np.random.randint(0, len(output_list))
        inputs_vis, targets_vis, mask_vis, outputs_vis = output_list[vis_index]
        outputs_cpu = outputs_vis.cpu().detach().numpy()
        inputs_cpu = inputs_vis.cpu().detach().numpy()
        targets_cpu = targets_vis.cpu().detach().numpy()
        mask_cpu = mask_vis.cpu().detach().numpy()
        visualize_point_cloud_trajectories(inputs_cpu,targets_cpu,mask_cpu,outputs_cpu, save_path='./vis_pc_eval.png')
        visualize_pred_point_cloud(inputs_cpu,targets_cpu,mask_cpu,outputs_cpu, save_path='./vis_pc2_eval.png')

    if eval_save and utils.is_main_process():
        print(os.getcwd())
        output_dir = "outputs/eval_ep{}/".format(epoch)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        for i, tup in enumerate(output_list):
            if epoch % 20 == 0:            # for all data
            # if True:                                # for isaac data
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
    print(f'Average CLS Score: {avg_cls:.4f}')

    wandb.log({"trajectory similarity score": avg_score})
    wandb.log({"trajectory CLS score": avg_cls})

    return avg_loss



def train(config, args):
    
    # Setup logger
    logger = setup_logger('train_logger', config.log_path)
    logger.info("Starting the project...")

    # Check device
    logger.info(f"Using device: {config.device}")
    
    # Load dataset and create train and eval loader
    # transform = PointCloudTrajectoryTransform()
    transform = None
    train_dataset = PointCloudTrajectoryDataset(config.data_path, split='train', transform=transform)
    eval_dataset = PointCloudTrajectoryDataset(config.data_path, split='eval')
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_val = torch.utils.data.SequentialSampler(eval_dataset)
    train_loader = DataLoader(dataset=train_dataset, sampler=sampler_train, batch_size=config.batch_size, num_workers=config.num_workers, persistent_workers=True)
    eval_loader = DataLoader(dataset=eval_dataset, sampler=sampler_val, batch_size=config.batch_size, num_workers=config.num_workers, persistent_workers=True)


    embed_dict = torch.load('data/description_embeddings_mirrored.pt')
    
    if config.enable_cls:
        num_points = config.num_points + 1
    else:
        num_points = config.num_points
    # Initialize model
    model = TrajectoryTransformer(
        input_dim=config.pcd_output_dim, 
        hidden_dim=config.model_dim, 
        output_dim=config.output_dim, 
        nhead=config.n_heads, 
        num_encoder_layers=config.num_encoder_layers, 
        num_decoder_layers=config.num_decoder_layers, 
        num_points=num_points, 
        num_frames=config.num_frames, 
        point_dim=config.point_dim,
        device=config.device
    ).to('cuda')
    
    model_without_ddp = model
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        print('Finish creating ddp model')
    
    combined_params = list(model_without_ddp.parameters())

    # Define loss and optimizer
    criterion = Loss()
    optimizer = optim.AdamW(combined_params, lr=config.learning_rate, weight_decay=0.01)    # AdamW   weight decay
    scheduler = get_scheduler(optimizer, config.epochs / 5)  # Make sure to define or adjust this function

    # Training loop
    model.train()

    for epoch in range(config.epochs):
        for batch_idx, batch in enumerate(train_loader):
            inputs, targets, names, _ = batch['point_cloud'].to(config.device), batch['trajectory'].to(config.device), batch['name'], batch['description_embed'].to(config.device)
            # description_embed = randomize_description_embed(names, embed_dict).to(config.device)          # random description embedding
            description_embed = fixed_description_embed(names, embed_dict).to(config.device)                # fixed description embedding

            optimizer.zero_grad()
            
            if epoch == 0 and batch_idx == 0 and utils.is_main_process():
                print('in train: inputs ', inputs.shape, ',  trajecotry ', targets.shape)
                print('inputs ', inputs[0][0], ',  trajecotry ', targets[0][0])

            inputs_pro, targets_pro, mask_pro, cls_point, mask = preprocess(inputs, targets)

            # start_time = time.perf_counter()
            # description_embed = llm_embedding(llm_model, tokenizer, inputs_pro, names)
            # end_time = time.perf_counter()
            # elapsed_time = end_time - start_time
            # print(f"The LLM embedding took {elapsed_time:.6f} seconds to complete.")

            outputs_off, z_mu, z_logvar = model(inputs_pro, description_embed, targets_pro, 'train')
            outputs_traj = inputs_pro.unsqueeze(2) + outputs_off

            loss = criterion(outputs_traj, targets_pro, mask_pro, z_mu, z_logvar)
            loss.backward()
            max_grad_norm = 5.0 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            # log loss and lerning rate
            wandb.log({"train_loss": loss.item(), "learning_rate": optimizer.param_groups[0]['lr']})

            if batch_idx % 10 == 0:
                logger.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')

                if config.train_vis and utils.is_main_process():
                    vis_index = np.random.randint(0, targets.shape[0])
                    outputs_cpu = outputs_traj.cpu().detach().numpy()
                    inputs_cpu = inputs_pro.cpu().detach().numpy()
                    targets_cpu = targets_pro.cpu().detach().numpy()
                    mask_cpu = mask.cpu().detach().numpy()
                    visualize_point_cloud_trajectories(inputs_cpu[vis_index], targets_cpu[vis_index], mask_cpu[vis_index], outputs_cpu[vis_index], save_path='./vis_pc_train.png')
                    visualize_pred_point_cloud(inputs_cpu[vis_index], targets_cpu[vis_index], mask_cpu[vis_index], outputs_cpu[vis_index], save_path='./vis_pc2_train.png')

        if epoch % 1 == 0:  # eval every 10 epoch
            logger.info("Starting evaluation...")
            eval_loss = evaluate(config=config, model=model, eval_loader=eval_loader, criterion=criterion, epoch=epoch)
            wandb.log({"eval_loss": eval_loss})
            if not os.path.exists(config.save_path):
                os.mkdir(config.save_path)
            if utils.is_main_process():
                save_model(model_without_ddp, os.path.join(config.save_path, f"model_{epoch}.pth"))
                logger.info(f'Model_{epoch} saved.')
            model.train()
        
        scheduler.step()

    logger.info("Project completed.")

if __name__ == '__main__':
    args, _ = get_args()
    init_distributed_mode(args)

    # Initialize configuration
    config = Config()

    # Initialize wandb
    # wandb.init(mode="disabled")
    wandb.init(project='point trajctory prediction', name=config.run_name)

    # # config wandb
    wandb.config = {
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
    }
    
    # Start training
    train(config, args)
