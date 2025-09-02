import torch
import os
from datetime import datetime

class Config:
    """Configuration class for the project."""
    def __init__(self):

        # self.data_path = '/mnt/petrelfs/wangchenting/DistillUMT/codebase-umt/points-trajectory-prediction/data/TNLC_normalized_data.h5'
        # self.data_path = '/data2/chaonan/points-traj-prediction/data/all_normalized_data.h5'
        # self.data_path = '/data2/chaonan/points-traj-prediction/data/all_data.h5'
        # self.data_path = '/data2/chaonan/points-traj-prediction/data/all_data_samedirection.h5'
        # self.data_path = '/data2/chaonan/points-traj-prediction/data/all_data_samedirection_mirrored_desemb_0926_allupdated.h5'
        self.data_path ='/data2/chaonan/points-traj-prediction/data/all_data_samedirection_mirrored_1030_traj2data.h5'
        # self.data_path = '/data2/chaonan/points-traj-prediction/data/TNLC_data_samedirection_onlyleft_1030_0926data.h5'
        # self.data_path = '/data2/chaonan/points-traj-prediction/data/TNLC_data_mirrored_desemb_0916.h5'
        # self.data_path = '/data2/chaonan/points-traj-prediction/data/all_data_samedirection_mirrored_1030_traj2data.h5'
        # self.data_path = '/data2/chaonan/points-traj-prediction/data/all_data_samedirection_mirrored_1108_onlyLonggSleevea0_traj2data.h5'
        # self.data_path = '/data2/chaonan/points-traj-prediction/data/all_data_samedirection_mirrored_1108_onlyLonggSleevea0a1_traj2data.h5'
        # self.data_path = '/data2/chaonan/points-traj-prediction/data/TNLC_data_small_1025.h5'

        self.batch_size = 8
        self.num_workers = 4

        # Training-related configurations
        self.epochs = 200
        self.learning_rate = 1e-4 * 6
        
        self.save_path = '/data2/chaonan/points-traj-prediction/lyw_outputs/weights_hist_one_step_clip_embedding_seq'

        self.enable_padding_mask = False
        self.enable_occlusion = True
        self.enable_rotation = True
        self.enable_normalization = True
        self.enable_cls = False     
        # Now deprecate CLS in both model and loss

        self.input_dim = 3
        # Model-related configurations
        self.pcd_output_dim = 128
        # self.pcd_output_dim = 3
        self.model_dim = 256
        self.output_dim = 3

        self.n_heads = 4
        self.num_encoder_layers = 4
        self.num_decoder_layers = 4
        self.num_points = 2048  # Number of trajectory sequences to predict
        self.num_frames = 21  # Number of points per trajectory
        self.point_dim = 3  # Dimensionality of each point

        # device
        # torch.cuda.set_device(1) 
        self.device = torch.device('cuda')

        # visualize
        self.train_vis = True
        self.eval_vis = True

        # set log config
        log_dir = 'logs/'
        os.makedirs(log_dir, exist_ok=True)
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_path = os.path.join(log_dir, f'training_{current_time}.log')

        # config for wandb
        self.run_name = f'training_{current_time}.log'