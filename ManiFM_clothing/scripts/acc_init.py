import yaml
from accelerate import Accelerator
import os

with open('configs/acc_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

accelerator = Accelerator(**config)