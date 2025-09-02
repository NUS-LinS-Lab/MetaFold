export CUDA_VISIBLE_DEVICES=0,1

accelerate launch --config_file configs/acc_config.yaml scripts/test.py 