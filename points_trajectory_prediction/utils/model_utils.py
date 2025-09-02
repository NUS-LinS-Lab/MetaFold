import torch
import torch.nn as nn
import logging
from torch.optim.lr_scheduler import StepLR

# Initialize weights of the model
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Save the model state
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Load the model state
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

# Define a learning rate scheduler
def get_scheduler(optimizer, step_size=30, gamma=0.1):
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    return scheduler


