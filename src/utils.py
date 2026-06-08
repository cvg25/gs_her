import torch
import random, os
import numpy as np
import os
from pathlib import Path
import yaml

def seed_everything(seed=47):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_device(device_number, verbose=True):
    if torch.cuda.is_available():
        device = f'cuda:{int(device_number)}'
        assert device_number < torch.cuda.device_count(), f'Error, device {device} not available.'
    else:
        device = 'cpu'
    if verbose: print(f"Device available: '{device}' - {torch.cuda.get_device_name(device_number)}")
    return device

def compute_gradient_norm(parameters):
    total_norm = 0.
    for p in parameters:
        if p.grad is not None and p.requires_grad:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm 

def get_params_from_yaml_file(fpath):
    params = None
    with open(Path(fpath), 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    return params

def compute_num_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model_checkpoint(root_path, model, tag='model', verbose=True):
    path = root_path/f'checkpoint_{tag}.pth'
    torch.save(model.state_dict(), f=path)
    if verbose: print(f'Saved model checkpoint at: {path}')
    return path

def load_model_checkpoint(file_path, model, device):
    try:
        model.load_state_dict(torch.load(file_path, map_location=device, weights_only=True))
    except Exception as e:
        print(f'Encountered exception when loading checkpoint {e}')
    return model