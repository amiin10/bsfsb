"""Small reproducibility / device helpers shared by pinn.py and train.py."""
import os
import random

import numpy as np
import torch


def seed_torch(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
