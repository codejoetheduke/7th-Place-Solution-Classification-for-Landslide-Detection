#src/utils/config_yolo.py
import os
import random
import numpy as np
import torch

class CFG:
    seed = 42
    batch=20
    random_state = 42
    folds = 10
    epochs=100
    imgsz=320
    device=0
    optimizer='AdamW'
    lr0=3e-4
    momentum=0.9
    weight_decay=1e-2
    close_mosaic=30
    seed=42
    patience=10
    threshold=0.5

def seed_everything(seed=CFG.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)