# Import Libraries
import os
from numpy.core.defchararray import index
import torch
from torch.utils import data
from torchvision import transforms
from torch.utils.data import DataLoader
from src.data_loader.data_set import Data_Set
from src.data_loader.utils import get_train_val_split
from src.constants import BASE_DIR, FREIHAND_DATA
from src.utils import read_json
import matplotlib.pyplot as plt
import numpy as np
import copy
from easydict import EasyDict as edict
from src.utils import read_json
from src.visualization.visualize import plot_hand
import pandas as pd
from tqdm import tqdm

train_param = edict(
    read_json(f"{BASE_DIR}/src/experiments/config/training_config.json")
)
train_param.augmentation_flags.resize = True
train_param.augmentation_flags.random_crop = True
train_data = Data_Set(
    config=train_param,
    transform=transforms.ToTensor(),
    split="test",
    experiment_type="hybrid2",
    source="mpii",
)
# print(da)
data_loader = DataLoader(train_data, batch_size=128, num_workers=4)
for i in enumerate(data_loader):
    print(i[1].keys())
    break
# val_data = copy.copy(train_data)
# val_data.is_training(False)
# data_loader = (DataLoader(val_data, batch_size=128, num_workers=4),)
# for i in tqdm(iter(data_loader)):
    # i
