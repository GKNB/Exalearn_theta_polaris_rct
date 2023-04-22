print("Start at the beginning of training!")
import io, os, sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import FloatTensor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
import torch.utils.data.distributed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import argparse
import preprocess_data as predata

print("Start very beginning of training!")



X_scaled = np.load('/home/twang3/myWork/multitask_all_Y_balanced.npy')
y_scaled = np.load('/home/twang3/myWork/multitask_all_P_balanced.npy')
print(X_scaled.shape, y_scaled.shape)

X_scaled = np.load('/grand/CSC249ADCD08/twang/real_work_polaris_gpu/sc23_step1_all/Output-all-Y.npy')
y_scaled = np.load('/grand/CSC249ADCD08/twang/real_work_polaris_gpu/sc23_step1_all/Output-all-P.npy')
print(X_scaled.shape, y_scaled.shape)

