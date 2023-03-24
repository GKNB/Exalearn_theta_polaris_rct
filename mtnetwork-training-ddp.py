#!/usr/bin/env python

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

#----------------------Parser settings---------------------------

parser = argparse.ArgumentParser(description='Training_DDP')

parser.add_argument('--batch_size',     type=int,   default=512,
                    help='input batch size for training (default: 512)')
parser.add_argument('--epochs',         type=int,   default=50,
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr',             type=float, default=0.0001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--seed',           type=int,   default=42,
                    help='random seed (default: 42)')
parser.add_argument('--log-interval',   type=int,   default=10, 
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--device',         default='gpu', choices=['cpu', 'gpu'],
                    help='Whether this is running on cpu or gpu')
parser.add_argument('--phase',          type=int,   default=0,
                    help='the current phase of workflow, phase0 will not read model')
parser.add_argument('--num_threads',    type=int,   default=0, 
                    help='set number of threads per worker. only work for cpu')
parser.add_argument('--num_workers',    type=int,   default=1, 
                    help='set the number of op workers. only work for gpu')
parser.add_argument('--data_root_dir',              default='./',
                    help='the root dir of gsas output data')
parser.add_argument('--model_dir',                  default='./',
                    help='the directory where save and load model')
parser.add_argument('--rank_data_gen',  type=int,   required=True,
                    help='number of ranks used to generate input data')
parser.add_argument('--rank_in_max',    type=int,   required=True,
                    help='inner block size for two-layer merging')

args = parser.parse_args()
args.cuda = ( args.device.find("gpu")!=-1 and torch.cuda.is_available() )


#--------------------DDP initialization-------------------------

size = int(os.getenv("PMI_SIZE"))
rank = int(os.getenv("PMI_RANK"))
local_rank = int(os.getenv("PMI_LOCAL_RANK"))

print("DDP: I am worker size = {}, rank = {}, local_rank = {}".format(size, rank, local_rank))

# Pytorch will look for these:
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(size)

if args.device == "gpu": backend = 'nccl'
elif args.device == "cpu": backend = 'gloo'

if rank == 0:
    print(args)
    print(backend)

torch.distributed.init_process_group(backend=backend, init_method='file:///grand/CSC249ADCD08/twang/real_work_polaris_gpu/sharedfile', world_size=size, rank=rank)
print("rank = {}, is_initialized = {}, nccl_avail = {}, get_rank = {}, get_size = {}".format(rank, torch.distributed.is_initialized(), torch.distributed.is_nccl_available(), torch.distributed.get_rank(), torch.distributed.get_world_size()))

if args.cuda:
    # DDP: pin GPU to local rank.
    print("rank = {}, local_rank = {}, num_of_gpus = {}".format(rank, local_rank, torch.cuda.device_count()))

    # Handles the case where we pinned GPU to local rank in run script
    if torch.cuda.device_count() == 1:
        torch.cuda.set_device(0)
    else:
        torch.cuda.set_device(int(local_rank))
#        torch.cuda.set_device(torch.cuda.device_count() - 1 - int(local_rank)) # handles Polaris NUMA topology
    torch.cuda.manual_seed(args.seed)

if (not args.cuda) and (args.num_threads!=0):
    torch.set_num_threads(args.num_threads)

print("Rank = {}".format(rank), " Torch Thread setup with number of threads: ", torch.get_num_threads(), " with number of inter_op threads: ", torch.get_num_interop_threads())

torch.manual_seed(args.seed)



#------------------------Model----------------------------
class FullModel( torch.nn.Module ):
    def __init__(self, len_input, num_hidden, num_output, 
                 conv1=(16, 3, 1), 
                 pool1=(2, 2), 
                 conv2=(32, 4, 2), 
                 pool2=(2, 2), 
                 fc1=256, 
                 num_classes=3):
        super(FullModel, self).__init__()
        
        n = len_input
        # In-channels, Out-channels, Kernel_size, stride ...
        self.conv1 = torch.nn.Conv1d(1, conv1[0], conv1[1], stride=conv1[2])
        n = (n - conv1[1]) // conv1[2] + 1
        
        self.pool1 = torch.nn.MaxPool1d(pool1[0], stride=pool1[1] )
        n = (n - pool1[0]) // pool1[1] + 1
        
        self.conv2 = torch.nn.Conv1d(conv1[0], conv2[0], conv2[1], stride=conv2[2])
        n = (n - conv2[1]) // conv2[2] + 1
        
        self.pool2 = torch.nn.MaxPool1d(pool2[0], stride=pool2[1] )
        n = (n - pool2[0]) // pool2[1] + 1
        
        self.features = torch.nn.Sequential( self.conv1, self.pool1, self.conv2, self.pool2 )
        self.fc1 = torch.nn.Linear(n*conv2[0], fc1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Linear(fc1, num_classes)        
        self.regression_layer=torch.nn.Linear(num_hidden, num_output)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        class_output = self.fc2(x)
        regression_output = self.regression_layer(x)
        return class_output, regression_output


#-----------------------------Loading data--------------------------------

X_scaled = np.load('/home/twang3/myWork/multitask_all_Y_balanced.npy')
y_scaled = np.load('/home/twang3/myWork/multitask_all_P_balanced.npy')
if rank == 0:
    print(X_scaled.shape, y_scaled.shape)

X_scaled = np.float32(X_scaled)
y_scaled = np.float32(y_scaled)
train_idx, test_idx = train_test_split(range(len(X_scaled)), test_size=0.05, random_state=42)
test_idx.sort()
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]
if rank == 0:
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

X_train_torch = torch.from_numpy(X_train)
y_train_torch = torch.from_numpy(y_train).reshape(-1,y_train.shape[1])
X_train_torch = FloatTensor(X_train_torch)
y_train_torch = FloatTensor(y_train_torch)

X_test_torch = torch.from_numpy(X_test)
y_test_torch = torch.from_numpy(y_test).reshape(-1,y_train.shape[1])
X_test_torch = FloatTensor(X_test_torch)
y_test_torch = FloatTensor(y_test_torch)

X_train_torch = X_train_torch.reshape((X_train_torch.shape[0], 1, X_train_torch.shape[1]))
X_test_torch = X_test_torch.reshape((X_test_torch.shape[0], 1, X_test_torch.shape[1]))
if rank == 0:
    print(X_train_torch.shape,y_train_torch.shape,X_test_torch.shape, y_test_torch.shape)


#----------------DDP: use DistributedSampler to partition the train/test data--------------------
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}
train_dataset = torch.utils.data.TensorDataset(X_train_torch,y_train_torch)
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=size, rank=rank)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

test_dataset = torch.utils.data.TensorDataset(X_test_torch,y_test_torch)
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=size, rank=rank)
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, sampler=test_sampler, **kwargs)
print("partition the train/test data finish on rank {}".format(rank))


#----------------------------setup model---------------------------------

model = FullModel(len_input = 2806, num_hidden = 256, num_output = 3, num_classes = 3)
if args.phase > 0:
    model.load_state_dict(torch.load(args.model_dir + "/full_model_phase{}.pt".format(args.phase-1)))
if args.cuda:
    model.cuda()
    model = DDP(model)
print("setup model finish on rank {}".format(rank))


#---------------------------setup optimizer------------------------

# DDP: scale learning rate according to the number of GPUs.
#optimizer = torch.optim.Adam(list(simple_xfer_model.parameters()) + list(lenet_trained_model.parameters()), lr=args.lr)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * size)
#optimizer = torch.optim.Adam(list(simple_xfer_model.parameters()) + list(lenet_trained_model.parameters()), lr=args.lr * np.sqrt(size))

criterion1 = torch.nn.MSELoss()
criterion2 = torch.nn.BCEWithLogitsLoss()
print("setup optimizer and loss finish on rank {}".format(rank))


def metric_average(val, name):
    # Sum everything and divide by total size:
    dist.all_reduce(val,op=dist.reduce_op.SUM)
    val /= size
    return val

def modify_lr(optimizer, warmup_epoch, first_decay_epoch, second_decay_epoch, current_epoch, lr_init, lr_final, decay_factor):

    lr_modified = lr_final
    if current_epoch <= warmup_epoch:
        lr_modified = lr_init + (lr_final - lr_init) / (warmup_epoch - 1) * (current_epoch - 1)

    if current_epoch > first_decay_epoch:
        lr_modified = lr_modified * decay_factor

    if current_epoch > second_decay_epoch:
        lr_modified = lr_modified * decay_factor

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_modified

    if(rank == 0):
        print("rank = ", rank, " opt len = ", len(optimizer.param_groups), " lr = ", optimizer.param_groups[0]['lr'])


#------------------------------start training----------------------------------

def train(epoch):
    model.train()
    train_sampler.set_epoch(epoch)

    running_loss  = torch.tensor(0.0)
    running_loss1 = torch.tensor(0.0)
    running_loss2 = torch.tensor(0.0)
    if args.device == "gpu":
        running_loss, running_loss1, running_loss2  = running_loss.cuda(), running_loss1.cuda(), running_loss2.cuda()

    for batch_idx, current_batch in enumerate(train_loader):      
        if args.cuda:
            inp, current_batch_y = current_batch[0].cuda(), current_batch[1].cuda()
        else:
            inp, current_batch_y = current_batch[0],        current_batch[1]

        optimizer.zero_grad()
        class_output, regression_output = model(inp)
        regression_gndtruth = current_batch_y[:,0:3]
        class_gndtruth = current_batch_y[:,3:6]

        loss1 = criterion1(regression_output, regression_gndtruth)
        loss2 = criterion2(class_output, class_gndtruth )
        loss  = loss1 + loss2
        loss.backward()
        optimizer.step()
        running_loss  += loss.item()
        running_loss1 += loss1.item()
        running_loss2 += loss2.item()
    
        if batch_idx % args.log_interval == 0 and rank == 0:
            print("[Rank = {}] Train Epoch: {} [{}/{} ({:.1f}%)]\tloss1: {:15.6f}, loss2: {:15.6f}, loss_tot: {:15.6f}".format(rank, epoch, batch_idx, len(train_loader),
                100.0 * batch_idx / len(train_loader), loss1.item(), loss2.item(), loss.item()))
    running_loss  = running_loss  / len(train_loader)
    running_loss1 = running_loss1 / len(train_loader)
    running_loss2 = running_loss2 / len(train_loader)
    loss_avg  = metric_average(running_loss,  'running_loss')
    loss1_avg = metric_average(running_loss1, 'running_loss1')
    loss2_avg = metric_average(running_loss2, 'running_loss2')
    if rank == 0: 
        print("Training set: Average loss1: {:15.8f}, loss2: {:15.8f}, loss: {:15.8f}".format(loss1_avg, loss2_avg, loss_avg))

def test():
    model.eval()
    
    test_loss  = torch.tensor(0.0)
    if args.device == "gpu":
        test_loss  = test_loss.cuda()

    for inp, current_batch_y in test_loader:
        if args.cuda:
            inp, current_batch_y = inp.cuda(), current_batch_y.cuda()

        class_output, regression_output = model(inp)
        test_loss += criterion1(regression_output, current_batch_y[:,0:3]).item()

    test_loss /= len(test_loader)
    test_loss = metric_average(test_loss, 'avg_loss')

    if rank == 0:
        print('Test set: Average loss: {:15.8f}\n'.format(test_loss))

time_tot = time.time()
for epoch in range(1, args.epochs + 1):
    e_start = time.time()
    modify_lr(optimizer, 50, 250, 500, epoch, args.lr, args.lr * size, 0.5)
    train(epoch)
    test()
    e_end = time.time()
    if rank==0:
        print("Epoch - %d time: %s seconds" %(epoch, e_end - e_start))
time_tot = time.time() - time_tot
print("Rank = {} Total training time = {} with num_epochs = {} and num_processes = {}".format(rank, time_tot, args.epochs, size))

torch.distributed.destroy_process_group()

#if rank == 0:
#    torch.save(simple_xfer_model.state_dict(), args.model_dir + "/simple_xfer_model_phase{}.pt".format(args.phase))
#    torch.save(lenet_trained_model.state_dict(), args.model_dir + "/lenet_model_phase{}.pt".format(args.phase))


