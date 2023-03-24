#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import inspect
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch import FloatTensor
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler, BatchSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, mean_squared_error
import horovod.torch as hvd 
import io, os, sys
import time

import preprocess_data as predata

print("Start very beginning of training!")

#----------------------Training settings---------------------------

parser = argparse.ArgumentParser(description='MVP_horovod')
#parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
#parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#                    help='learning rate (default: 0.01)')
#parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
#parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                    help='how many batches to wait before logging training status')
#parser.add_argument('--fp16-allreduce', action='store_true', default=False,
#                    help='use fp16 compression during allreduce')
parser.add_argument('--device', default='cpu',
                    help='Wheter this is running on cpu or gpu')
parser.add_argument('--phase', type=int, default=0,
                    help='the current phase of workflow, phase0 will not read model')
parser.add_argument('--num_threads', type=int, default=0, 
                    help='set number of threads per worker')
parser.add_argument('--data_root_dir', default='./',
                    help='the root dir of gsas output data')
parser.add_argument('--model_dir', default='./',
                    help='the directory where save and load model')
parser.add_argument('--rank_data_gen', type=int, default=256,
                    help='number of ranks used to generate input data')
parser.add_argument('--rank_in_max', type=int, default=64,
                    help='inner block size for two-layer merging')
args = parser.parse_args()

args.cuda = args.device.find("gpu")!=-1



#-----------------------Hyper parameters and data setting---------------------

SEQ_LENGTH = 256
STRIDE = 1
HIDDEN_SIZE = 128
LEARNING_RATE = 0.001
DROPOUT = 0.2
NUM_OUTPUTS = 2
NUM_LAYERS = 3
BATCH_SIZE = 512

root_path = args.data_root_dir + '/phase{}'.format(args.phase) + '/'
print("root_path for data = ", root_path)
path_in_list = ['test_cubic/', 
                'test_trigonal_part1/', 
                'test_trigonal_part2/', 
                'test_tetragonal/']
filename_in_list = ['cubic_1001460_cubic_part', 
                    'trigonal_1522004_trigonal_part', 
                    'trigonal_1522004_trigonal_part',
                    'tetragonal_1531431_tetragonal_part']
rank_max_list = [args.rank_data_gen, args.rank_data_gen, args.rank_data_gen, args.rank_data_gen]
rank_in_max = args.rank_in_max



#---------------------Horovod: initialize library----------------------

hvd.init()
torch.manual_seed(args.seed)
print("Horovod: I am worker %s of %s." %(hvd.rank(), hvd.size()))
if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)
if (args.num_threads!=0):
    torch.set_num_threads(args.num_threads)

if hvd.rank()==0:
    print("Torch Thread setup: ")
    print(" Number of threads: ", torch.get_num_threads())




#------------------------Model----------------------------

class Lenet1d( torch.nn.Module ):
    def __init__(self, len_input, 
                 conv1=(16, 3, 1), 
                 pool1=(2, 2), 
                 conv2=(32, 4, 2), 
                 pool2=(2, 2), 
                 fc1=256, 
                 num_classes=3):
        super(Lenet1d, self).__init__()
        
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
        self.prediction = torch.nn.Sequential( self.fc1, self.relu, self.fc2 )            
    
    def fingerprint(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = self.relu(x)
        return x
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.prediction(x)
        return x


class XFerLearning(torch.nn.Module):
    def __init__(self, num_hidden, num_output):
        super(XFerLearning, self).__init__()
        self.regression_layer=torch.nn.Linear(num_hidden, num_output)
    
    def forward(self, X):
        return self.regression_layer(X)    


#-----------------------Helper function-----------------------------
# Where are these functions used???

# motivated out of https://stackoverflow.com/questions/40084931/taking-subarrays-from-numpy-array-with-given-stride-stepsize
# Window len = L, Stride len/stepsize = S
def strided_app(a, L, S ):  
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

def batch_stride(X, L, S):
    nrows = ((X.shape[1]-L)//S)+1
    X_strided = np.empty(shape=(X.shape[0],L, nrows))
    for i in range(0, X.shape[0]-1):
        X_strided[i,:,:]=(strided_app(X[i,:],L,S)).T
    return X_strided

#https://stackoverflow.com/questions/43870647/how-to-duplicate-a-row-or-column-in-a-numpy-array
def dup_cols(a, indx, num_dups=1):
    return np.insert(a,[indx+1]*num_dups,a[:,[indx]],axis=1)

#https://deeplizard.com/learn/video/0LhiS6yu2qQ
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))

def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

#-----------------------------Loading data--------------------------------

#X_scaled = np.load('/home/twang3/myWork/multitask_all_Y_balanced.npy')
#y_scaled = np.load('/home/twang3/myWork/multitask_all_P_balanced.npy')

print("Start reading and preprocessing data!")
X_scaled, y_scaled = predata.read_and_merge_data(root_path, path_in_list, filename_in_list, rank_max_list, rank_in_max, do_saving=False, filename_prefix="Output")
print(X_scaled.shape, y_scaled.shape)


X_scaled = np.float32(X_scaled)
y_scaled = np.float32(y_scaled)
train_idx, test_idx = train_test_split(range(len(X_scaled)), test_size=0.05, random_state=42)
test_idx.sort()
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


num_gpus = 0
multigpu = False
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
if device == 'cuda':
    num_gpus = torch.cuda.device_count()
    if  num_gpus > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        multigpu = False


X_train_torch = torch.from_numpy(X_train)
y_train_torch = torch.from_numpy(y_train).reshape(-1,y_train.shape[1])
X_train_torch = FloatTensor(X_train_torch)
y_train_torch = FloatTensor(y_train_torch)

X_test_torch = torch.from_numpy(X_test)#.cuda()
y_test_torch = torch.from_numpy(y_test).reshape(-1,y_train.shape[1])#.cuda()
X_test_torch = FloatTensor(X_test_torch)
y_test_torch = FloatTensor(y_test_torch)
X_test_torch = X_test_torch.to(device)
y_test_torch = y_test_torch.to(device)

X_train_torch = X_train_torch.reshape((X_train_torch.shape[0], 1, X_train_torch.shape[1]))
X_test_torch = X_test_torch.reshape((X_test_torch.shape[0], 1, X_test_torch.shape[1]))

print(X_train_torch.shape,y_train_torch.shape,X_test_torch.shape, y_test_torch.shape)
#sampler = WeightedRandomSampler(unique_sample_weights, BATCH_SIZE)



#----------------Horovod: use DistributedSampler to partition the train/test data--------------------

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_dataset = TensorDataset(X_train_torch,y_train_torch)
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, **kwargs)

test_dataset = TensorDataset(X_test_torch,y_test_torch)
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler, **kwargs)



#----------------------------setup model---------------------------------

lenet_trained_model = Lenet1d(2806, num_classes=3)
#if args.phase > 0 and hvd.rank() == 0:
if args.phase > 0:
    lenet_trained_model.load_state_dict(torch.load(args.model_dir + "/lenet_model_phase{}.pt".format(args.phase-1), map_location=torch.device('cpu')))
#lenet_trained_model.load_state_dict(torch.load('/home/twang3/myWork/exalearn-inverse-application/pytorch_classifier/IEEEBigData_fingerprint_model.pt', map_location=torch.device('cpu')))
if multigpu:
    #dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    trained_model = nn.DataParallel(rnn)
lenet_trained_model = lenet_trained_model.to(device)

simple_xfer_model = XFerLearning(256,3)
#if args.phase > 0 and hvd.rank() == 0:
if args.phase > 0:
    simple_xfer_model.load_state_dict(torch.load(args.model_dir + "/simple_xfer_model_phase{}.pt".format(args.phase-1), map_location=torch.device('cpu')))
simple_xfer_model = simple_xfer_model.to(device)



#---------------------------setup optimizer with horovod------------------------

#optimizer = torch.optim.Adam(list(simple_xfer_model.parameters()) + list(lenet_trained_model.parameters()), lr=LEARNING_RATE)
#optimizer = torch.optim.Adam(list(simple_xfer_model.parameters()) + list(lenet_trained_model.parameters()), lr=LEARNING_RATE * hvd.size())
optimizer = torch.optim.Adam(list(simple_xfer_model.parameters()) + list(lenet_trained_model.parameters()), lr=LEARNING_RATE * np.sqrt(hvd.size()))

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(simple_xfer_model.state_dict(), root_rank=0)
hvd.broadcast_parameters(lenet_trained_model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=(list(simple_xfer_model.named_parameters()) + list(lenet_trained_model.named_parameters())))



criterion1 = torch.nn.MSELoss()
criterion2 = torch.nn.BCEWithLogitsLoss()


#------------------------------start training----------------------------------

print("the number of cpu threads: {}".format(torch.get_num_threads()))
t = time.time()

for epoch in range(1, args.epochs + 1):

    if(hvd.rank() == 0):
        print("Epoch: {:4}".format(epoch))

    simple_xfer_model.train()
    lenet_trained_model.train()
    train_sampler.set_epoch(epoch)

    for batch_idx, current_batch in enumerate(train_loader):      
    
        optimizer.zero_grad()
        inp = current_batch[0].to(device)
        xfer_features = lenet_trained_model.fingerprint(inp)
        class_output = lenet_trained_model.fc2(xfer_features)
        regression_output = simple_xfer_model(xfer_features)        
    
        current_batch_y = current_batch[1].to(device)
        regression_gndtruth = current_batch_y[:,0:3]
        loss1 = criterion1(regression_output, regression_gndtruth)
        class_gndtruth = current_batch_y[:,3:6]
        loss2 = criterion2(class_output, class_gndtruth )
    
        loss = loss1 + loss2
        loss.backward()
    
        optimizer.step()
    
        elapsed = time.time() - t
    
        if (batch_idx % 20 == 0):
            print("Rank = ", hvd.rank(), "  Batch {:3}/{}: loss1: {:15.6f}, loss2: {:15.6f}, loss_tot: {:15.6f}".format(batch_idx, len(train_loader), loss1.item(), loss2.item(), loss.item()), " Time Taken: {:15.6f}".format(elapsed))
            t = time.time()

    if (epoch % 10 == 0):
        simple_xfer_model.eval()
        lenet_trained_model.eval()
        test_loss = 0.0
        for inp, current_batch_y in test_loader:
            xfer_features = lenet_trained_model.fingerprint(inp)            
            regression_output = simple_xfer_model(xfer_features)
            class_output = lenet_trained_model.fc2(xfer_features)
            test_loss += criterion1(regression_output, current_batch_y[:,0:3].to(device)).item()
        test_loss /= len(test_sampler)
        test_loss = metric_average(test_loss, 'avg_loss')
        if hvd.rank() == 0:
            print("Epoch = ", epoch, " with test Loss: {:15.6f}".format(test_loss))

if hvd.rank() == 0:
    torch.save(simple_xfer_model.state_dict(), args.model_dir + "/simple_xfer_model_phase{}.pt".format(args.phase))
    torch.save(lenet_trained_model.state_dict(), args.model_dir + "/lenet_model_phase{}.pt".format(args.phase))

##         fig,ax = plt.subplots(figsize=(10,5), nrows=1,ncols=3)
##         ax[0].plot(y_test_torch.cpu().detach().numpy()[:,0],regression_output.cpu().detach().numpy()[:,0])
##         ax[0].plot(y_test_torch.cpu().detach().numpy()[:,0],y_test_torch.cpu().detach().numpy()[:,0])
##         ax[1].plot(y_test_torch.cpu().detach().numpy()[:,1],regression_output.cpu().detach().numpy()[:,1])
##         ax[1].plot(y_test_torch.cpu().detach().numpy()[:,1],y_test_torch.cpu().detach().numpy()[:,1])
##         ax[2].plot(y_test_torch.cpu().detach().numpy()[:,2],regression_output.cpu().detach().numpy()[:,2])
##         ax[2].plot(y_test_torch.cpu().detach().numpy()[:,2],y_test_torch.cpu().detach().numpy()[:,2])
##         plt.show()
#    y_test_torch_cpu = y_test_torch.cpu().detach().numpy()
#    pred_labels_cpu = class_output.cpu().detach().numpy()
##         print(y_test_torch_cpu.shape, pred_labels_cpu.shape)
##         print(y_test_torch_cpu[1:10,3:6], pred_labels_cpu[1:10,:])
#    gnd_labels = np.argmax(y_test_torch_cpu[:,3:6], axis=1)
#    pred_labels = np.argmax(pred_labels_cpu, axis=1)
##         print(gnd_labels.shape,pred_labels.shape)
##         print(gnd_labels[1:10],pred_labels[1:10])
#    cm = confusion_matrix(gnd_labels,pred_labels)
#    print(cm)
#if ((index_epoch+1) % 50 == 0):
#    filename = 'models/mt_all_balanced_lenet_learning_regressor_'+ str(args.epochs) + '_' + str(index_epoch) + '_epoch.pkl'
#    torch.save(simple_xfer_model.state_dict(), filename)
#    filename = 'models/mt_all_balanced_lenet_learning_lenet_'+ str(args.epochs) + '_' + str(index_epoch) + '_epoch.pkl'
#    torch.save(lenet_trained_model.state_dict(), filename)






