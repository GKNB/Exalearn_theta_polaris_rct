from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import numpy as np
import h5py
import time
from collections import deque


#'root_path' is the data directory
#'path_in_list' is the directory name for each class
#'filename_in_list' is filename format for each class
#'rank_max_list' is the maximun rank for each class
#'do_saving' decide if we want to save data
#'filename_prefix' describe the filename, which should be filename_prefix-all-Y.npy and filename_prefix-all-P.npy
def read_and_merge_data(root_path, path_in_list, filename_in_list, rank_max_list, do_saving=False, filename_prefix="Output"):

    start = time.time()
    t_stage = -time.time()
    y_all = []
    X_all = []

    for si in range(4):
        X = deque()
        y = deque()
        sz = 0
        rank_max = rank_max_list[si]
        path_in = path_in_list[si]
        filename_in = filename_in_list[si]
    
        for ri in range(rank_max):
            with h5py.File(root_path + path_in + filename_in + str(ri) + '.hdf5', 'r') as f:
                dhisto = f['histograms']
                X_sub = dhisto[:, 1, :]
                X_shape = X_sub.shape
                dparams = f['parameters']
                y_sub = dparams[:]
                y_shape = y_sub.shape
    
                X.append(X_sub)
                y.append(y_sub)
                sz += y_shape[0]

        X_shape = list(X_shape)
        y_shape = list(y_shape)
        X_shape[0] = sz
        y_shape[0] = sz
        X_res = np.empty(tuple(X_shape))
        y_res = np.empty(tuple(y_shape))
    
        sz_tot = 0
        for ri in range(rank_max):
            X_temp = X.popleft()
            y_temp = y.popleft()
            sz_temp = y_temp.shape[0]
    
            X_res[sz_tot:sz_tot+sz_temp] = X_temp
            y_res[sz_tot:sz_tot+sz_temp] = y_temp
            sz_tot += sz_temp

        X_all.append(X_res)
        y_all.append(y_res)
    
    t_stage += time.time()
    print("Read and merge hdf5 data takes {}".format(t_stage))
    t_stage = -time.time()


#---------------------------merge two trigonal ranges together-----------------------------
    
    X_all[1] = np.concatenate((X_all[1], X_all[2]), axis=0)
    X_all.pop(2)
    y_all[1] = np.concatenate((y_all[1], y_all[2]), axis=0)
    y_all.pop(2)
    
    t_stage += time.time()
    print("Merge two trigonal ranges takes {}".format(t_stage))
    t_stage = -time.time()

    
#----------------------normalize X-data, concatenate and save to file------------------------
    
    for si in range(3):
        scaler = MinMaxScaler(copy=True)
        X_all[si] = scaler.fit_transform(X_all[si].T)
        X_all[si] = X_all[si].T
    
    X_all[0] = np.tile(X_all[0], (int(X_all[2].shape[0] / X_all[0].shape[0]), 1))
    X_all[1] = np.tile(X_all[1], (2, 1))
    for si in range(3):
        print('si = ', si, ' Shape of X_all read: ', X_all[si].shape)

    X_final = np.concatenate([X_all[0], X_all[1], X_all[2]], axis=0)
    print(X_final.shape)

    t_stage += time.time()
    print("Normalize and concatnate X-data takes {}".format(t_stage))

    if do_saving:
        with open(filename_prefix + '-all-Y.npy', 'wb') as f:
            np.save(f, X_final)
            

    t_stage = -time.time()
    
#------------------------create Y-data with size n*6 and save to file-----------------------
# 0 for cubic,      [a, a, 1.5708, 0, 0, 1] 
# 1 for trigonal    [a, a, gamma,  1, 0, 0]
# 2 for tetragonal  [a, b, 1.5708, 0, 1, 0]
    
    y_cubic         = y_all[0]
    print("before processing, y_cubic has shape: ", y_cubic.shape)
    class_label_cubic = np.array([0, 0, 1])
    class_label_cubic = np.tile(class_label_cubic, (y_cubic.shape[0], 1))
    y_cubic_angle = np.ones_like(y_cubic) * 1.57079632679
    y_cubic = np.concatenate([y_cubic, y_cubic, y_cubic_angle, class_label_cubic], axis=1)
    print("after processing, y_cubic has shape: ", y_cubic.shape)
    #for i in range(0, y_cubic.shape[0], int(y_cubic.shape[0] / 16)):
    #    print("cubic ", y_cubic[i])
    
    y_trigonal      = y_all[1]
    print("before processing, y_trigonal has shape: ", y_trigonal.shape)
    class_label_trigonal = np.array([1, 0, 0])
    class_label_trigonal = np.tile(class_label_trigonal, (y_trigonal.shape[0], 1))
    y_trigonal_side = y_trigonal[:,0]
    print(y_trigonal_side.shape)
    y_trigonal_side = np.reshape(y_trigonal_side, (-1, 1))
    print(y_trigonal_side.shape)
    y_trigonal = np.concatenate([y_trigonal, y_trigonal_side, class_label_trigonal], axis=1)
    y_trigonal[:, [1, 2]] = y_trigonal[:, [2, 1]]
    y_trigonal[:, 2] = np.deg2rad(y_trigonal[:, 2])
    print("after processing, y_trigonal has shape: ", y_trigonal.shape)
    #for i in range(0, y_trigonal.shape[0], int(y_trigonal.shape[0] / 16)):
    #    print("trigonal ", y_trigonal[i])
    
    y_tetragonal    = y_all[2]
    print("before processing, y_tetragonal has shape: ", y_tetragonal.shape)
    class_label_tetragonal = np.array([0, 1, 0])
    class_label_tetragonal = np.tile(class_label_tetragonal, (y_tetragonal.shape[0], 1))
    y_tetragonal_angle = np.ones((y_tetragonal.shape[0], 1)) * 1.57079632679
    y_tetragonal = np.concatenate([y_tetragonal, y_tetragonal_angle, class_label_tetragonal], axis=1)
    print("after processing, y_tetragonal has shape: ", y_tetragonal.shape)
    #for i in range(0, y_tetragonal.shape[0], int(y_tetragonal.shape[0] / 16)):
    #    print("tetragonal ", y_tetragonal[i])
    
    y_cubic = np.tile(y_cubic, (int(y_tetragonal.shape[0] / y_cubic.shape[0]), 1))
    y_trigonal = np.tile(y_trigonal, (2, 1))
    print("cubic has shape ", y_cubic.shape)
    print("trigonal has shape ", y_trigonal.shape)
    print("tetragonal has shape ", y_tetragonal.shape)
    
    y_final = np.concatenate([y_cubic, y_trigonal, y_tetragonal], axis=0)
    print(y_final.shape)

    t_stage += time.time()
    print("Create, normalize and concatnate y-data takes {}".format(t_stage))

    if do_saving:
        with open(filename_prefix + '-all-P.npy', 'wb') as f:
            np.save(f, y_final)
    
    end = time.time()
    print("Total running time for merging = ", end - start)

    return X_final, y_final
