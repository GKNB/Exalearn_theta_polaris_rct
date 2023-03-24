from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import numpy as np
import h5py
import time


#'root_path' is the data directory
#'path_in_list' is the directory name for each class
#'filename_in_list' is filename format for each class
#'rank_max_list' is the maximun rank for each class
#'rank_in_max' is the inner block size of merging
#'do_saving' decide if we want to save data
#'filename_prefix' describe the filename, which should be filename_prefix-all-Y.npy and filename_prefix-all-P.npy
def read_and_merge_data(root_path, path_in_list, filename_in_list, rank_max_list, rank_in_max, do_saving=False, filename_prefix="Output"):

    start = time.time()
    y_all = []
    X_all = []

#---------------------------merge all ranks together-----------------------------

    for si in range(4):
    
        rank_max = rank_max_list[si]
        assert (int(rank_max) % int(rank_in_max) == 0), "rank_max must be integer multiple of rank_in_max"
        
        path_in = path_in_list[si]
        filename_in = filename_in_list[si]
    
        for ri_out in range(0, int(rank_max / rank_in_max)):
            for ri_in in range(ri_out * rank_in_max, (ri_out + 1) * rank_in_max):
                
                with h5py.File(root_path + path_in + filename_in + str(ri_in) + '.hdf5', 'r') as f:
                    dhisto = f['histograms']
                    X_sub = dhisto[:, 1, :]
                    dparams = f['parameters']            
                    y_sub = dparams[:]
        
                    if ri_in == ri_out * rank_in_max:
                        X = X_sub
                        y = y_sub
                    else:
                        X = np.concatenate((X, X_sub[:]), axis=0)
                        y = np.concatenate((y, y_sub[:]), axis=0)
        
            if ri_out == 0:
                X_all.append(X)
                y_all.append(y)
            else:
                X_all[si] = np.concatenate((X_all[si], X[:]), axis=0)
                y_all[si] = np.concatenate((y_all[si], y[:]), axis=0)
    
        print('si = ', si, ' Shape of X_all read: ', X_all[si].shape)
        print('si = ', si, ' Shape of y_all read: ', y_all[si].shape)
    
#---------------------------merge two trigonal ranges together-----------------------------
    
    X_all[1] = np.concatenate((X_all[1], X_all[2]), axis=0)
    X_all.pop(2)
    y_all[1] = np.concatenate((y_all[1], y_all[2]), axis=0)
    y_all.pop(2)
    
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

    if do_saving:
        with open(filename_prefix + '-all-Y.npy', 'wb') as f:
            np.save(f, X_final)
    
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

    if do_saving:
        with open(filename_prefix + '-all-P.npy', 'wb') as f:
            np.save(f, y_final)
    
    end = time.time()
    print("Total running time for merging = ", end - start)

    return X_final, y_final
