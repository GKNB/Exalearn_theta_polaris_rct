import numpy as np

#X_scaled = np.load('/home/twang3/myWork/multitask_all_Y_balanced.npy')
#y_scaled = np.load('/home/twang3/myWork/multitask_all_P_balanced.npy')

X_scaled = np.load('/home/twang3/myWork/exalearn_project/run_theta/baseline/ml/Output-all-Y.npy')
y_scaled = np.load('/home/twang3/myWork/exalearn_project/run_theta/baseline/ml/Output-all-P.npy')

X_scaled = np.float32(X_scaled)
y_scaled = np.float32(y_scaled)

print(X_scaled.shape, y_scaled.shape)

print(X_scaled.min(), X_scaled.max(), y_scaled.min(axis=0), y_scaled.max(axis=0))


for i in range(0, y_scaled.shape[0], int(y_scaled.shape[0] / 64)):
    print(y_scaled[i])
