import numpy as np

import old_preprocess_data as predata1
import preprocess_data as predata2

import os, psutil
process = psutil.Process(os.getpid())

root_path = "/grand/CSC249ADCD08/twang/real_work_polaris_gpu/sc23_step2_all/result/Ex2/data/phase0/"
path_in_list = ['test_cubic/',
                'test_trigonal_part1/',
                'test_trigonal_part2/',
                'test_tetragonal/']
filename_in_list = ['cubic_1001460_cubic_part',
                    'trigonal_1522004_trigonal_part',
                    'trigonal_1522004_trigonal_part',
                    'tetragonal_1531431_tetragonal_part']
rank_max_list = [128, 128, 128, 128]

#print("Start test old")
#X_old, y_old = predata1.read_and_merge_data(root_path, path_in_list, filename_in_list, rank_max_list, do_saving=False, filename_prefix="Output")
#print(process.memory_info().rss / 1024 / 1024, " MB")
print("Start test new")
X_new, y_new = predata2.read_and_merge_data(root_path, path_in_list, filename_in_list, rank_max_list, do_saving=False, filename_prefix="Output")
print(process.memory_info().rss / 1024 / 1024, " MB")
#print(np.array_equal(X_old, X_new))
#print(np.array_equal(y_old, y_new))
