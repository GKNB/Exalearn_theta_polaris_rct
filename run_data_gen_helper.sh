#!/bin/bash

exp_dir_name=$1
exp_idx=$2
num_phase=$3

python ./data_gen_helper.py \
	${exp_dir_name}/Ex${exp_idx}/configs/ \
	2.5 5.5 0.0001 \
	2.5 5.5 0.002 \
	20 88 0.2 \
	2.5 5.5 0.002 \
	92 120 0.2 \
	3.5 4.5 0.0005 \
	3.5 4.5 0.001 \
	$num_phase \
	/grand/CSC249ADCD08/twang/cif_file_in/ \
	${exp_dir_name}/Ex${exp_idx}/data

# line 3: where to generate config file
# line 4: experiment idx, shall be 1(nosplit), 2(serial), 3(parallel)
# line 5: num of phases, shall be 1 for Ex1, and n for Ex2/3
# line 17: where are cif file, should be fixed and better not to change later
# line 18: where to output hdf5 data
