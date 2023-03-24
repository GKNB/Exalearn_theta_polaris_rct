#!/bin/bash

model_dir=$1/Ex1/model/
mkdir -p ${model_dir}

python ./serial_multiphase_rct.py \
	--num_rank_sim_per_node 32 \
	--num_node 1 \
	--num_phase 1 \
	--num_epoch 250 \
	--config_root_dir $1/Ex1/configs/ \
	--data_root_dir $1/Ex1/data/ \
	--model_dir ${model_dir} \
       	--rank_in_max 32
