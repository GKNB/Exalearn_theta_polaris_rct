#!/bin/bash

model_dir=$1/Ex4/model/
mkdir -p ${model_dir}

python ./parallel_multiphase_rct.py \
	--num_rank_sim 512 \
	--num_node_ml 16 \
	--num_phase 4 \
	--num_epoch 250 \
	--config_root_dir $1/Ex4/configs/\
	--data_root_dir $1/Ex4/data/ \
	--model_dir ${model_dir} \
       	--rank_in_max 128
