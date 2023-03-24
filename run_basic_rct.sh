#!/bin/bash

model_dir=$1/Ex2/model/
mkdir -p ${model_dir}

python ./serial_multiphase_rct.py \
	--num_rank_sim 512 \
	--num_node_ml 16 \
	--num_phase 1 \
	--num_epoch 250 \
	--config_root_dir $1/Ex2/configs/ \
	--data_root_dir $1/Ex2/data/ \
	--model_dir ${model_dir} \
       	--rank_in_max 128
