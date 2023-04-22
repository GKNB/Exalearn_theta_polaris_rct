#!/bin/bash

########################### Parameters we need to modify #########################

exp_dir_name=/grand/CSC249ADCD08/twang/real_work_polaris_gpu/sc23_step1_all/result/
num_phase=4

##################################################################################

echo "Please make sure that run_data_gen_helper.sh has been set correctly!"
cat ./run_data_gen_helper.sh

if [ -d $exp_dir_name ] 
then
	echo "Directory $exp_dir_name exists."
else
	mkdir -p $exp_dir_name
	cp ./run_data_gen_helper.sh $exp_dir_name
	echo "num_phase = $num_phase, exp_dir_name= $exp_dir_name" | tee -a $exp_dir_name/preset_log
	./run_data_gen_helper.sh $exp_dir_name 1 1 | tee -a $exp_dir_name/preset_log
	./run_data_gen_helper.sh $exp_dir_name 2 $num_phase | tee -a $exp_dir_name/preset_log
	./run_data_gen_helper.sh $exp_dir_name 3 $num_phase | tee -a $exp_dir_name/preset_log
fi

