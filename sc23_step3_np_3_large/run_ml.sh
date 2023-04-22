#!/bin/sh
#PBS -l select=4:system=polaris
#PBS -l place=scatter
#PBS -l walltime=2:00:00
#PBS -q preemptable
#PBS -A CSC249ADCD08
#PBS -l filesystems=home:grand

start0=$(date +%s.%N)

export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB
export IBV_FORK_SAFE=1
export HDF5_USE_FILE_LOCKING=FALSE
export LD_LIBRARY_PATH=/home/twang3/g2full_polaris/GSASII/bindist:$LD_LIBRARY_PATH

num_phase=1
num_node=1
num_epoch=500
num_sim_process_per_node=32
active_learning_sleep_time_in_seconds=1

exp_root_dir="/grand/CSC249ADCD08/twang/real_work_polaris_gpu/sc23_step3_np_3_large/result/"
ckpt_root_dir="/grand/CSC249ADCD08/twang/real_work_polaris_gpu/sc23_step3_np_3_large/"
exp_idx=2

for((phase_idx=0;phase_idx<num_phase;phase_idx++))
do
	data_root_dir="${exp_root_dir}/Ex${exp_idx}/data/"
	
	CMD_ML="python /grand/CSC249ADCD08/twang/real_work_polaris_gpu/mtnetwork-training-ddp.py \
		--batch_size=512 \
		--device=gpu \
		--epoch=${num_epoch} \
		--phase=${phase_idx} \
		--log-interval=1 \
		--lr=0.0002 \
		--num_workers=1 \
		--data_root_dir=${data_root_dir} \
		--ckpt_dir=${ckpt_root_dir} \
		--rank_data_gen=256
		"
	echo $CMD_ML
	
	SHARED_FILE=/grand/CSC249ADCD08/twang/real_work_polaris_gpu/sc23_step3_np_3_large/sharedfile
	if [[ -f "${SHARED_FILE}" ]];
	then
		rm ${SHARED_FILE}
	fi
	
	NNODES_ML=${num_node}
	NRANKS_PER_NODE_ML=4
	NDEPTH=8
	NTOTRANKS_ML=$(( NNODES_ML * NRANKS_PER_NODE_ML ))
	echo "ML: NUM_OF_NODES= ${NNODES_ML} TOTAL_NUM_RANKS= ${NTOTRANKS_ML} RANKS_PER_NODE= ${NRANKS_PER_NODE_ML}"
	
	start2=$(date +%s.%N)
	mpiexec -n ${NTOTRANKS_ML} --ppn ${NRANKS_PER_NODE_ML} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=8 ./set_affinity_gpu_polaris.sh $CMD_ML
	end2=$(date +%s.%N)
	
	echo "ML is finished!"
	runtime2=$( echo "$end2 - $start2" | bc -l )
	echo "Time logging: running time for ML in seconds = ${runtime2}"
	
	########################### ML Finish ###########################
	
done
	
end0=$(date +%s.%N)
runtime0=$( echo "$end0 - $start0" | bc -l )
echo "Time logging: Total running time in seconds = ${runtime0}"
