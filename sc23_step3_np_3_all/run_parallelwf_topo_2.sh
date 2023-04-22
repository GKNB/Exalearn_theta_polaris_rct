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

num_phase=3
num_node=8
num_epoch=2000
num_sim_process_per_node=32
num_sim_process_per_node_parallel=24
active_learning_sleep_time_in_seconds=100

exp_root_dir="/grand/CSC249ADCD08/twang/real_work_polaris_gpu/sc23_step3_np_3_all/result/"
ckpt_root_dir="/grand/CSC249ADCD08/twang/real_work_polaris_gpu/sc23_step3_np_3_all/"
exp_idx=3

config_root_dir="${exp_root_dir}/Ex${exp_idx}/configs/"
data_root_dir="${exp_root_dir}/Ex${exp_idx}/data/"

sim_file=log_parallelwf_sim_${num_epoch}_epoch_data_230m_node_${num_node}_phase_${num_phase}_topo_2_v3
ml_file=log_parallelwf_ml_${num_epoch}_epoch_data_230m_node_${num_node}_phase_${num_phase}_topo_2_v3
al_file=log_parallelwf_al_${num_epoch}_epoch_data_230m_node_${num_node}_phase_${num_phase}_topo_2_v3

phase_idx=0


########################### Start Sim ###########################
	
CMD_SIM="python /grand/CSC249ADCD08/twang/real_work_polaris_gpu/mpi_sweep_hdf5_multi_sym_polaris.py \
	${config_root_dir}/configs_phase${phase_idx}/config_1001460_cubic.txt \
	${config_root_dir}/configs_phase${phase_idx}/config_1522004_trigonal_part1.txt \
	${config_root_dir}/configs_phase${phase_idx}/config_1522004_trigonal_part2.txt \
	${config_root_dir}/configs_phase${phase_idx}/config_1531431_tetragonal.txt \
	"
echo $CMD_SIM >> $sim_file
	
NNODES_SIM=${num_node}
NRANKS_PER_NODE_SIM=${num_sim_process_per_node}
NTOTRANKS_SIM=$(( NNODES_SIM * NRANKS_PER_NODE_SIM ))
echo "Sim: NUM_OF_NODES= ${NNODES_SIM} TOTAL_NUM_RANKS= ${NTOTRANKS_SIM} RANKS_PER_NODE= ${NRANKS_PER_NODE_SIM}" >> $sim_file
	
start1=$(date +%s.%N)
mpiexec -n ${NTOTRANKS_SIM} --ppn ${NRANKS_PER_NODE_SIM} --env OMP_NUM_THREADS=1 --depth=1 --cpu-bind depth ${CMD_SIM} >> $sim_file
end1=$(date +%s.%N)
runtime1=$( echo "$end1 - $start1" | bc -l )
echo "Time logging: running time for sim in seconds = ${runtime1}" >> $sim_file
	
########################### Sim Finish ###########################


########################### Start Parallel work ###########################

for((phase_idx=1;phase_idx<num_phase;phase_idx++))
do
	CMD_ML="python /grand/CSC249ADCD08/twang/real_work_polaris_gpu/mtnetwork-training-ddp.py \
		--batch_size=1024 \
		--device=gpu \
		--epoch=${num_epoch} \
		--phase=$((${phase_idx}-1)) \
		--log-interval=10 \
		--lr=0.0001 \
		--num_workers=1 \
		--data_root_dir=${data_root_dir} \
		--ckpt_dir=${ckpt_root_dir} \
		--rank_data_gen=${NTOTRANKS_SIM}
		"
	echo $CMD_ML >> $ml_file

	NNODES_ML=${num_node}
	NRANKS_PER_NODE_ML=4
	NTOTRANKS_ML=$(( NNODES_ML * NRANKS_PER_NODE_ML ))
	echo "ML: NUM_OF_NODES= ${NNODES_ML} TOTAL_NUM_RANKS= ${NTOTRANKS_ML} RANKS_PER_NODE= ${NRANKS_PER_NODE_ML}" >> $ml_file

	SHARED_FILE=/grand/CSC249ADCD08/twang/real_work_polaris_gpu/sc23_step3_np_3_all/sharedfile
	if [[ -f "${SHARED_FILE}" ]];
	then
		rm ${SHARED_FILE}
	fi

	CMD_SIM="python /grand/CSC249ADCD08/twang/real_work_polaris_gpu/mpi_sweep_hdf5_multi_sym_polaris.py \
		${config_root_dir}/configs_phase${phase_idx}/config_1001460_cubic.txt \
		${config_root_dir}/configs_phase${phase_idx}/config_1522004_trigonal_part1.txt \
		${config_root_dir}/configs_phase${phase_idx}/config_1522004_trigonal_part2.txt \
		${config_root_dir}/configs_phase${phase_idx}/config_1531431_tetragonal.txt \
		"
	echo $CMD_SIM >> $sim_file
	
	NNODES_SIM=${num_node}
	NRANKS_PER_NODE_SIM=${num_sim_process_per_node_parallel}
	NTOTRANKS_SIM=$(( NNODES_SIM * NRANKS_PER_NODE_SIM ))
	echo "Sim: NUM_OF_NODES= ${NNODES_SIM} TOTAL_NUM_RANKS= ${NTOTRANKS_SIM} RANKS_PER_NODE= ${NRANKS_PER_NODE_SIM}" >> $sim_file

	(	
	start1=$(date +%s.%N); \
	mpiexec -n ${NTOTRANKS_SIM} --ppn ${NRANKS_PER_NODE_SIM} --env OMP_NUM_THREADS=1 --cpu-bind list:2:3:4:5:6:7:10:11:12:13:14:15:18:19:20:21:22:23:26:27:28:29:30:31 ${CMD_SIM} >> $sim_file; \
	end1=$(date +%s.%N); \	
	runtime1=$( echo "$end1 - $start1" | bc -l ); \
	echo "Time logging: running time for sim in seconds = ${runtime1}" >> $sim_file; \
	if [[ $phase_idx != $(($num_phase - 1)) ]]; then start3=$(date +%s.%N); mpiexec -n 1 --ppn 1 --cpu-bind list:0 sleep $active_learning_sleep_time_in_seconds; end3=$(date +%s.%N); runtime3=$( echo "$end3 - $start3" | bc -l ); echo "Time logging: running time for active learning in seconds = ${runtime3}" >> $al_file; fi
	)&
	
	(
	start2=$(date +%s.%N); \
	mpiexec -n ${NTOTRANKS_ML} --ppn ${NRANKS_PER_NODE_ML} --env OMP_NUM_THREADS=2 --cpu-bind list:0,1:8,9:16,17:24,25 ./set_affinity_gpu_polaris.sh $CMD_ML >> $ml_file; \
	end2=$(date +%s.%N); \
	runtime2=$( echo "$end2 - $start2" | bc -l ); \
	echo "Time logging: running time for ML in seconds = ${runtime2}" >> $ml_file
	)&
	wait
done

########################### Parallel work Finish ###########################

CMD_ML="python /grand/CSC249ADCD08/twang/real_work_polaris_gpu/mtnetwork-training-ddp.py \
		--batch_size=1024 \
		--device=gpu \
		--epoch=${num_epoch} \
		--phase=$((${num_phase}-1)) \
		--log-interval=10 \
		--lr=0.0001 \
		--num_workers=1 \
		--data_root_dir=${data_root_dir} \
		--ckpt_dir=${ckpt_root_dir} \
		--rank_data_gen=${NTOTRANKS_SIM}
		"
echo $CMD_ML >> $ml_file

NNODES_ML=${num_node}
NRANKS_PER_NODE_ML=4
NDEPTH=8
NTOTRANKS_ML=$(( NNODES_ML * NRANKS_PER_NODE_ML ))
echo "ML: NUM_OF_NODES= ${NNODES_ML} TOTAL_NUM_RANKS= ${NTOTRANKS_ML} RANKS_PER_NODE= ${NRANKS_PER_NODE_ML}" >> $ml_file

SHARED_FILE=/grand/CSC249ADCD08/twang/real_work_polaris_gpu/sc23_step3_np_3_all/sharedfile
if [[ -f "${SHARED_FILE}" ]];
then
	rm ${SHARED_FILE}
fi

start2=$(date +%s.%N)
mpiexec -n ${NTOTRANKS_ML} --ppn ${NRANKS_PER_NODE_ML} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=8 ./set_affinity_gpu_polaris.sh $CMD_ML >> $ml_file
end2=$(date +%s.%N)
runtime2=$( echo "$end2 - $start2" | bc -l )
echo "Time logging: running time for ML in seconds = ${runtime2}" >> $ml_file
	

end0=$(date +%s.%N)
runtime0=$( echo "$end0 - $start0" | bc -l )
echo "Time logging: Total running time in seconds = ${runtime0}"
