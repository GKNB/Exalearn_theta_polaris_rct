#!/bin/sh
#PBS -l select=8:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q preemptable
#PBS -A CSC249ADCD08
#PBS -l filesystems=home:grand

export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB
export IBV_FORK_SAFE=1
export HDF5_USE_FILE_LOCKING=FALSE
export LD_LIBRARY_PATH=/home/twang3/g2full_polaris/GSASII/bindist:$LD_LIBRARY_PATH

num_node=$1

exp_root_dir="/grand/CSC249ADCD08/twang/real_work_polaris_gpu/sc23_step1_all/result/"
exp_idx=1

config_root_dir="${exp_root_dir}/Ex${exp_idx}/configs/"
phase_idx=0

########################### Start ML ###########################

echo "Start doing simulation!"

CMD_ML="python /grand/CSC249ADCD08/twang/real_work_polaris_gpu/sc23_step1_all/mtnetwork-training-ddp.py \
	--batch_size=1024 \
	--device=gpu \
	--epoch=40 \
	--phase=0 \
	--log-interval=10 \
	--lr=0.0001 \
	--num_workers=1 \
	--data_root_dir='./' \
	--model_dir='./' \
	--rank_data_gen=256
	"
echo $CMD_ML

SHARED_FILE=/grand/CSC249ADCD08/twang/real_work_polaris_gpu/sc23_step1_all/sharedfile
if [[ -f "${SHARED_FILE}" ]];
then
	rm ${SHARED_FILE}
fi

NNODES=${num_node}
NRANKS_PER_NODE=4
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "Sim: NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE}"


#mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --cpu-bind list:0:8:16:24 ./set_affinity_gpu_polaris.sh $CMD_ML
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --cpu-bind list:0,1:8,9:16,17:24,25 ./set_affinity_gpu_polaris.sh $CMD_ML
#mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --cpu-bind list:0,1,2:8,9,10:16,17,18:24,25,26 ./set_affinity_gpu_polaris.sh $CMD_ML
#mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --cpu-bind list:0,1,2,3:8,9,10,11:16,17,18,19:24,25,26,27 ./set_affinity_gpu_polaris.sh $CMD_ML
#mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --cpu-bind list:0,1,2,3,4:8,9,10,11,12:16,17,18,19,20:24,25,26,27,28 ./set_affinity_gpu_polaris.sh $CMD_ML
#mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --cpu-bind list:0,1,2,3,4,5:8,9,10,11,12,13:16,17,18,19,20,21:24,25,26,27,28,29 ./set_affinity_gpu_polaris.sh $CMD_ML
#mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --cpu-bind list:0,1,2,3,4,5,6:8,9,10,11,12,13,14:16,17,18,19,20,21,22:24,25,26,27,28,29,30 ./set_affinity_gpu_polaris.sh $CMD_ML
#mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --cpu-bind list:0,1,2,3,4,5,6,7:8,9,10,11,12,13,14,15:16,17,18,19,20,21,22,23:24,25,26,27,28,29,30,31 ./set_affinity_gpu_polaris.sh $CMD_ML


echo "ML is finished!"

########################### ML Finish ###########################
