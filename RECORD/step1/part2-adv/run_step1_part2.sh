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

num_node=8
num_sim_process_per_node=32

exp_root_dir="/grand/CSC249ADCD08/twang/real_work_polaris_gpu/sc23_step1_all/result/"
exp_idx=1

config_root_dir="${exp_root_dir}/Ex${exp_idx}/configs/"
phase_idx=0


########################### Start Sim ###########################

echo "Start doing simulation!"

CMD_SIM="python /grand/CSC249ADCD08/twang/real_work_polaris_gpu/mpi_sweep_hdf5_multi_sym_polaris.py \
	${config_root_dir}/configs_phase${phase_idx}/config_1001460_cubic.txt \
	${config_root_dir}/configs_phase${phase_idx}/config_1522004_trigonal_part1.txt \
	${config_root_dir}/configs_phase${phase_idx}/config_1522004_trigonal_part2.txt \
	${config_root_dir}/configs_phase${phase_idx}/config_1531431_tetragonal.txt \
	"
echo $CMD_SIM

NNODES=${num_node}
NRANKS_PER_NODE=${num_sim_process_per_node}
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "Sim: NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE}"

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=1 --cpu-bind depth ${CMD_SIM}

echo "Sim is finished!"

########################### Sim Finish ###########################
