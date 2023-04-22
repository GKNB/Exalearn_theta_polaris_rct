#!/bin/sh
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug
#PBS -A CSC249ADCD08
#PBS -l filesystems=home:grand

export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB
export IBV_FORK_SAFE=1

export HDF5_USE_FILE_LOCKING=FALSE
export LD_LIBRARY_PATH=/home/twang3/g2full_polaris/GSASII/bindist:$LD_LIBRARY_PATH

echo $PWD

exp_root_dir="/grand/CSC249ADCD08/twang/real_work_polaris_gpu/experiment/small_example_v1/"
exp_idx=1
config_root_dir="${exp_root_dir}/Ex${exp_idx}/configs/"
phase_idx=0

CMD="python /grand/CSC249ADCD08/twang/real_work_polaris_gpu/mpi_sweep_hdf5_multi_sym_polaris.py \
	${config_root_dir}/configs_phase${phase_idx}/config_1001460_cubic.txt \
	${config_root_dir}/configs_phase${phase_idx}/config_1522004_trigonal_part1.txt \
	${config_root_dir}/configs_phase${phase_idx}/config_1522004_trigonal_part2.txt \
	${config_root_dir}/configs_phase${phase_idx}/config_1531431_tetragonal.txt \
	"
echo $CMD

NNODES=2
NRANKS_PER_NODE=24
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE}"

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --cpu-bind list:2:3:4:5:6:7:10:11:12:13:14:15:18:19:20:21:22:23:26:27:28:29:30:31 $CMD > ${exp_root_dir}/log_sim










#mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth python main.py --device gpu --epoch 600 --log-interval 10 --lr 0.0001 --num_workers 1




#mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth echo $PMI_SIZE $PMI_LOCAL_RANK $PMI_LOCAL_SIZE $PMI_RANK
#mpiexec -n 1 --ppn 1 --depth=${NDEPTH} --cpu-bind depth printenv | grep "PMI"
#mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth printenv | grep "PMI"
#mpiexec -n 1 --ppn 1 --depth=${NDEPTH} --cpu-bind depth echo ${PMI_SIZE}
#mpiexec -n 1 --ppn 1 --depth=${NDEPTH} --cpu-bind depth echo ${OMP_NUM_THREADS}












