#!/bin/sh
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug
#PBS -A CSC249ADCD08
#PBS -l filesystems=home:grand

# MPI example w/ 4 MPI ranks per node spread evenly across cores
NNODES=2
NRANKS_PER_NODE=4
NDEPTH=8
NTHREADS=1

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB
export IBV_FORK_SAFE=1

echo $PWD

CMD="python /grand/CSC249ADCD08/twang/real_work_polaris_gpu/mtnetwork-training-ddp.py \
	--device=gpu \
	--epoch=250 \
	--phase=0 \
	--log-interval=10 \
	--lr=0.0001 \
	--num_workers=1 \
	--data_root_dir='./' \
	--model_dir='./' \
	--rank_data_gen=56 \
	--rank_in_max=28 \
	"

echo $CMD
rm /grand/CSC249ADCD08/twang/real_work_polaris_gpu/sharedfile

#mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth ./set_affinity_gpu_polaris.sh $CMD
#mpiexec -n ${NTOTRANKS} --ppn 4 --cpu-bind list:0:8:16:24 ./set_affinity_gpu_polaris.sh $CMD
#mpiexec -n ${NTOTRANKS} --ppn 4 --cpu-bind list:0,1:8,9:16,17:24,25 ./set_affinity_gpu_polaris.sh $CMD
#mpiexec -n ${NTOTRANKS} --ppn 4 --cpu-bind list:0,1,2:8,9,10:16,17,18:24,25,26 ./set_affinity_gpu_polaris.sh $CMD
#mpiexec -n ${NTOTRANKS} --ppn 4 --cpu-bind list:0,1,2,3:8,9,10,11:16,17,18,19:24,25,26,27 ./set_affinity_gpu_polaris.sh $CMD
#mpiexec -n ${NTOTRANKS} --ppn 4 --cpu-bind list:0,1,2,3,4:8,9,10,11,12:16,17,18,19,20:24,25,26,27,28 ./set_affinity_gpu_polaris.sh $CMD
#mpiexec -n ${NTOTRANKS} --ppn 4 --cpu-bind list:0,1,2,3,4,5:8,9,10,11,12,13:16,17,18,19,20,21:24,25,26,27,28,29 ./set_affinity_gpu_polaris.sh $CMD
#mpiexec -n ${NTOTRANKS} --ppn 4 --cpu-bind list:0,1,2,3,4,5,6:8,9,10,11,12,13,14:16,17,18,19,20,21,22:24,25,26,27,28,29,30 ./set_affinity_gpu_polaris.sh $CMD
#mpiexec -n ${NTOTRANKS} --ppn 4 --cpu-bind list:0,1,2,3,4,5,6,7:8,9,10,11,12,13,14,15:16,17,18,19,20,21,22,23:24,25,26,27,28,29,30,31 ./set_affinity_gpu_polaris.sh $CMD





#mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth python main.py --device gpu --epoch 600 --log-interval 10 --lr 0.0001 --num_workers 1




#mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth echo $PMI_SIZE $PMI_LOCAL_RANK $PMI_LOCAL_SIZE $PMI_RANK
#mpiexec -n 1 --ppn 1 --depth=${NDEPTH} --cpu-bind depth printenv | grep "PMI"
#mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth printenv | grep "PMI"
#mpiexec -n 1 --ppn 1 --depth=${NDEPTH} --cpu-bind depth echo ${PMI_SIZE}
#mpiexec -n 1 --ppn 1 --depth=${NDEPTH} --cpu-bind depth echo ${OMP_NUM_THREADS}












