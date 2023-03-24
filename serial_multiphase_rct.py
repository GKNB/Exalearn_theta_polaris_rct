from radical import entk
import os
import argparse, sys, math

class MVP(object):

    def __init__(self):
        self.set_argparse()
        self.am = entk.AppManager()

    def set_resource(self, res_desc):
        self.am.resource_desc = res_desc

    def set_argparse(self):
        parser = argparse.ArgumentParser(description="Exalearn_MVP_EnTK_serial")

        parser.add_argument("--num_rank_sim_per_node", type=int, default=1,
                            help = "number of MPI ranks for sim per node")
        parser.add_argument("--num_node", type=int, default=1,
                            help = "number of nodes for entire job")
        parser.add_argument("--num_phase", type=int, default=1,
                            help = "number of phases")
        parser.add_argument("--num_epoch", type=int, default=50,
                            help = "number of epochs")
        parser.add_argument("--config_root_dir", required=True,
                            help = "root directory of configs")
        parser.add_argument("--data_root_dir", required=True,
                            help = "root directory of data")
        parser.add_argument("--model_dir", default='./',
                            help = "directory of model")
        parser.add_argument("--rank_in_max", type=int, required=True,
                            help = "inner blocksize for merging")

        args = parser.parse_args()
        self.args = args

    # This is for simulation, return a stage which has a single sim task
    def run_mpi_sweep_hdf5_py(self, phase_idx):

        nproc = int(self.args.num_rank_sim_per_node * self.args.num_node)
        pre_exec_list = [
            "module load conda",
            "export HDF5_USE_FILE_LOCKING=FALSE",
            "export LD_LIBRARY_PATH=/home/twang3/g2full_polaris/GSASII/bindist:$LD_LIBRARY_PATH",
            "echo $LD_LIBRARY_PATH",
            "export OMP_NUM_THREADS=1"
            ]

        t = entk.Task()
        t.pre_exec = pre_exec_list
        t.executable = 'python'
        t.arguments = ['/grand/CSC249ADCD08/twang/real_work_polaris_gpu/mpi_sweep_hdf5_multi_sym_polaris.py',
                       '{}/configs_phase{}/config_1001460_cubic.txt'.format(self.args.config_root_dir, phase_idx),
                       '{}/configs_phase{}/config_1522004_trigonal_part1.txt'.format(self.args.config_root_dir, phase_idx),
                       '{}/configs_phase{}/config_1522004_trigonal_part2.txt'.format(self.args.config_root_dir, phase_idx),
                       '{}/configs_phase{}/config_1531431_tetragonal.txt'.format(self.args.config_root_dir, phase_idx)]
        t.post_exec = []
        t.cpu_reqs = {
            'cpu_processes': nproc,
            'cpu_process_type': 'MPI',
            'cpu_threads': 1,
            'cpu_thread_type': 'OpenMP'
        }
        
        s = entk.Stage()
        s.add_tasks(t)
        return s


    # This is for training, return a stage which has a single training task
    def run_mtnetwork_training_ddp_py(self, phase_idx):

        n_node = int(self.args.num_node)
        n_proc = n_node * 4
        
        t = entk.Task()
        t.pre_exec = ['module load conda']
        t.executable = 'python'
        t.arguments = ['/grand/CSC249ADCD08/twang/real_work_polaris_gpu/mtnetwork-training-ddp.py',
                       '--device=gpu',
                       '--epoch={}'.format(self.args.num_epoch),
                       '--phase={}'.format(phase_idx),
                       '--log-interval=10',
                       '--lr=0.0001',
                       '--num_workers=1',
                       '--data_root_dir={}'.format(self.args.data_root_dir),
                       '--model_dir={}'.format(self.args.model_dir),
                       '--rank_data_gen={}'.format(self.args.num_rank_sim_per_node * self.args.num_node),
                       '--rank_in_max={}'.format(self.args.rank_in_max)]
        t.post_exec = []
        t.cpu_reqs = {
                'cpu_processes'    : n_proc,
                'cpu_process_type' : 'MPI',
                'cpu_threads'      : 1,
                'cpu_thread_type'  : None
                }

        t.gpu_reqs = {
                'gpu_processes'     : 1,
                'gpu_process_type'  : 'CUDA',
                'gpu_threads'       : 1,
                'gpu_thread_type'   : None
                }

        s = entk.Stage()
        s.add_tasks(t)
        return s


    def generate_pipeline(self):
        
        p = entk.Pipeline()
        for phase in range(int(self.args.num_phase)):
#            s1 = self.run_mpi_sweep_hdf5_py(phase)
#            p.add_stages(s1)
            s2 = self.run_mtnetwork_training_ddp_py(phase)
            p.add_stages(s2)
        return p

    def run_workflow(self):
        p = self.generate_pipeline()
        self.am.workflow = [p]
        self.am.run()


if __name__ == "__main__":

    mvp = MVP()
    n_nodes = mvp.args.num_node
    mvp.set_resource(res_desc = {
        'resource': 'anl.polaris',
        'queue'   : 'debug',
#        'queue'   : 'default',
        'walltime': 30, #MIN
        'cpus'    : 64 * n_nodes,
        'gpus'    : 4 * n_nodes,
        'project' : 'CSC249ADCD08'
        })
    mvp.run_workflow()
