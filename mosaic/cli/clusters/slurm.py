
import os

from .hostlist import expand_hostlist


__all__ = ['node_list', 'submission_script']


def node_list(host_name):
    """
    Attempt to find a node list for SLURM clusters.

    Parameters
    ----------
    host_name

    Returns
    -------

    """
    slurm_nodes = os.environ.get('SLURM_NODELIST', None)

    if slurm_nodes is None:
        return

    slurm_list = expand_hostlist(slurm_nodes)
    print(slurm_list)
    slurm_list.remove(host_name)

    return slurm_list


def submission_script(name, num_nodes, num_workers, num_threads, node_memory):
    """
    Generate a submission script for SLURM clusters.

    Parameters
    ----------
    name
    num_nodes
    num_workers
    num_threads
    node_memory

    Returns
    -------
    str

    """

    return f"""#!/bin/bash -l
#SBATCH --job-name={name}
#SBATCH --time=48:00:0
#SBATCH --nodes={num_nodes+1}
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task={num_workers*num_threads}
#SBATCH --mem={node_memory}G
#SBATCH -o out.log
#SBATCH -e err.log

#SBATCH --account=<budget_allocation>
#SBATCH --partition=<partition>
#SBATCH --qos=<quality_of_service>

name={name}
num_nodes={num_nodes}
num_workers_per_node={num_workers}
num_threads_per_worker={num_threads}

# load any modules before activating the conda env
# for example:
# module load anaconda3/personal

# activate conda environment
conda activate stride

# set number of threads per process
# use $(ppn) to use one worker per node and as many threads pr worker as cores in the node
export OMP_NUM_THREADS=$num_threads_per_worker
export OMP_PLACES=cores

# set any environment variables
# for example:
# export DEVITO_COMPILER=icc

# add any commands to be executed in the remote node before starting the runtime
# for example:
# export SSH_FLAGS="source /etc/profile; module load anaconda3/personal; conda activate stride"

# run our job
cd $SLURM_SUBMIT_DIR
ls -l
date
mrun -n $num_nodes -nw $num_workers_per_node -nth $num_threads_per_worker python forward.py &> $name-output.log
date

stat=$?
echo "Exit status: $stat" >> "$name-output.log"
"""
