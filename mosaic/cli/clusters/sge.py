
import os


__all__ = ['node_list', 'submission_script']


def node_list(host_name):
    """
    Attempt to find a node list for SGE clusters.

    Parameters
    ----------
    host_name

    Returns
    -------

    """
    sge_nodes = os.environ.get('PE_HOSTFILE', None)

    if sge_nodes is None:
        return

    sge_list = []
    with open(sge_nodes, 'r') as file:
        lines = file.readlines()

        for line in lines:
            line = line.strip().split(' ')

            if line[0] != host_name:
                sge_list.append(line[0])

    return sge_list


def submission_script(name, num_nodes, num_workers, num_threads, node_memory):
    """
    Generate a submission script for SGE clusters.

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

name={name}
num_nodes={num_nodes}
num_workers_per_node={num_workers}
num_threads_per_worker={num_threads}

#$ -P <project_id>
#$ -A <sub_project_id>

# only allow C nodes
#$ -ac allow=C

# wall clock time (format hours:minutes:seconds).
#$ -l h_rt=48:00:00

# amount of RAM per core (must be an integer)
# node_memory/(num_threads_per_worker*num_workers_per_node)
#$ -l mem={int(node_memory/(num_threads*num_workers))}G

# set the name of the job.
#$ -N {name}

# select the MPI parallel environment and number of cores.
# num_threads_per_worker*num_workers_per_node*(num_nodes+1)
#$ -pe mpi {num_threads*num_workers*(num_nodes+1)}

# set the working directory
#$ -cwd

# set output logs
#$ -o out.log
#$ -e err.log

# load any modules before activating the conda env
# for example:
# module load compilers/intel/2020/release

# activate conda environment
conda activate stride

# set number of threads per process
# use $(ppn) to use one worker per node and as many threads pr worker as cores in the node
export OMP_NUM_THREADS=$num_workers_per_node \\* $num_threads_per_worker

# set any environment variables
# for example:
# export DEVITO_COMPILER=icc

# run our job
ls -l
date
mrun -n $num_nodes -nw $num_workers_per_node -nth $num_threads_per_worker python forward.py &> $name-output.log
date

stat=$?
echo "Exit status: $stat" >> "$name-output.log"
"""
