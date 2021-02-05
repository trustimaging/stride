
import os
import click
import subprocess

import mosaic
from .. import init, stop
from ..utils.logger import _stdout, _stderr


@click.command()
@click.argument('type', required=True, nargs=1)
@click.argument('name', required=True, nargs=1)
@click.option('--nnodes', '-n', type=int, required=True, show_default=True,
              help='number of nodes to be generated')
@click.option('--nworkers', '-nw', type=int, required=True, show_default=True,
              help='number of workers to be spawned')
@click.option('--nthreads', '-nth', type=int, required=True, show_default=True,
              help='number of threads per worker')
@click.option('--memory', '-m', type=int, required=True, show_default=True,
              help='available memory per node (in GBytes)')
@click.version_option()
def go(type, name, **kwargs):
    num_nodes = kwargs.get('nnodes', None)
    num_workers = kwargs.get('nworkers', None)
    num_threads = kwargs.get('nthreads', None)
    node_memory = kwargs.get('memory', None)

    if type not in ['sge', 'slurm', 'pbs', 'torque']:
        raise ValueError('Cluster type %s is not valid.' % type)

    run_file = f"""
#!/bin/bash -l

name={name}
num_nodes={num_nodes}
num_workers_per_node={num_workers}
num_threads_per_worker={num_threads}

#$ -P <project_id>
#$ -A <sub_project_id> 

# only allow C nodes
#$ -ac allow=C

# wall clock time (format hours:minutes:seconds).
#$ -l h_rt=48:00:0

# amount of RAM per core (must be an integer)
#$ -l mem={int(node_memory/(num_threads*num_workers))}G

# set the name of the job.
#$ -N $name

# select the MPI parallel environment and number of cores.
# there's 40 cores per node
#$ -pe mpi {num_threads*num_workers*(num_nodes+1)}

# set the working directory
#$ -cwd

# activate conda environment
conda activate stride

# set number of threads per process
# use $(ppn) to use one process per node and as many threads as cores in the node
export OMP_NUM_THREADS={num_workers*num_threads}

# run our job
ls -l
mrun -n $num_nodes -nw $num_workers_per_node -nth $num_threads_per_worker python forward.py &> {name}-output.log
    """

    with open('', 'w') as file:
        file.write(run_file)


if __name__ == '__main__':
    go(auto_envvar_prefix='MOSAIC')
