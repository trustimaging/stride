
import click
import subprocess as cmd_subprocess

from . import clusters
from .. import init, stop, runtime
from ..comms import get_hostname
from ..utils import subprocess
from ..utils.logger import _stdout, _stderr


@click.command()
@click.argument('cmd', required=False, nargs=-1)
# runtime type
@click.option('--head', 'runtime_type', flag_value='head', show_default=True,
              help='start the head runtime')
@click.option('--monitor', 'runtime_type', flag_value='monitor', show_default=True,
              help='start the monitor runtime')
@click.option('--node', 'runtime_type', flag_value='node', show_default=True,
              help='start the node runtime')
@click.option('--indices', '-i', type=str, required=False, show_default=True,
              help='runtime indices if any')
@click.option('--daemon/--inproc', type=bool, default=False, required=True, show_default=True,
              help='whether to run as a daemon')
# network config
@click.option('--nnodes', '-n', type=int, required=False, default=1, show_default=True,
              help='number of nodes to be generated')
@click.option('--nworkers', '-nw', type=int, required=False, default=1, show_default=True,
              help='number of workers to be spawned')
@click.option('--nthreads', '-nth', type=int, required=False, show_default=True,
              help='number of threads per worker')
# comms config
@click.option('--address', type=str, required=False, show_default=True,
              help='IP address to use for the runtime')
@click.option('--port', type=int, required=False, show_default=True,
              help='port to use for the runtime')
@click.option('--monitor-address', type=str, required=False, show_default=True,
              help='IP address of the monitor')
@click.option('--monitor-port', type=int, required=False, show_default=True,
              help='port of the monitor')
@click.option('--pubsub-port', type=int, required=False, show_default=True,
              help='publishing port of the monitor')
# cluster options
@click.option('--local/--cluster', '-l/-c', default=False, required=True, show_default=True,
              help='whether to run mosaic locally or in a cluster system')
@click.option('--reuse-head/--free-head', '-rh/-fh', default=False, required=True, show_default=True,
              help='whether to create workers in the head node')
# log level
@click.option('--perf', 'log_level', flag_value='perf', default='perf', show_default=True,
              help='set log level to PERF')
@click.option('--info', 'log_level', flag_value='info', show_default=True,
              help='set log level to INFO')
@click.option('--debug', 'log_level', flag_value='debug', show_default=True,
              help='set log level to DEBUG')
@click.option('--error', 'log_level', flag_value='error', show_default=True,
              help='set log level to ERROR')
# profiling
@click.option('--profile/--perf', default=False, required=False, show_default=True,
              help='whether to profile the mosaic run')
@click.version_option()
def go(cmd=None, **kwargs):
    runtime_type = kwargs.get('runtime_type', None)
    runtime_indices = kwargs.get('indices', None)
    local = kwargs.get('local', False)
    reuse_head = kwargs.get('reuse_head', False)

    if runtime_indices is not None:
        runtime_indices = tuple([int(i) for i in runtime_indices.split(':')])

    if not local:
        num_nodes = kwargs.get('nnodes', 1)
    else:
        num_nodes = 1
    num_workers = kwargs.get('nworkers', 1)
    num_threads = kwargs.get('nthreads', None)
    log_level = kwargs.get('log_level', 'perf')
    profile = kwargs.get('profile', False)

    # If not in local mode, find the node list
    node_list = None
    if not local and runtime_type in [None, 'monitor']:
        # sun grid engine   - PE_HOSTFILE
        # slurm             - SLURM_JOB_NODELIST
        # pbs/torque        - PBS_NODEFILE

        host_name = get_hostname()

        sge_nodes = clusters.sge.node_list(host_name, reuse_head)
        pbs_nodes = clusters.pbs.node_list(host_name, reuse_head)
        slurm_nodes = clusters.slurm.node_list(host_name, reuse_head)

        if sge_nodes is not None:
            node_list = sge_nodes

        elif pbs_nodes is not None:
            node_list = pbs_nodes

        elif slurm_nodes is not None:
            node_list = slurm_nodes

        else:
            local = True

        if node_list is not None and num_nodes != len(node_list):
            node_list = node_list[:num_nodes]

    runtime_config = {
        'runtime_indices': runtime_indices,
        'address': kwargs.get('address', None),
        'port': kwargs.get('port', None),
        'monitor_address': kwargs.get('monitor_address', None),
        'monitor_port': kwargs.get('monitor_port', None),
        'pubsub_port': kwargs.get('pubsub_port', None),
        'num_nodes': num_nodes,
        'num_workers': num_workers,
        'num_threads': num_threads,
        'mode': 'local' if local is True else 'cluster',
        'reuse_head': reuse_head,
        'log_level': log_level,
        'profile': profile,
        'node_list': node_list,
    }

    # Initialise the runtime
    if runtime_type is not None:
        if kwargs.get('daemon', False):
            def start_runtime(*args, **extra_kwargs):
                extra_kwargs.update(runtime_config)

                init(runtime_type, **extra_kwargs, wait=True)

            runtime_subprocess = subprocess(start_runtime)(runtime_type, daemon=True)
            runtime_subprocess.start_process()

        else:
            init(runtime_type, **runtime_config, wait=True)

        return

    else:
        init('monitor', **runtime_config, wait=False)
        _runtime = runtime()

    # Get the initialised runtime
    loop = _runtime.get_event_loop()
    _runtime.init_file(runtime_config)

    def run_head():
        process = cmd_subprocess.run(' '.join(cmd), shell=True,
                                     stdout=_stdout,
                                     stderr=_stderr)

        if process.returncode == 0:
            _runtime.logger.info('Process ended with code: %d' % process.returncode)

        else:
            raise RuntimeError('Process ended with code: %d' % process.returncode)

    async def main():
        await loop.run_in_executor(run_head)

    try:
        loop.run(main)

    finally:
        stop()


if __name__ == '__main__':
    go(auto_envvar_prefix='MOSAIC')
