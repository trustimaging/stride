
import os
import click
import subprocess

import mosaic
from .. import init, stop
from ..utils.logger import _stdout, _stderr


@click.command()
@click.argument('cmd', required=True, nargs=-1)
@click.option('--nworkers', '-n', type=int, required=False, show_default=True,
              help='number of workers to be spawned')
@click.option('--nthreads', '-nth', type=int, required=False, show_default=True,
              help='number of threads per worker')
@click.version_option()
def go(cmd, **kwargs):
    runtime_config = {
        'mode': 'local',
        'log_level': 'info',
        'num_workers': kwargs.get('nworkers', None),
        'num_threads': kwargs.get('nthreads', None),
    }

    # Initialise monitor and, if needed nodes and workers
    _runtime = init('monitor', **runtime_config, wait=False)

    # Get the initialised runtime and store its
    # ID, address and port in a tmp file for the
    # head to use
    runtime = mosaic.runtime()
    runtime_id = runtime.uid
    runtime_address = runtime.address
    runtime_port = runtime.port

    path = os.path.join(os.getcwd(), 'mosaic-workspace')
    if not os.path.exists(path):
        os.makedirs(path)

    filename = os.path.join(path, 'monitor.key')
    with open(filename, 'w') as file:
        file.write('[ADDRESS]\n')
        file.write('UID=%s\n' % runtime_id)
        file.write('ADD=%s\n' % runtime_address)
        file.write('PRT=%s\n' % runtime_port)

    loop = _runtime.get_event_loop()

    def run_head():
        process = subprocess.run(cmd,
                                 stdout=_stdout,
                                 stderr=_stderr)

        runtime.logger.info('Process ended with code: %d' % process.returncode)

    async def main():
        await loop.run_in_executor(run_head, args=(), kwargs={})

    try:
        loop.run(main, args=(), kwargs={}, wait=True)

    finally:
        stop()

        try:
            os.remove(filename)
            os.rmdir(path)

        except Exception:
            pass


if __name__ == '__main__':
    go(auto_envvar_prefix='MOSAIC')
