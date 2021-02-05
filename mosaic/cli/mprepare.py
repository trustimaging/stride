
import click

from . import clusters


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
def go(cluster_type, name, **kwargs):
    num_nodes = kwargs.get('nnodes', None)
    num_workers = kwargs.get('nworkers', None)
    num_threads = kwargs.get('nthreads', None)
    node_memory = kwargs.get('memory', None)

    valid_clusters = ['sge']

    if cluster_type not in valid_clusters:
        raise ValueError('Cluster type %s is not valid (%s).' % (cluster_type, ', '.join(valid_clusters)))

    submission_script = getattr(clusters, cluster_type).submission_script

    run_file = submission_script(name, num_nodes, num_workers, num_threads, node_memory)

    with open('', 'w') as file:
        file.write(run_file)


if __name__ == '__main__':
    go(auto_envvar_prefix='MOSAIC')
