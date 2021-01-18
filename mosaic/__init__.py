
from .core import tessera
from .runtime import Head, Monitor, Node, Worker
from .utils.subprocess import subprocess
from .utils import logger as mlogger
from .utils import gather
from .file_manipulation import yaml, h5


_runtime = None
_runtime_types = {
    'head': Head,
    'monitor': Monitor,
    'node': Node,
    'worker': Worker,
}


def init(runtime_type='head', runtime_indices=(),
         address=None, port=None,
         parent_id=None, parent_address=None, parent_port=None,
         monitor_address=None, monitor_port=None,
         num_workers=None, num_threads=None,
         mode='local', monitor_strategy='round-robin',
         log_level='info', wait=False):
    global _runtime

    mlogger.log_level = log_level

    runtime_config = {
        'runtime_indices': runtime_indices,
        'mode': mode,
        'monitor_strategy': monitor_strategy,
        'num_workers': num_workers,
        'num_threads': num_threads,
        'log_level': log_level,
    }

    if address is not None and port is not None:
        runtime_config['address'] = address
        runtime_config['port'] = port

    if parent_id is not None and parent_address is not None and parent_port is not None:
        runtime_config['parent_id'] = parent_id
        runtime_config['parent_address'] = parent_address
        runtime_config['parent_port'] = parent_port

    elif monitor_address is not None and monitor_port is not None:
        runtime_config['monitor_address'] = monitor_address
        runtime_config['monitor_port'] = monitor_port

    elif runtime_type != 'head':
        ValueError('Either parent address:port or the monitor address:port are needed to '
                   'init a %s' % runtime_type)

    # Create global runtime
    try:
        _runtime = _runtime_types[runtime_type](**runtime_config)
    except KeyError:
        raise KeyError('Endpoint type is not recognised, available types are head, '
                       'monitor, node and worker')

    loop = _runtime.get_event_loop()
    loop.run(_runtime.init, kwargs=runtime_config, wait=True)

    _runtime.wait(wait=wait)


def __getattr__(key):
    global _runtime

    try:
        return getattr(_runtime, key)

    except AttributeError:
        raise AttributeError('module mosaic has no attribute %s' % key)


def clear_runtime():
    global _runtime

    if _runtime is not None:
        del _runtime
        _runtime = None


def runtime():
    global _runtime

    return _runtime


def stop():
    global _runtime

    loop = _runtime.get_event_loop()

    return loop.run(_runtime.stop, args=(), kwargs={})


def run(main, *args, **kwargs):
    global _runtime

    init(*args, **kwargs)

    try:
        loop = _runtime.get_event_loop()
        loop.run(main, args=(_runtime,), kwargs={}, wait=True)

    finally:
        stop()
