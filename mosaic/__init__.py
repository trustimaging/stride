
__version__ = '1.0'

import asyncio

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
         log_level='info', node_list=None,
         asyncio_loop=None, wait=False,
         **kwargs):
    """
    Starts the global mosaic runtime.

    Parameters
    ----------
    runtime_type : str, optional
        Type of runtime to instantiate, defaults to ``head``.
    runtime_indices : tuple, optional
        Indices associated with the runtime, defaults to None.
    address : str, optional
        Address to use for the runtime, defaults to None. If None, the comms will
        try to guess the address.
    port : int, optional
        Port to use for the runtime, defaults to None. If None, the comms will
        test ports until one becomes available.
    parent_id : str, optional
        UID of the parent runtime, if any.
    parent_address : str, optional
        Address of the parent runtime, if any.
    parent_port : int, optional
        Port of the parent runtime, if any.
    monitor_address : str, optional
        Address of the monitor to connect to.
    monitor_port : int, optional
        Port of the monitor to connect to.
    num_workers : int, optional
        Number of workers to instantiate in each node, defaults to 1.
    num_threads : int, optional
        Number of threads to assign to each worker, defaults to the number of
        available cores over ``num_workers``.
    mode : str, optional
        Mode of the runtime, defaults to ``local``.
    monitor_strategy : str, optional
        Strategy used by the monitor to allocate tessera, defaults to round robin.
    log_level : str, optional
        Log level, defaults to ``info``.
    node_list : list, optional
        List of available node addresses to connect to.
    asyncio_loop: object, optional
        Async loop to use in our mosaic event loop, defaults to new loop.
    wait : bool, optional
        Whether or not to return control to calling frame, defaults to False.
    kwargs : optional
        Extra keyword arguments.

    Returns
    -------

    """
    global _runtime

    if _runtime is not None:
        return _runtime

    mlogger.log_level = log_level

    runtime_config = {
        'runtime_indices': runtime_indices,
        'mode': mode,
        'monitor_strategy': monitor_strategy,
        'num_workers': num_workers,
        'num_threads': num_threads,
        'log_level': log_level,
        'node_list': node_list,
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

    loop = _runtime.get_event_loop(asyncio_loop=asyncio_loop)
    result = loop.run(_runtime.init, kwargs=runtime_config, wait=True)

    if wait is True:
        try:
            loop.run_forever()

        finally:
            loop.stop()

    return result


def __getattr__(key):
    global _runtime

    try:
        return getattr(_runtime, key)

    except AttributeError:
        raise AttributeError('module mosaic has no attribute %s' % key)


def clear_runtime():
    """
    Clear the global runtime.

    Returns
    -------

    """
    global _runtime

    if _runtime is not None:
        mlogger.clear_logger()

        del _runtime
        _runtime = None


def runtime():
    """
    Access the global runtime.

    Returns
    -------

    """
    global _runtime

    return _runtime


def stop():
    """
    Stop the global runtime.

    Returns
    -------

    """
    global _runtime

    loop = _runtime.get_event_loop()

    try:
        loop.run(_runtime.stop, args=(), kwargs={}, wait=True)

    finally:
        loop.stop()
        clear_runtime()


def run(main, *args, **kwargs):
    """
    Initialise the runtime and then run the ``main`` in it.

    Parameters
    ----------
    main : callable
        Entry point for mosaic.
    args : tuple, optional
        Arguments to `mosaic.init`.
    kwargs : optional
        Keyword arguments to `mosaic.init`.

    Returns
    -------

    """
    global _runtime

    init(*args, **kwargs)

    loop = _runtime.get_event_loop()

    try:
        loop.run(main, args=(_runtime,), kwargs={}, wait=True)

    finally:
        stop()


async def interactive(switch, *args, **kwargs):
    """
    Initialise the runtime interactively.

    Parameters
    ----------
    switch : str
        Whether to switch interactive mode ``on`` or ``off``.
    args : tuple, optional
        Arguments to `mosaic.init`.
    kwargs : optional
        Keyword arguments to `mosaic.init`.

    Returns
    -------

    """
    global _runtime

    if switch == 'on':
        if _runtime is not None:
            return

        fut = init(*args, **kwargs,
                   mode='interactive',
                   asyncio_loop=asyncio.get_event_loop())

        await fut

    else:
        if _runtime is None:
            return

        loop = _runtime.get_event_loop()

        try:
            await loop.run(_runtime.stop, args=(), kwargs={})

        finally:
            clear_runtime()
