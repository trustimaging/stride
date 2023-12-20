
import sys
import psutil
import asyncio
import traceback
import threading

import mosaic

__all__ = ['sizeof', 'remote_sizeof', 'set_main_thread', 'memory_limit',
           'memory_used', 'cpu_count', 'gpu_count', 'MultiError']


def sizeof(obj, seen=None):
    """
    Recursively finds size of objects.

    Parameters
    ----------
    obj : object
        Object to check size.
    seen

    Returns
    -------
    float
        Size in bytes.

    """
    ignore = (asyncio.Future,) + mosaic.types.remote_types
    if isinstance(obj, ignore):
        return 0
    try:
        if hasattr(obj, 'nbytes') and isinstance(obj.nbytes, int):
            size = obj.nbytes
        else:
            size = sys.getsizeof(obj)
    except Exception:
        size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)

    try:
        if isinstance(obj, dict):
            size += sum([sizeof(v, seen) for v in obj.values()])
            size += sum([sizeof(k, seen) for k in obj.keys()])
        elif hasattr(obj, '__dict__'):
            size += sizeof(obj.__dict__, seen)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            try:
                size += sum([sizeof(i, seen) for i in obj])
            except TypeError:
                pass
    except RuntimeError:
        pass

    return size


async def remote_sizeof(obj, seen=None, pending=False):
    """
    Recursively finds size of remote objects.

    Parameters
    ----------
    obj : object
        Object to check size.
    pending : bool
        Only count pending objects.
    seen

    Returns
    -------
    float
        Size in bytes.

    """
    if isinstance(obj, asyncio.Future):
        return 0
    if isinstance(obj, mosaic.types.awaitable_types):
        size = await obj.size(pending=pending)
    else:
        if pending:
            size = 0
        else:
            try:
                if hasattr(obj, 'nbytes') and isinstance(obj.nbytes, int):
                    size = obj.nbytes
                else:
                    size = sys.getsizeof(obj)
            except Exception:
                size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)

    try:
        if isinstance(obj, dict):
            _size = await asyncio.gather(*(remote_sizeof(v, seen, pending) for v in obj.values()))
            size += sum(_size)
            _size = await asyncio.gather(*(remote_sizeof(k, seen, pending) for k in obj.keys()))
            size += sum(_size)
        elif hasattr(obj, '__dict__'):
            size += await remote_sizeof(obj.__dict__, seen, pending)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            _size = await asyncio.gather(*(remote_sizeof(i, seen, pending) for i in obj))
            size += sum(_size)
    except RuntimeError:
        pass

    return size


def set_main_thread():
    """
    Set current thread as main thread.

    Returns
    -------

    """
    threading.current_thread().name = 'MainThread'
    threading.current_thread().__class__ = threading._MainThread


def memory_used(pid=None):
    """
    Get the memory currently being used by the system.

    Parameters
    ----------
    pid : int, optional
        PID for which to get memory.

    Returns
    -------
    float
        Memory used.

    """
    if pid is None:
        mem_total = memory_limit()
        mem = max(mem_total - psutil.virtual_memory().available,
                  psutil.virtual_memory().used)
    else:
        proc = psutil.Process(pid)
        mem = proc.memory_info().rss

    return mem


def memory_limit():
    """
    Get the memory limit (in bytes) for this system.

    Takes the minimum value from the following locations:
    - Total system host memory
    - Cgroups limit (if set)
    - RSS rlimit (if set)

    Returns
    -------
    float
        Memory limit.

    """
    limit = psutil.virtual_memory().total

    # Check cgroups if available
    if sys.platform == "linux":
        try:
            with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
                cgroups_limit = int(f.read())

            if cgroups_limit > 0:
                limit = min(limit, cgroups_limit)

        except Exception:
            pass

    # Check rlimit if available
    try:
        import resource

        hard_limit = resource.getrlimit(resource.RLIMIT_RSS)[1]
        if hard_limit > 0:
            limit = min(limit, hard_limit)

    except (ImportError, OSError):
        pass

    return limit


def cpu_count():
    """
    Get the number of available cores in the node.

    Returns
    -------
        int
            Number of CPUs.

    """
    num_logical_cpus = psutil.cpu_count(logical=True)
    num_cpus = psutil.cpu_count(logical=False) or num_logical_cpus

    return num_cpus


def gpu_count():
    """
    Get the number of available GPUs in the node.

    Returns
    -------
        int
            Number of GPUs.

    """
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
    except (ImportError, ValueError):
        return None

    return len(gpus)


class MultiError(Exception):

    def __init__(self, exc):
        self.errors = []
        self.add(exc)
        super().__init__(exc)

    def add(self, exc):
        if isinstance(exc, MultiError):
            self.errors += exc.errors
        else:
            try:
                raise exc
            except Exception:
                et, ev, tb = sys.exc_info()
                tb = traceback.format_tb(tb)
                tb = ''.join(tb)
                self.errors.append((et, ev, tb))

    def __str__(self):
        error_str = str(self.errors[-1][1]) + '\n\nError stack:\n\n'

        for error in self.errors:
            et, ev, tb = error
            error_str += 'Traceback (most recent call last):\n'
            error_str += f'{tb}{et.__name__}: {str(ev)}\n\n'

        return error_str
