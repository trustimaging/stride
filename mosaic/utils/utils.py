
import sys
import psutil
import traceback
import threading
import numpy as np

__all__ = ['sizeof', 'set_main_thread', 'memory_limit', 'cpu_count', 'MultiError']


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
    if isinstance(obj, np.ndarray):
        size = obj.nbytes
    else:
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
            size += sum([sizeof(i, seen) for i in obj])
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
