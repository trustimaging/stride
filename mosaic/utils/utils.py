
import sys
import psutil
import threading
import numpy as np

__all__ = ['sizeof', 'set_main_thread', 'memory_limit']


def sizeof(obj, seen=None):
    """Recursively finds size of objects"""
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
    if isinstance(obj, dict):
        size += sum([sizeof(v, seen) for v in obj.values()])
        size += sum([sizeof(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += sizeof(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([sizeof(i, seen) for i in obj])

    return size


def set_main_thread():
    threading.current_thread().name = 'MainThread'
    threading.current_thread().__class__ = threading._MainThread


def memory_limit():
    """Get the memory limit (in bytes) for this system.
    Takes the minimum value from the following locations:
    - Total system host memory
    - Cgroups limit (if set)
    - RSS rlimit (if set)
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
