
import sys
import importlib


__all__ = ['mimport']


def mimport(name, fromlist=()):
    """
    ``mosaic_import`` encapsulates importlib, allowing for dynamically importing a module by name, given a series of
    filesystem locations in which to look.

    Unlike usual Python imports, the cache of imported modules is purposefully ignored and modules will always
    be re-imported even if they have been imported previously.

    Parameters
    ----------
    name : str
        Name of the module to be imported
    fromlist : tuple, optional
        List of paths in which to look for the module

    Returns
    -------
    Python module
        Imported Python module

    """
    _saved_path = sys.path
    sys.path = list(fromlist)
    imported_module = importlib.reload(importlib.__import__(name, fromlist=fromlist))
    sys.path = _saved_path

    return imported_module
