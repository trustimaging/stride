
from ..utils.mimport import mimport


__all__ = ['ImportedFunction']


class ImportedFunction:
    """
    A function that is imported dynamically and contains information on how to serialise
    and deserialise it.

    Parameters
    ----------
    name : str
        Name of the function.
    module : str
        Name of the module to which it belongs.
    path : tuple
        List of paths to look for the function.

    """

    def __init__(self, name, module, path):
        self._name = name
        self._module = module
        self._path = path

        self._import()

    def __call__(self, *args, **kwargs):
        self._fun(*args, **kwargs)

    def _import(self):
        value = mimport(self._module, fromlist=self._path)
        self._fun = getattr(value, self._name)

    _serialisation_attrs = ['_name', '_module', '_path']

    def _serialisation_helper(self):
        state = {}

        for attr in self._serialisation_attrs:
            state[attr] = getattr(self, attr)

        return state

    @classmethod
    def _deserialisation_helper(cls, state):
        instance = cls.__new__(cls)

        for attr, value in state.items():
            setattr(instance, attr, value)

        instance._import()

        return instance

    def __reduce__(self):
        state = self._serialisation_helper()
        return self._deserialisation_helper, (state,)
