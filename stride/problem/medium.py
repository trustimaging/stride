
from .base import ProblemBase


__all__ = ['Medium']


class Medium(ProblemBase):
    """
    A Medium contains the fields that define a physical medium,
    such as density or longitudinal speed of sound.

    A Medium defines these properties by keeping track of a series of named fields, that can be added to
    it through ``Medium.add``.

    A field with a name ``field_name`` can be accessed directly through ``medium.field_name`` or
    ``medium['field_name']``.

    The Medium also provides utilities for loading and dumping these fields and for plotting them.

    Parameters
    ----------
    name : str
        Alternative name to give to the medium.
    problem : Problem
        Problem to which the Medium belongs.
    grid : Grid or any of Space or Time
        Grid on which the Medium is defined

    """

    def __init__(self, name='medium', problem=None, **kwargs):
        super().__init__(name=name, problem=problem, **kwargs)

        self._fields = dict()

    def _get(self, item):
        if item in super().__getattribute__('_fields').keys():
            return self._fields[item]

        return super().__getattribute__(item)

    def __getattr__(self, item):
        return self._get(item)

    def __getitem__(self, item):
        return self._get(item)

    @property
    def fields(self):
        """
        Access fields dictionary.

        """
        return self._fields

    def items(self):
        """
        Access all fields as (name, field) pairs.

        Returns
        -------
        Fields
            Iterable of (name, field) pairs.

        """
        return self._fields.items()

    def add(self, field):
        """
        Add a named field to the Medium.

        Parameters
        ----------
        field : Field object
            Field to add to the Medium.

        Returns
        -------

        """
        self._fields[field.name] = field

    def load(self, *args, **kwargs):
        """
        Load all fields in the Medium.

        See :class:`~mosaic.file_manipulation.h5.HDF5` for more information on the parameters of this method.

        Returns
        -------

        """
        for field_name, field in self._fields.items():
            field.load(*args, **kwargs)

    def dump(self, *args, **kwargs):
        """
        Dump all fields in the Medium.

        See :class:`~mosaic.file_manipulation.h5.HDF5` for more information on the parameters of this method.

        Returns
        -------

        """
        for field_name, field in self._fields.items():
            field.dump(*args, **kwargs)

    def plot(self, **kwargs):
        """
        Plot all fields in the Medium.

        Parameters
        ----------
        kwargs
            Arguments for plotting the fields.

        Returns
        -------
        axes
            Axes on which the plotting is done.

        """
        axes = []
        for field in self._fields.values():
            axis = field.plot(**kwargs)
            axes.append(axis)

        return axes
