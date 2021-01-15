
import numpy as np
from collections import OrderedDict

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
        super().__init__(name, problem, **kwargs)

        self._fields = OrderedDict()

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

    def damping(self, damping_coefficient=None, mask=False):
        """
        Create a damping field based on the dimensions of the grid.

        Parameters
        ----------
        damping_coefficient : float
            Value of the maximum damping of the field.
        mask : bool
            Create the damping layer as a mask (interior filled with ones) or not (interior filled with zeros).

        Returns
        -------
        ndarray
            Tensor containing the damping field.

        """
        # Create a damping field that corresponds to the given field, only scalar for now
        if mask:
            damp = np.ones(self.space.extended_shape, dtype=np.float32)
        else:
            damp = np.zeros(self.space.extended_shape, dtype=np.float32)

        if damping_coefficient is None:
            damping_coefficient = 0.5 * np.log(1.0 / 0.001) / np.max(self.space.extra)

        spacing = self.space.spacing
        absorbing = self.space.absorbing

        for dimension in range(self.space.dim):
            for index in range(absorbing[dimension]):
                # Damping coefficient
                pos = np.abs((absorbing[dimension] - index + 1) / float(absorbing[dimension]))
                val = damping_coefficient * (pos - np.sin(2 * np.pi * pos) / (2 * np.pi))

                # : slices
                all_ind = [slice(0, d) for d in damp.shape]
                # Left slice for dampening for dimension
                all_ind[dimension] = slice(index, index + 1)
                damp[tuple(all_ind)] += val / spacing[dimension]
                # right slice for dampening for dimension
                all_ind[dimension] = slice(damp.shape[dimension] - index, damp.shape[dimension] - index + 1)
                damp[tuple(all_ind)] += val / spacing[dimension]

        return damp

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
        kwargs : dict
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
