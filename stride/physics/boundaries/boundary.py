
import numpy as np
from abc import ABC, abstractmethod


__all__ = ['Boundary']


class Boundary(ABC):
    """
    Base class for Boundaries that can be applied to the different problem types.

    Parameters
    ----------
    grid : DevitoGrid

    """

    def __init__(self, grid):
        self._grid = grid

    @abstractmethod
    def apply(self, *args, **kwargs):
        """
        Generate the necessary pieces to make the boundary work.

        Returns
        -------
        term
            Any extra terms to add to the equations.
        list
            Equations to execute before the state equation.
        list
            Equations to execute after the state equation.

        """
        pass

    def clear(self):
        """
        Perform any clearing operations if needed.

        Returns
        -------

        """
        pass

    def deallocate(self):
        """
        Perform any deallocation operations if needed.

        Returns
        -------

        """
        pass

    def damping(self, dimensions=None, damping_coefficient=None, mask=False,
                damping_type='sine', velocity=1.0, power_degree=2, reflection_coefficient=1e-3,
                assign=False, **kwargs):
        """
        Create a damping field based on the dimensions of the grid.

        Parameters
        ----------
        dimensions : tuple of ints, optional
            Whether or not to fill only certain dimensions, defaults to ``None``, all dimensions.
        damping_coefficient : float, optional
            Value of the maximum damping of the field.
        mask : bool, optional
            Create the damping layer as a mask (interior filled with ones) or not (interior filled with zeros).
        damping_type : str, optional
            Expression to be used for the shape of the damping function, defaults to ``sine``.
        velocity : ndarray or float, optional
            Velocity in the boundary region, defaults to 1.0.
        power_degree : int, optional
            Degree of the power to use for ``power`` damping, defaults to 2.
        reflection_coefficient : float, optional
            Theoretical reflection coefficient of the layer, defaults to 1e-3.
        assign : bool, optional
            Whether to assign or sum the value at each location, defaults to ``False``.

        Returns
        -------
        ndarray
            Tensor containing the damping field.

        """
        space = self._grid.space

        dimensions = tuple(range(space.dim)) if dimensions is None else dimensions

        # Create a damping field that corresponds to the given field, only scalar for now
        shape = np.array(space.extended_shape).take(dimensions)
        if mask:
            damp = np.ones(shape, dtype=np.float32)

            if damping_coefficient is not None:
                damp *= damping_coefficient
        else:
            damp = np.zeros(shape, dtype=np.float32)

        spacing = space.spacing
        absorbing = space.absorbing

        for dim_i, dimension in zip(range(len(dimensions)), dimensions):

            dimension_coefficient = damping_coefficient
            if dimension_coefficient is None:
                dimension_coefficient = (power_degree + 1) / 2 * np.log(1.0 / reflection_coefficient)

                dimension_coefficient = dimension_coefficient / (absorbing[dimension]*spacing[dimension]) \
                    if absorbing[dimension] > 15 else 0.67 / spacing[dimension]

            for index in range(absorbing[dimension]):
                # Damping coefficient
                pos = np.abs((absorbing[dimension] - index - 1) / float(absorbing[dimension] - 1))

                if damping_type == 'sine':
                    pos = pos - np.sin(2 * np.pi * pos) / (2 * np.pi)
                    if mask:
                        pos = - pos

                    val = dimension_coefficient * pos

                elif damping_type == 'cosine':
                    pos = np.cos(np.pi / 2 * (1 - pos))
                    if mask:
                        pos = - pos

                    val = pos

                elif damping_type == 'power':
                    pos = pos**power_degree
                    if mask:
                        pos = - pos

                    val = dimension_coefficient * pos

                else:
                    raise ValueError('Allowed dumping type are (`sine`, `power`)')

                # : slices
                all_ind = [slice(index, s-index) for s in shape]

                # Left slice for dampening for dimension
                all_ind[dim_i] = slice(index, index + 1)
                if assign:
                    damp[tuple(all_ind)] = val
                else:
                    damp[tuple(all_ind)] += val

                # : slices
                all_ind = [slice(index, s-index) for s in shape]

                # right slice for dampening for dimension
                all_ind[dim_i] = slice(damp.shape[dim_i] - index - 1, damp.shape[dim_i] - index)
                if assign:
                    damp[tuple(all_ind)] = val
                else:
                    damp[tuple(all_ind)] += val

        if damping_coefficient is None:
            if damp.shape == velocity.shape:
                damp *= velocity

            else:
                damp *= np.max(velocity)

        return damp
