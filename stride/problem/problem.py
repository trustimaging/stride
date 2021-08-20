
import os
import numpy as np

from .base import Gridded
from . import Medium, Transducers, Geometry, Acquisitions
from .. import plotting


__all__ = ['Problem', 'SubProblem']


class Problem(Gridded):
    """
    The Problem is the object that fully defines the setting in which Stride works.

    The problem defines a medium with a set of fields (such as Vp or density), some
    transducers (such as a series of scalar point transducers), a geometry where those
    transducers are located in space, and the acquisitions that happen given that geometry.

    The problem also defines a problem type, which determines the physics of interest, such
    as the second-order isotropic acoustic wave equation. And a numerical implementation
    of those physics, such as through the finite-difference library Devito.

    Parameters
    ----------
    name : str
        Name of the problem.
    grid : Grid or any of Space or Time
        Grid on which the Problem is defined
    input_folder : str, optional
        Default folder from which files should be read, defaults to current working directory.
    output_folder : str, optional
        Default folder to which files should be written, defaults to current working directory.
    medium : Medium, optional
        Predefined Medium of the problem.
    transducers : Transducers, optional
        Predefined Transducers of the problem.
    geometry : Geometry, optional
        Predefined Geometry of the problem.
    acquisitions : Acquisitions, optional
        Predefined Acquisitions of the problem.

    """

    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = name
        self.input_folder = kwargs.pop('input_folder', os.getcwd())
        self.output_folder = kwargs.pop('output_folder', os.getcwd())

        self.problem_config = {}

        medium = kwargs.pop('medium', None)
        if medium is None:
            medium = Medium(problem=self)

        self.medium = medium

        transducers = kwargs.pop('transducers', None)
        if transducers is None:
            transducers = Transducers(problem=self)

        self.transducers = transducers

        geometry = kwargs.pop('geometry', None)
        if geometry is None:
            geometry = Geometry(transducers=transducers, problem=self)

        self.geometry = geometry

        acquisitions = kwargs.pop('acquisitions', None)
        if acquisitions is None:
            acquisitions = Acquisitions(geometry=geometry, problem=self)

        self.acquisitions = acquisitions

    def load(self, *args, **kwargs):
        """
        Load all elements in the Problem.

        See :class:`~mosaic.file_manipulation.h5.HDF5` for more information on the parameters of this method.

        Returns
        -------

        """
        self.medium.load(*args, **kwargs)
        self.transducers.load(*args, **kwargs)
        self.geometry.load(*args, **kwargs)
        self.acquisitions.load(*args, **kwargs)

    def dump(self, *args, **kwargs):
        """
        Dump all elements in the Problem.

        See :class:`~mosaic.file_manipulation.h5.HDF5` for more information on the parameters of this method.

        Returns
        -------

        """
        self.medium.dump(*args, **kwargs)
        self.transducers.dump(*args, **kwargs)
        self.geometry.dump(*args, **kwargs)
        self.acquisitions.dump(*args, **kwargs)

    def plot(self, **kwargs):
        """
        Plot all elements in the Problem.

        Parameters
        ----------
        kwargs
            Arguments for plotting the fields.

        Returns
        -------

        """
        kwargs['plot'] = False
        plot_medium = kwargs.pop('medium', True)
        plot_geometry = kwargs.pop('geometry', True)
        plot_acquisitions = kwargs.pop('acquisitions', True)

        # Medium
        medium_axes = [None]

        if plot_medium:
            medium_axes = self.medium.plot(**kwargs)

        medium_axes = medium_axes if len(medium_axes) else [None]

        # Geometry
        geometry_axes = medium_axes

        if plot_geometry:
            geometry_axes = []

            for axis in medium_axes:
                geometry_axes.append(self.geometry.plot(axis=axis, title=None, **kwargs))

        plotting.show(geometry_axes)

        # Acquisitions
        if plot_acquisitions:
            acquisitions_axes = self.acquisitions.plot()
            plotting.show(acquisitions_axes)

    def sub_problem(self, shot_id):
        """
        Create a subset object for a certain shot.

        A SubProblem contains everything that is needed to fully determine how to run a particular shot.
        This method takes care of selecting creating a SubProblem instance and populating it
        appropriately.

        Parameters
        ----------
        shot_id : int
            ID of the shot for which this sub-problem will be generated.

        Returns
        -------
        SubProblem
            Newly created SubProblem instance.

        """

        if isinstance(shot_id, (np.int32, np.int64)):
            shot_id = int(shot_id)

        sub_problem = SubProblem(self.name,
                                 input_folder=self.input_folder,
                                 output_folder=self.output_folder,
                                 grid=self.grid)

        shot = self.acquisitions.get(shot_id)

        # Set up transducers
        sub_problem.transducers = self.transducers.sub_problem(shot, sub_problem)

        # Set up geometry
        sub_problem.geometry = self.geometry.sub_problem(shot, sub_problem)

        # Set up acquisitions
        shot = shot.sub_problem(shot, sub_problem)
        sub_problem.shot = shot
        sub_problem.shot_id = shot.id
        sub_problem.acquisitions = self.acquisitions.sub_problem(shot, sub_problem)

        return sub_problem


class SubProblem(Gridded):
    """
    The SubProblem is the object that fully defines how a specific Shot is to be run. The SubProblem
    resembles the Problem from which ir originates, but takes from it only those parts that
    are relevant for this particular Shot.

    The SubProblem defines a medium with a set of fields (such as Vp or density), some
    transducers (such as a series of scalar point transducers), a geometry where those
    transducers are located in space, and the acquisitions that happen given that geometry.

    The SubProblem also defines a problem type, which determines the physics of interest, such
    as the second-order isotropic acoustic wave equation. And a numerical implementation
    of those physics, such as through the finite-difference library Devito.

    Parameters
    ----------
    name : str
        Name of the problem.
    grid : Grid or any of Space or Time
        Grid on which the Problem is defined
    input_folder : str, optional
        Default folder from which files should be read, defaults to current working directory.
    output_folder : str, optional
        Default folder to which files should be written, defaults to current working directory.
    transducers : Transducers, optional
        Predefined Transducers of the problem.
    geometry : Geometry, optional
        Predefined Geometry of the problem.
    acquisitions : Acquisitions, optional
        Predefined Acquisitions of the problem.

    """

    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.shot = None
        self.shot_id = -1

        self.name = name
        self.input_folder = kwargs.pop('input_folder', os.getcwd())
        self.output_folder = kwargs.pop('output_folder', os.getcwd())

        self.problem_config = {}
        self.transducers = kwargs.pop('transducers', None)
        self.geometry = kwargs.pop('geometry', None)
        self.acquisitions = kwargs.pop('acquisitions', None)
