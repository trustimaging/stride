
import os
import numpy as np
from collections import OrderedDict

from .base import Gridded
from . import Medium, Transducers, Geometry, Acquisitions, Shot
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
        kwargs['project_name'] = kwargs.get('project_name', self.name)
        kwargs['path'] = kwargs.get('path', self.input_folder)

        self.medium.load(*args, **kwargs)
        self.transducers.load(*args, **kwargs)
        self.geometry.load(*args, **kwargs)
        self.acquisitions.load(*args, **kwargs)

        grid_properties = ['space', 'time', 'slow_time']
        problem_properties = ['medium', 'transducers', 'geometry', 'acquisitions']

        for problem_property in problem_properties:
            problem_property = getattr(self, problem_property)

            for grid_property in grid_properties:
                if getattr(self, grid_property) is None and getattr(problem_property, grid_property) is not None:
                    setattr(self._grid, grid_property, getattr(problem_property, grid_property))

        for problem_property in problem_properties:
            problem_property = getattr(self, problem_property)

            for grid_property in grid_properties:
                if getattr(problem_property, grid_property) is None:
                    setattr(problem_property._grid, grid_property, getattr(self, grid_property))

    def time_resample(self, new_step, new_num=None, **kwargs):
        dt_in = self.time.step  # Extract current parameters
        start = self.time.start
        stop = self.time.stop
        num = self.time.num

        new_start = 0.  # Calculate new parameters

        interp_num = int((num)*(dt_in/new_step))
        interp_stop = new_start + new_step*(interp_num - 1)

        if new_num is not None:  # Do we need to pad the array or not?
            new_stop = new_start + new_step*(new_num - 1)
        else:
            new_num = interp_num
            new_stop = interp_stop

        print('Old num: {:f}'.format(num))
        print('Calc num: {:f}'.format(interp_num))
        print('New num: {:f}'.format(new_num))

        print('Old stop: {:f}'.format(stop))
        print('New stop: {:f}'.format(new_stop))

        self.grid.time.__init__(start=new_start, step=new_step, num=new_num)  # Update time
        try:
            del self.grid.time.__dict__['grid']
            del self.grid.time.__dict__['extended_grid']
        except:
            print('no grid')

        shots_out = OrderedDict()
        for shot_id, shot in self.acquisitions._shots.items():
            shot_out = Shot(shot_id,
                        sources=shot.sources, receivers=shot.receivers,
                        geometry=self.geometry, problem=self)  # create new shot

            wavelet_out = shot.wavelets._resample(factor=dt_in/new_step, new_num=new_num)  # resample wavelet

            shot_out.wavelets.data[:] = wavelet_out

            observed_out = shot.observed._resample(factor=dt_in/new_step, new_num=new_num)  # resample observed

            shot_out.observed.data[:] = observed_out

            # shot.delays._resample(factor=0.9)  # Not sure what delays are, and whether they need resampling?

            shots_out[shot_id] = shot_out

        # import IPython.terminal.debugger as ipdb; ipdb.set_trace()
        self.acquisitions._shots.clear()  # Clear old shots
        for shot in shots_out:  # Replace with new shots
            self.acquisitions.add(shots_out[shot])

    def dump(self, *args, **kwargs):
        """
        Dump all elements in the Problem.

        See :class:`~mosaic.file_manipulation.h5.HDF5` for more information on the parameters of this method.

        Returns
        -------

        """
        kwargs['project_name'] = kwargs.get('project_name', self.name)
        kwargs['path'] = kwargs.get('path', self.output_folder)

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
            acquisitions_axes = self.acquisitions.plot(plot=False)
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
