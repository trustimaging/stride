
import functools
import numpy as np
from collections import OrderedDict

try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    ENABLED_2D_PLOTTING = True

except ModuleNotFoundError:
    ENABLED_2D_PLOTTING = False

from .data import Traces
from .base import ProblemBase


__all__ = ['Shot', 'Acquisitions']


class Shot(ProblemBase):
    """
    A Shot is an even in which one or more transducers act as sources with a given wavelet and one or more
    transducers act as receivers and record some observed data.

    Therefore a shot object maintains data about the ids of the transducer locations that will act as sources,
    the ids of the transducer locations that will act as receivers, as well as the wavelets that will be fired and
    the observed data that is recorded.

    Parameters
    ----------
    id : int
        Identifier assigned to this shot.
    name : str
        Optional name for the shot.
    problem : Problem
        Problem to which the Shot belongs.
    geometry : Geometry
        Geometry referenced by the source/receiver transducer locations of the shot.
    sources : list
        Sources with which to initialise the shot, defaults to empty.
    receivers : list
        Receivers with which to initialise the shot, defaults to empty.
    grid : Grid or any of Space or Time
        Grid on which the Acquisitions is defined

    """

    def __init__(self, id, name=None, problem=None, **kwargs):
        name = name or 'shot_%05d' % id
        super().__init__(name, problem, **kwargs)

        if id < 0:
            raise ValueError('The shot needs a positive ID')

        self.id = id

        if problem is not None:
            geometry = problem.geometry
        else:
            geometry = kwargs.pop('geometry', None)

        if geometry is None:
            raise ValueError('A Shot has be defined with respect to a Geometry')

        self._geometry = geometry
        self._acquisitions = None

        self._sources = OrderedDict()
        self._receivers = OrderedDict()
        self.wavelets = None
        self.observed = None

        sources = kwargs.pop('sources', None)
        receivers = kwargs.pop('receivers', None)

        if sources is not None and receivers is not None:
            for source in sources:
                self._sources[source.id] = source

            for receiver in receivers:
                self._receivers[receiver.id] = receiver

            self.wavelets = Traces('wavelets', transducer_ids=self.source_ids, grid=self.grid)
            self.observed = Traces('observed', transducer_ids=self.receiver_ids, grid=self.grid)

    @property
    def geometry(self):
        return self._geometry

    @property
    def source_ids(self):
        """
        Get ids of sources in this Shot in a list.

        """
        return list(self._sources.keys())

    @property
    def receiver_ids(self):
        """
        Get ids of receivers in this Shot in a list.

        """
        return list(self._receivers.keys())

    @property
    def sources(self):
        """
        Get sources in this Shot as a list.

        """
        return list(self._sources.values())

    @property
    def receivers(self):
        """
        Get receivers in this Shot as a list.

        """
        return list(self._receivers.values())

    @property
    def num_sources(self):
        """
        Get number of sources in the Shot.

        """
        return len(self.source_ids)

    @property
    def num_receivers(self):
        """
        Get number of receivers in the Shot.

        """
        return len(self.receiver_ids)

    @property
    def source_coordinates(self):
        """
        Get the coordinates of all sources packed in an array format.

        Coordinates are defined as a 2 or 3-dimensional array with shape (n_sources, n_dimensions).

        """
        coordinates = np.zeros((self.num_sources, self.space.dim), dtype=np.float32)
        source_index = 0
        for source in self.sources:
            coordinates[source_index, :] = source.coordinates
            source_index += 1

        return coordinates

    @property
    def receiver_coordinates(self):
        """
        Get the coordinates of all receivers packed in an array format.

        Coordinates are defined as a 2 or 3-dimensional array with shape (n_receivers, n_dimensions).

        """
        coordinates = np.zeros((self.num_receivers, self.space.dim), dtype=np.float32)
        receiver_index = 0
        for receiver in self.receivers:
            coordinates[receiver_index, :] = receiver.coordinates
            receiver_index += 1

        return coordinates

    def sub_problem(self, shot, sub_problem):
        """
        Create a subset object for a certain shot.

        A SubProblem contains everything that is needed to fully determine how to run a particular shot.
        This method takes care of generating a new Shot object that is linked to this new SubProblem.

        Parameters
        ----------
        shot : Shot
            Shot for which the SubProblem is being generated.
        sub_problem : SubProblem
            Container for the sub-problem being generated.

        Returns
        -------
        Shot
            Newly created Shot instance.

        """
        shot = Shot(self.id,
                    name=self.name, problem=sub_problem,
                    grid=self.grid, geometry=sub_problem.geometry)

        for source_id in self.source_ids:
            location = sub_problem.geometry.get(source_id)
            shot._sources[location.id] = location

        for receiver_id in self.receiver_ids:
            location = sub_problem.geometry.get(receiver_id)
            shot._receivers[location.id] = location

        if self.wavelets is not None:
            shot.wavelets = self.wavelets

        if self.observed is not None:
            shot.observed = self.observed

        return shot

    def plot(self, **kwargs):
        """
        Plot wavelets and observed for this shot if they are allocated.

        Parameters
        ----------
        kwargs : dict
            Arguments for plotting.

        Returns
        -------
        axes
            Axes on which the plotting is done.

        """
        axes = []

        if self.wavelets is not None and self.wavelets.allocated:
            axes.append(self.wavelets.plot(**kwargs))

        if self.observed is not None and self.observed.allocated:
            axes.append(self.observed.plot(**kwargs))

        return axes

    def plot_wavelets(self, **kwargs):
        """
        Plot wavelets for this shot if they are allocated.

        Parameters
        ----------
        kwargs : dict
            Arguments for plotting.

        Returns
        -------
        axes
            Axes on which the plotting is done.

        """
        if self.wavelets is not None and self.wavelets.allocated:
            return self.wavelets.plot(**kwargs)

    def plot_observed(self, **kwargs):
        """
        Plot observed for this shot if they are allocated.

        Parameters
        ----------
        kwargs : dict
            Arguments for plotting.

        Returns
        -------
        axes
            Axes on which the plotting is done.

        """
        if self.observed is not None and self.observed.allocated:
            return self.observed.plot(**kwargs)

    def append_observed(self, *args, **kwargs):
        """
        Append the shot to the corresponding Acquisitions file.

        See :class:`~mosaic.file_manipulation.h5.HDF5` for more information on the parameters of this method.

        Returns
        -------

        """
        kwargs['parameter'] = 'acquisitions'
        kwargs['version'] = kwargs.get('version', 0)

        self._acquisitions.append(*args, **kwargs)

    def __get_desc__(self):
        description = {
            'id': self.id,
            'num_sources': self.num_sources,
            'num_receivers': self.num_receivers,
            'source_ids': self.source_ids,
            'receiver_ids': self.receiver_ids,
        }

        if self.wavelets is not None and self.wavelets.allocated:
            description['wavelets'] = self.wavelets.__get_desc__()

        if self.observed is not None and self.observed.allocated:
            description['observed'] = self.observed.__get_desc__()

        return description

    def __set_desc__(self, description):
        self.id = description.id

        for source_id in description.source_ids:
            self._sources[source_id] = self._geometry.get(source_id)

        for receiver_id in description.receiver_ids:
            self._receivers[receiver_id] = self._geometry.get(receiver_id)

        if 'wavelets' in description:
            self.wavelets = Traces('wavelets', transducer_ids=self.source_ids, grid=self.grid)
            self.wavelets.__set_desc__(description.wavelets)

        if 'observed' in description:
            self.observed = Traces('observed', transducer_ids=self.receiver_ids, grid=self.grid)
            self.observed.__set_desc__(description.observed)


class Acquisitions(ProblemBase):
    """
    Acquisitions establish a series of shots that will be or have been fired to generate data.

    A shot is an even in which one or more transducer locations act as sources with a given wavelet and one or more
    transducer locations act as receivers and record some observed data.

    Shots are identified through a numerical ID, which is >= 0.

    Shots can be added through ``Acquisitions.add(shot)`` and can be accessed through
    ``Acquisitions.get(shot_id)``.

    The Acquisitions also provides utilities for loading and dumping these shots and their data.

    Parameters
    ----------
    name : str
        Alternative name to give to the medium.
    problem : Problem
        Problem to which the Acquisitions belongs.
    geometry : Geometry
        Geometry referenced by the source/receiver transducer locations of the shot.
    grid : Grid or any of Space or Time
        Grid on which the Acquisitions is defined

    """

    def __init__(self, name='acquisitions', problem=None, **kwargs):
        super().__init__(name, problem, **kwargs)

        if problem is not None:
            geometry = problem.geometry
        else:
            geometry = kwargs.pop('geometry', None)

        if geometry is None:
            raise ValueError('An Acquisition has be defined with respect to a Geometry')

        self._geometry = geometry
        self._shots = OrderedDict()
        self._shot_selection = []

    @property
    def shots(self):
        """
        Get all shots in the Acquisitions as a list.

        """
        return list(self._shots.values())

    @property
    def shot_ids(self):
        """
        Get all IDs of shots in the Acquisitions as a list.

        """
        return list(self._shots.keys())

    @property
    def num_shots(self):
        """
        Get number of shots in the Acquisitions.

        """
        return len(self.shot_ids)

    @property
    def num_sources_per_shot(self):
        """
        Get maximum number of sources in any shot.

        """
        num_transducers = max(*[each.num_sources for each in self._shots.values()])
        return num_transducers

    @property
    def num_receivers_per_shot(self):
        """
        Get maximum number of receivers in any shot.

        """
        num_transducers = max(*[each.num_receivers for each in self._shots.values()])
        return num_transducers

    @property
    def remaining_shots(self):
        """
        Get dict of all shots that have no observed allocated.

        """
        shots = OrderedDict()
        for shot_id, shot in self._shots.items():
            if not shot.observed.allocated:
                shots[shot_id] = shot

        return shots

    @property
    def remaining_shot_ids(self):
        """
        Get list of all shot IDs that have no observed allocated.

        """
        shot_ids = []
        for shot_id, shot in self._shots.items():
            if not shot.observed.allocated:
                shot_ids.append(shot_id)

        return shot_ids

    def add(self, item):
        """
        Add a new shot to the Acquisitions.

        Parameters
        ----------
        item : Shot
            Shot to be added to the Acquisitions.

        Returns
        -------

        """
        if item.id in self._shots.keys():
            raise ValueError('Shot with ID "%d" already exists in the Acquisitions' % item.id)

        self._shots[item.id] = item
        item._acquisitions = self

    def get(self, id):
        """
        Get a shot from the Acquisitions with a known id.

        Parameters
        ----------
        id : int
            Identifier of the shot.

        Returns
        -------
        Shot
            Found Shot.

        """
        if isinstance(id, (np.int32, np.int64)):
            id = int(id)

        if not isinstance(id, int) or id < 0:
            raise ValueError('Shot IDs have to be positive integer numbers')

        return self._shots[id]

    def set(self, item):
        """
        Change an existing shot in the Acquisitions.

        Parameters
        ----------
        item : Shot
            Shot to be modified in the Acquisitions.

        Returns
        -------

        """
        if item.id not in self._shots.keys():
            raise ValueError('Shot with ID "%d" does not exist in the Acquisitions' % item.id)

        self._shots[item.id] = item

    def select_shot_ids(self, start=None, end=None, num=None, every=1, randomly=False):
        """
        Select a number of shots according to the rules given in the arguments to the method.

        For every call to this method a new group of shots will be selected according to
        those rules until all shots have been selected. At that point, the selection will
        start again.

        Parameters
        ----------
        start : int, optional
            Start of the slice, defaults to the first id.
        end : int, optional
            End of the slice, defaults to the last id.
        num : int, optional
            Number of shots to select every time the method is called.
        every : int, optional
            How many shots to skip in the selection, defaults to 1, which means taking all shots
            subsequently.
        randomly : bool, optional
            Whether to select the shots at random at in order, defaults to False.

        Returns
        -------
        list
            List with selected shots.

        """
        if not len(self._shot_selection):
            ids_slice = slice(start or 0, end)
            shot_ids = self.shot_ids
            shot_ids.sort()
            shot_ids = shot_ids[ids_slice]

            if randomly is True:
                self._shot_selection = np.random.permutation(shot_ids).tolist()

            else:
                num_groups = int(np.ceil(len(shot_ids) / num))

                self._shot_selection = []

                for group_index in range(num_groups):

                    group = []
                    num_remaining = len(shot_ids)

                    start_index = group_index if every > 1 else 0
                    for index in range(start_index, num_remaining, every):
                        group.append(shot_ids[index])

                        if len(group) == num or not len(shot_ids):
                            break

                    self._shot_selection += group
                    shot_ids = shot_ids if every > 1 else list(set(shot_ids)-set(group))

        next_slice = self._shot_selection[:num]
        self._shot_selection = self._shot_selection[num:]

        return next_slice

    def default(self):
        """
        Fill the container with the default configuration.

        In this case, that means that every location in the Geometry
        acts as a source once while every location acts as a receiver.

        This generates as many shots as there are locations available in the
        Geometry. Each Shot only has one source and as many receivers as locations
        are in the Geometry.

        Returns
        -------

        """
        for source in self._geometry.locations:
            receivers = self._geometry.locations

            self.add(Shot(source.id,
                          sources=[source], receivers=receivers,
                          geometry=self._geometry, problem=self.problem))

    def plot(self, **kwargs):
        """
        Plot wavelets and observed for for all shots if they are allocated.

        Parameters
        ----------
        kwargs : dict
            Arguments for plotting.

        Returns
        -------

        """
        self.plot_wavelets(**kwargs)
        self.plot_observed(**kwargs)

    def _plot(self, update):
        if not ENABLED_2D_PLOTTING:
            return None

        figure, axis = plt.subplots(1, 1)
        plt.subplots_adjust(bottom=0.25)
        axis.margins(x=0)

        ax_shot = plt.axes([0.15, 0.1, 0.7, 0.03])
        slider = Slider(ax_shot, 'shot ID',
                        self.shot_ids[0], self.shot_ids[-1],
                        valinit=self.shot_ids[0], valstep=1)

        update = functools.partial(update, figure, axis)
        update(self.shot_ids[0])

        slider.on_changed(update)
        axis.slider = slider

        return axis

    def plot_wavelets(self, **kwargs):
        """
        Plot wavelets for for all shots if they are allocated.

        Parameters
        ----------
        kwargs : dict
            Arguments for plotting.

        Returns
        -------

        """
        if not self.get(0).wavelets.allocated:
            return None

        def update(figure, axis, shot_id):
            axis.clear()

            self.get(int(shot_id)).plot_wavelets(axis=axis, **kwargs)
            axis.set_title(axis.get_title() + ' - shot %d' % shot_id)

            figure.canvas.draw_idle()

        return self._plot(update)

    def plot_observed(self, **kwargs):
        """
        Plot observed for for all shots if they are allocated.

        Parameters
        ----------
        kwargs : dict
            Arguments for plotting.

        Returns
        -------

        """
        if not self.get(0).observed.allocated:
            return None

        def update(figure, axis, shot_id):
            axis.clear()

            self.get(int(shot_id)).plot_observed(axis=axis, **kwargs)
            axis.set_title(axis.get_title() + ' - shot %d' % shot_id)

            figure.canvas.draw_idle()

        return self._plot(update)

    def sub_problem(self, shot, sub_problem):
        """
        Create a subset object for a certain shot.

        A SubProblem contains everything that is needed to fully determine how to run a particular shot.
        This method takes care of selecting the portions of the Acquisitions that are needed
        for a given shot.

        Parameters
        ----------
        shot : Shot
            Shot for which the SubProblem is being generated.
        sub_problem : SubProblem
            Container for the sub-problem being generated.

        Returns
        -------
        Acquisitions
            Newly created Acquisitions instance.

        """
        sub_acquisitions = Acquisitions(name=self.name,
                                        geometry=sub_problem.geometry,
                                        problem=sub_problem, grid=self.grid)
        sub_acquisitions.add(shot)

        return sub_acquisitions

    def __get_desc__(self):
        description = {
            'num_shots': self.num_shots,
            'shots': [],
        }

        for shot in self.shots:
            description['shots'].append(shot.__get_desc__())

        return description

    def __set_desc__(self, description):
        for shot_desc in description.shots:
            if shot_desc.id not in self._shots:
                shot = Shot(shot_desc.id,
                            geometry=self._geometry,
                            problem=self.problem, grid=self.grid)
                self.add(shot)

            shot = self.get(shot_desc.id)
            shot.__set_desc__(shot_desc)
