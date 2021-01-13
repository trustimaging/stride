
import numpy as np
from collections import OrderedDict

from .base import GriddedSaved, ProblemBase
from .. import plotting


__all__ = ['Geometry']


class TransducerLocation(GriddedSaved):

    def __init__(self, id, transducer=None, coordinates=None, orientation=None,
                 name=None, *args, **kwargs):
        name = name or 'transducer_instance_%05d' % id
        super().__init__(name, *args, **kwargs)

        if id < 0:
            raise ValueError('The transducer location needs a positive ID')

        self.id = id
        self.transducer = transducer

        if coordinates is not None:
            coordinates = transducer.coordinates + coordinates
        self.coordinates = coordinates

        self.orientation = orientation

    def check_bounds(self):
        pass

    def sub_problem(self, shot, sub_problem):
        sub_location = TransducerLocation(self.id,
                                          name=self.name, grid=self.grid)

        transducer = sub_problem.transducers.get(self.transducer.id)
        sub_location.transducer = transducer

        sub_location.coordinates = self.coordinates
        sub_location.orientation = self.orientation

        return sub_location

    def __get_desc__(self):
        description = {
            'id': self.id,
            'transducer_id': self.transducer.id,
            'coordinates': self.coordinates,
        }

        if self.orientation is not None:
            description['orientation'] = self.orientation

        return description

    def __set_desc__(self, description, transducers=None):
        self.id = description.id
        self.transducer = transducers.get(description.transducer_id)
        self.coordinates = description.coordinates.load()

        if 'orientation' in description:
            self.orientation = description.orientation.load()


class Geometry(ProblemBase):
    """
    The Geometry represents a series of transducers that exist within the confines of the Medium.

    Transducers can be added through ``Geometry.add(transducer, coordinates, orientation)`` and can be accessed through
    ``Geometry.get(transducer_id)``.

    The Geometry also provides utilities for loading and dumping these transducers and for plotting them.

    Parameters
    ----------
    problem : Problem
        Problem to which the Medium belongs.
    transducers : dict-like, optional
        Transducers with which to initialise the Geometry, defaults to empty.

    """

    def __init__(self, name='geometry', problem=None, **kwargs):
        super().__init__(name, problem, **kwargs)

        if problem is not None:
            transducers = problem.transducers
        else:
            transducers = kwargs.pop('transducers', None)

        if transducers is None:
            raise ValueError('A Geometry has be defined with respect to a set of Transducers')

        self._locations = OrderedDict()
        self._transducers = transducers

    def add(self, id, transducer, coordinates, orientation=None):
        """
        Add a new transducer to the Geometry.

        Parameters
        ----------
        id : int
            ID of the instantiation of the transducer in the geometry.
        transducer : Transducer
            Transducer to be added to the Geometry.
        coordinates : array
            Coordinates of the transducer in the grid.
        orientation : array, optional
            Orientation vector of the transducer.

        Returns
        -------

        """
        if id in self._locations.keys():
            raise ValueError('Transducer with ID "%d" already exists in the Geometry' % id)

        instance = TransducerLocation(id, transducer, coordinates, orientation,
                                      grid=self.grid)
        self._locations[id] = instance

    def add_location(self, item):
        """
        Add a new transducer to the Geometry.

        Parameters
        ----------
        item : TransducerLocation
            Transducer instance to be added to the Geometry.

        Returns
        -------

        """
        if item.id in self._locations.keys():
            raise ValueError('Transducer with ID "%d" already exists in the Geometry' % item.id)

        self._locations[item.id] = item

    def get(self, id):
        """
        Get a transducer from the Geometry with a known id.

        Parameters
        ----------
        id : int
            Identifier of the transducer.

        Returns
        -------
        Transducer
            Found Transducer.

        """
        if isinstance(id, (np.int32, np.int64)):
            id = int(id)

        if not isinstance(id, int) or id < 0:
            raise ValueError('Transducer IDs have to be positive integer numbers')

        return self._locations[id]

    def get_slice(self, start=None, end=None, step=None):
        """
        Get a slice of the indices of the transducer using ``slice(start, stop, step)``.

        Parameters
        ----------
        start : int, optional
            Start of the slice, defaults to the first id.
        end : int, optional
            End of the slice, defaults to the last id.
        step : int, optional
            Steps in between transducers, defaults to 1.

        Returns
        -------
        list
            Found transducers in the slice.

        """
        section = OrderedDict()
        if start is None:
            _range = range(end)
        elif step is None:
            _range = range(start, end)
        else:
            _range = range(start, end, step)

        for index in _range:
            section[list(self._locations.keys())[index]] = list(self._locations.values())[index]

        return section

    @property
    def transducers(self):
        return self._transducers

    @property
    def num_locations(self):
        """
        Get number of locations in the Geometry.

        Returns
        -------
        int
            Number of locations.

        """
        return len(self._locations.keys())

    @property
    def locations(self):
        """
        Get all locations in the Geometry.

        Returns
        -------
        list
            Transducer locations in the Geometry.

        """
        return list(self._locations.values())

    @property
    def location_ids(self):
        """
        Get all location IDs in the Geometry.

        Returns
        -------
        list
            IDs of the transducers.

        """
        return list(self._locations.keys())

    @property
    def coordinates(self):
        """
        Get the coordinates of all transducers packed in an array format.

        Returns
        -------
        2-dimensional array
            Array containing the coordinates of all transducers, with shape (n_transducers, n_dimensions).

        """
        coordinates = np.zeros((self.num_locations, self.space.dim), dtype=np.float32)
        index = 0
        for transducer in self._locations.values():
            coordinates[index, :] = transducer.coordinates
            index += 1

        return coordinates

    def plot(self, **kwargs):
        title = kwargs.pop('title', self.name)
        return plotting.plot_points(self.coordinates, title=title, **kwargs)

    def sub_problem(self, shot, sub_problem):
        sub_geometry = Geometry(name=self.name,
                                transducers=sub_problem.transducers,
                                problem=sub_problem, grid=self.grid)

        source_ids = shot.source_ids
        receiver_ids = shot.receiver_ids

        location_ids = list(set(source_ids) | set(receiver_ids))
        for location_id in location_ids:
            location = self.get(location_id)
            location = location.sub_problem(shot, sub_problem)

            sub_geometry.add_location(location)

        return sub_geometry

    def __get_desc__(self):
        description = {
            'num_locations': self.num_locations,
            'locations': [],
        }

        for location_id, location in self._locations.items():
            description['locations'].append(location.__get_desc__())

        return description

    def __set_desc__(self, description):
        for location_desc in description.locations:
            instance = TransducerLocation(location_desc.id)
            instance.__set_desc__(location_desc, self._transducers)

            self.add_location(instance)
