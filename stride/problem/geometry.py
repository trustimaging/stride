import numpy as np
from collections import OrderedDict

import mosaic.types
from .base import GriddedSaved, ProblemBase
from .. import plotting
from ..utils import geometries


__all__ = ['Geometry']


class TransducerLocation(GriddedSaved):
    """
    This determines the spatial location of a specific transducer device within the
    geometry.

    The location is determined by a numerical ID (>= 0), a transducer and the
    coordinates of the location within the space grid. In some cases, the orientation
    of the transducer might also be needed.

    Parameters
    ----------
    id : int
        Numerical ID of the location (>=0).
    name : str
        Optional name for the transducer location.
    transducer : Transducer
        Transducer device to which this location refers.
    coordinates : ndarray
        Coordinates of the transducer in the space grid.
    orientation : ndarray, optional
        Orientation of the transducer with respect to its location.

    """

    def __init__(self, id, transducer=None, coordinates=None, orientation=None,
                 name=None, **kwargs):
        name = name or 'transducer_instance_%05d' % id
        super().__init__(name=name, **kwargs)

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
        """
        Create a subset object for a certain shot.

        A SubProblem contains everything that is needed to fully determine how to run a particular shot.
        This method takes care of creating a TransducerLocation object that links to that
        new SubProblem.

        Parameters
        ----------
        shot : Shot
            Shot for which the SubProblem is being generated.
        sub_problem : SubProblem
            Container for the sub-problem being generated.

        Returns
        -------
        TransducerLocation
            Newly created TransducerLocation instance.

        """
        return self

    def __get_desc__(self, **kwargs):
        description = {
            'id': self.id,
            'transducer_id': self.transducer.id,
            'coordinates': self.coordinates,
        }

        if self.orientation is not None:
            description['orientation'] = self.orientation

        return description

    def __set_desc__(self, description, transducers=None, **kwargs):
        self.id = description.id
        self.transducer = transducers.get(description.transducer_id)
        if hasattr(description.coordinates, 'load'):
            self.coordinates = description.coordinates.load()
        else:
            self.coordinates = description.coordinates

        if 'orientation' in description:
            if hasattr(description.orientation, 'load'):
                self.orientation = description.orientation.load()
            else:
                self.orientation = description.orientation


class Geometry(ProblemBase):
    """
    The Geometry represents a series of transducer locations that exist within the confines of the grid.

    Transducer locations are identified through a numerical ID, which is >= 0.

    Transducer locations can be added at a certain location through ``Geometry.add(id, transducer, coordinates, [orientation])``
    and can be accessed through ``Geometry.get(location_id)``.

    The Geometry also provides utilities for loading and dumping these transducers and for plotting them.

    Parameters
    ----------
    name : str
        Alternative name to give to the medium.
    problem : Problem
        Problem to which the Geometry belongs.
    transducers : Transducers
        Transducers object to which the Geometry refers.
    grid : Grid or any of Space or Time
        Grid on which the Geometry is defined

    """

    def __init__(self, name='geometry', problem=None, **kwargs):
        super().__init__(name=name, problem=problem, **kwargs)

        if problem is not None:
            transducers = problem.transducers
        else:
            transducers = kwargs.pop('transducers', None)

        self._locations = OrderedDict()
        self._transducers = transducers

    def add(self, id, transducer, coordinates, orientation=None):
        """
        Add a new transducer location to the Geometry.

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
            raise ValueError('Transducer location with ID "%d" already exists in the Geometry' % id)

        instance = TransducerLocation(id, transducer, coordinates, orientation,
                                      grid=self.grid)
        self._locations[id] = instance

    def add_location(self, item):
        """
        Add an existing location to the Geometry.

        Parameters
        ----------
        item : TransducerLocation
            Transducer location instance to be added to the Geometry.

        Returns
        -------

        """
        if item.id in self._locations.keys():
            raise ValueError('Transducer location with ID "%d" already exists in the Geometry' % item.id)

        self._locations[item.id] = item

    def get(self, id):
        """
        Get a transducer location from the Geometry with a known id.

        Parameters
        ----------
        id : int
            Identifier of the transducer.

        Returns
        -------
        TransducerLocation
            Found TransducerLocation.

        """
        if isinstance(id, (np.int32, np.int64)):
            id = int(id)

        if not isinstance(id, int) or id < 0:
            raise ValueError('Transducer IDs have to be positive integer numbers')

        return self._locations[id]

    def get_slice(self, start=None, end=None, step=None):
        """
        Get a slice of the indices of the locations using ``slice(start, stop, step)``.

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
            Found transducer locations in the slice.

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

    def default(self, geometry_type, *args, **kwargs):
        """
        Fill the container with the default configuration.

        In this case, that means using one of the default geometry functions in ``stride.utils.geometries``
        and using the same transducer for all of them.

        Parameters
        ----------
        geometry_type : str
            Type of geometry to use.

        Returns
        -------

        """

        if geometry_type == 'elliptical':
            default_radius = ((self.space.limit[0] - 15.e-3) / 2,
                              (self.space.limit[1] - 13.e-3) / 2)
            default_centre = (self.space.limit[0] / 2,
                              self.space.limit[1] / 2)

            kwargs['radius'] = kwargs.get('radius', default_radius)
            kwargs['centre'] = kwargs.get('centre', default_centre)

        elif geometry_type == 'ellipsoidal':
            default_radius = ((self.space.limit[0] - 15.e-3) / 2,
                              (self.space.limit[1] - 15.e-3) / 2,
                              (self.space.limit[2] - 15.e-3) / 2)
            default_centre = (self.space.limit[0] / 2,
                              self.space.limit[1] / 2,
                              self.space.limit[2] / 2)

            if len(args) < 2:
                kwargs['radius'] = kwargs.get('radius', default_radius)
            if len(args) < 3:
                kwargs['centre'] = kwargs.get('centre', default_centre)
            kwargs['threshold'] = kwargs.get('threshold', 0.3)

        geometry_fun = getattr(geometries, geometry_type)
        coordinates = geometry_fun(*args, **kwargs)

        for index in range(coordinates.shape[0]):
            _coordinates = coordinates[index, :]
            if len(_coordinates) != self.space.dim:
                _coordinates = np.pad(_coordinates, ((0, 1),))
                _coordinates[-1] = self.space.limit[2] / 2

            self.add(index, self._transducers.get(0), _coordinates)

    @property
    def transducers(self):
        return self._transducers

    @property
    def num_locations(self):
        """
        Get number of locations in the Geometry.

        """
        return len(self._locations.keys())

    @property
    def locations(self):
        """
        Get all locations in the Geometry as a list.

        """
        return list(self._locations.values())

    @property
    def location_ids(self):
        """
        Get all location IDs in the Geometry as a list.

        """
        return list(self._locations.keys())

    @property
    def coordinates(self):
        """
        Get the coordinates of all locations packed in an array format.

        Coordinates are defined as a 2 or 3-dimensional array with shape (n_transducers, n_dimensions).

        """
        coordinates = np.zeros((self.num_locations, self.space.dim), dtype=np.float32)
        index = 0
        for location in self._locations.values():
            coordinates[index, :] = location.coordinates
            index += 1

        return coordinates

    def plot(self, **kwargs):
        """
        Plot the locations of the transducers as scattered points.

        Parameters
        ----------
        kwargs
            Arguments for plotting.

        Returns
        -------
        axes
            Axes on which the plotting is done.

        """
        title = kwargs.pop('title', self.name)
        plot = kwargs.pop('plot', True)

        coordinates = self.coordinates
        if self.space.dim > 2:
            coordinates = coordinates / np.array(self.space.spacing)

        axis = plotting.plot_points(coordinates, title=title, **kwargs)

        if plot is True:
            plotting.show(axis)

        return axis

    def sub_problem(self, shot, sub_problem):
        """
        Create a subset object for a certain shot.

        A SubProblem contains everything that is needed to fully determine how to run a particular shot.
        This method takes care of selecting the portions of the Geometry that are needed
        for a given shot.

        Parameters
        ----------
        shot : Shot
            Shot for which the SubProblem is being generated.
        sub_problem : SubProblem
            Container for the sub-problem being generated.

        Returns
        -------
        Geometry
            Newly created Geometry instance.

        """
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

    def __get_desc__(self, **kwargs):
        legacy = kwargs.pop('legacy', False)

        if legacy:
            description = {
                'num_locations': self.num_locations,
                'locations': [],
            }

            for location_id, location in self._locations.items():
                description['locations'].append(location.__get_desc__())

        else:
            description = {
                'num_locations': self.num_locations,
                'locations': {},
            }

            coordinates = None
            orientation = None

            index = 0
            for location_id, location in self._locations.items():
                location_desc = location.__get_desc__()
                if 'coordinates' in location_desc:
                    if coordinates is None:
                        coordinates = np.zeros((self.num_locations, self.space.dim), dtype=np.float32)
                    coordinates[index, :] = location_desc['coordinates']
                    location_desc['coordinates'] = index
                if 'orientation' in location_desc:
                    if orientation is None:
                        orientation = np.zeros((self.num_locations, self.space.dim), dtype=np.float32)
                    orientation[index, :] = location_desc['orientation']
                    location_desc['orientation'] = index

                description['locations'][str(location_id)] = location_desc
                index += 1

            if coordinates is not None:
                description['coordinates'] = coordinates

            if orientation is not None:
                description['orientation'] = orientation

        return description

    def __set_desc__(self, description, **kwargs):
        locations = description.locations
        if isinstance(locations, mosaic.types.Struct):
            locations = locations.values()

        if 'coordinates' in description:
            for location_desc in locations:
                idx = location_desc.coordinates
                location_desc.coordinates = description.coordinates[idx, :]

        if 'orientation' in description:
            for location_desc in locations:
                idx = location_desc.orientation
                location_desc.orientation = description.orientation[idx, :]

        for location_desc in locations:
            if location_desc.id not in self.location_ids:
                instance = TransducerLocation(location_desc.id)
                self.add_location(instance)

            instance = self.get(location_desc.id)
            instance.__set_desc__(location_desc, self._transducers, **kwargs)
