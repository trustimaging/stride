
from collections import OrderedDict

from mosaic.utils import camel_case

from .base import ProblemBase
from ..problem_definition import transducer_types


__all__ = ['Transducers']


class Transducers(ProblemBase):

    def __init__(self, name='transducers', problem=None, **kwargs):
        super().__init__(name, problem, **kwargs)

        self._transducers = OrderedDict()

    def add(self, item):
        """
        Add a new transducer to the Transducers.

        Parameters
        ----------
        item : Transducer
            Transducer to be added to the Transducers.

        Returns
        -------

        """
        if item.id in self._transducers.keys():
            raise ValueError('Transducer with ID "%d" already exists in the Transducers' % item.id)

        self._transducers[item.id] = item

    def get(self, id):
        """
        Get a transducer from the Transducers with a known id.

        Parameters
        ----------
        id : int
            Identifier of the transducer.

        Returns
        -------
        Transducer
            Found Transducer.

        """
        if not isinstance(id, int) or id < 0:
            raise ValueError('Transducer IDs have to be positive integer numbers')

        return self._transducers[id]

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
            section[list(self._transducers.keys())[index]] = list(self._transducers.values())[index]

        return section

    def items(self):
        return self._transducers.items()

    @property
    def num_transducers(self):
        """
        Get number of transducers in the Transducers.

        Returns
        -------
        int
            Number of transducers.

        """
        return len(self._transducers.keys())

    @property
    def transducers(self):
        """
        Get all transducers in the Transducers.

        Returns
        -------
        list
            List of the transducers.

        """
        return list(self._transducers.values())

    @property
    def transducer_ids(self):
        """
        Get all transducer IDs in the Transducers.

        Returns
        -------
        list
            IDs of the transducers.

        """
        return list(self._transducers.keys())

    def default(self):
        transducer = transducer_types.PointTransducer(0, grid=self.grid)
        self.add(transducer)

    def sub_problem(self, shot, sub_problem):
        sub_transducers = Transducers(name=self.name,
                                      problem=sub_problem, grid=self.grid)

        source_ids = shot.source_ids
        receiver_ids = shot.receiver_ids

        location_ids = list(set(source_ids) | set(receiver_ids))
        geometry = self.problem.geometry
        for location_id in location_ids:
            location = geometry.get(location_id)

            if location.transducer.id in sub_transducers.transducer_ids:
                continue

            transducer = location.transducer.sub_problem(shot, sub_problem)
            sub_transducers.add(transducer)

        return sub_transducers

    def __get_desc__(self):
        description = {
            'num_transducers': self.num_transducers,
            'transducers': [],
        }

        for transducer_id, transducer in self._transducers.items():
            description['transducers'].append(transducer.__get_desc__())

        return description

    def __set_desc__(self, description):
        for transducer_desc in description.transducers:
            transducer_type = getattr(transducer_types, camel_case(transducer_desc.type))
            transducer = transducer_type(transducer_desc.id, grid=self.grid)

            transducer.__set_desc__(transducer_desc)

            self.add(transducer)
