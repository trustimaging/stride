
from mosaic import h5

from .domain import Space, MeshedSpace, Time, SlowTime, Grid


__all__ = ['Gridded', 'Saved', 'GriddedSaved', 'ProblemBase']


class Gridded:
    """
    Objects of this type are defined over a spatio-temporal grid.

    This grid can be provided either as a Grid object or as any of Space, Time or SlowTime
    objects that define a grid.

    Parameters
    ----------
    grid : Grid, optional
        Existing grid, if not provided one will be created.
    space : Space, optional
    time : Time, optional
    slow_time : SlowTime, optional

    """

    def __init__(self, **kwargs):
        grid = kwargs.pop('grid', None)
        space = kwargs.pop('space', None)
        time = kwargs.pop('time', None)
        slow_time = kwargs.pop('slow_time', None)

        if grid is None:
            grid = Grid(space, time, slow_time)

        else:
            grid = Grid(grid.space, grid.time, grid.slow_time)

        self._grid = grid

    @property
    def grid(self):
        """
        Access the grid.

        """
        return self._grid

    @property
    def space(self):
        """
        Access the space grid.

        """
        return self._grid.space

    @property
    def time(self):
        """
        Access the time grid.

        """
        return self._grid.time

    @property
    def slow_time(self):
        """
        Access the slow time grid.

        """
        return self._grid.slow_time

    def resample(self, grid=None, space=None, time=None, slow_time=None):
        raise NotImplementedError('Resampling has not been implemented in this class yet.' +
                                  ' Alternatively, try to access space_resample or time_resample via problem.')


class Saved:
    """
    Saved objects include helper functions to interact with the file system.

    Classes that inherit from Saved need to define ``__get_desc__`` and ``__set_desc__``
    to define how the object is described to be saved and how a loaded description is
    digested by the class respectively.

    ``__get_desc__`` expects a dict-like object with all the attributes that need to
    be stored to disk and is called when dumping the object.

    ``__set_desc__`` will take a dict-like object as a parameter, which it can then be
    used to set the state of the object, and is called when loading it.

    Parameters
    ----------
    name : str
        Name of the saved object.

    """

    def __init__(self, **kwargs):
        self.name = kwargs.pop('name', self.__class__.__name__.lower())

    def dump(self, *args, **kwargs):
        """
        Dump the object according to the ``__get_desc__`` description.

        See :class:`~mosaic.file_manipulation.h5.HDF5` for more information on the parameters of this method.

        Returns
        -------

        """
        description = self.__get_desc__(**kwargs)

        kwargs['parameter'] = kwargs.get('parameter', self.name)
        with h5.HDF5(*args, **kwargs, mode='w') as file:
            file.dump(description)

    def append(self, *args, **kwargs):
        """
        Append the object to a file according to the ``__get_desc__`` description.

        See :class:`~mosaic.file_manipulation.h5.HDF5` for more information on the parameters of this method.

        Returns
        -------

        """
        kwargs['parameter'] = kwargs.get('parameter', self.name)
        if not h5.file_exists(*args, **kwargs):
            self.dump(*args, **kwargs)
            return

        description = self.__get_desc__(**kwargs)

        with h5.HDF5(*args, **kwargs, mode='a') as file:
            file.append(description)

    def load(self, *args, **kwargs):
        """
        Load the object using ``__set_desc__`` to digest the description.

        See :class:`~mosaic.file_manipulation.h5.HDF5` for more information on the parameters of this method.

        Returns
        -------

        """
        kwargs['parameter'] = self.name
        with h5.HDF5(*args, **kwargs, mode='r') as file:
            description = file.load(filter=kwargs.pop('filter', None), only=kwargs.pop('only', None))

            kwargs['filename'] = kwargs.pop('filename', file.filename)
            self.__set_desc__(description, **kwargs)

    def rm(self, *args, **kwargs):
        """
        Remove file.

        See :class:`~mosaic.file_manipulation.h5.rm` for more information on the parameters of this method.

        Returns
        -------

        """
        kwargs['parameter'] = self.name
        h5.rm(*args, **kwargs, mode='r')

    def __get_desc__(self, **kwargs):
        return {}

    def __set_desc__(self, description, **kwargs):
        pass


class GriddedSaved(Saved, Gridded):
    """
    Objects of this type are include utils to dump and load the instance, taking into
    account that it is defined over a grid.

    """

    def __init__(self, **kwargs):
        Saved.__init__(self, **kwargs)
        Gridded.__init__(self, **kwargs)

    def dump(self, *args, **kwargs):
        """
        Dump the object according to the ``__get_desc__`` description.

        It will ensure that the grid of the instance is also dumped to disk.

        See :class:`~mosaic.file_manipulation.h5.HDF5` for more information on the parameters of this method.

        Returns
        -------

        """
        grid_description = self.grid_description()

        description = self.__get_desc__(**kwargs)
        grid_description.update(description)

        kwargs['parameter'] = kwargs.get('parameter', self.name)
        with h5.HDF5(*args, **kwargs, mode='w') as file:
            file.dump(grid_description)

    def append(self, *args, **kwargs):
        """
        Append the object to a file according to the ``__get_desc__`` description.

        It will ensure that the grid of the instance is also dumped to disk.

        See :class:`~mosaic.file_manipulation.h5.HDF5` for more information on the parameters of this method.

        Returns
        -------

        """
        kwargs['parameter'] = kwargs.get('parameter', self.name)
        if not h5.file_exists(*args, **kwargs):
            self.dump(*args, **kwargs)
            return

        grid_description = self.grid_description()

        description = self.__get_desc__(**kwargs)
        grid_description.update(description)

        with h5.HDF5(*args, **kwargs, mode='a') as file:
            file.append(grid_description)

    def load(self, *args, **kwargs):
        """
        Load the object using ``__set_desc__`` to digest the description.

        It will use the grid loaded from file to determine the grid of the instance.

        See :class:`~mosaic.file_manipulation.h5.HDF5` for more information on the parameters of this method.

        Returns
        -------

        """
        kwargs['parameter'] = self.name
        with h5.HDF5(*args, **kwargs, mode='r') as file:
            description = file.load(filter=kwargs.pop('filter', None), only=kwargs.pop('only', None))

            if 'space' in description and self._grid.space is None:
                # NOTE the discriminator has to be the keys of description.space, not those of
                # description: StructuredData.__get_desc__ also writes a top-level 'shape', so
                # testing the outer description would send every field down the structured branch
                space_description = description.space

                if 'shape' in space_description:
                    space = Space(shape=space_description.shape,
                                  spacing=space_description.spacing,
                                  extra=space_description.extra,
                                  absorbing=space_description.absorbing)

                elif 'nodes' in space_description:
                    # the cell type is stored as a string, so HDF5 hands it back as bytes, and
                    # the degree comes back as a numpy integer, which is not an int as far as
                    # isinstance is concerned. Both have to be normalised before construction
                    cell_type = self._materialise(space_description.get('cell_type', None))

                    if isinstance(cell_type, bytes):
                        cell_type = cell_type.decode()

                    geometry_degree = self._materialise(
                        space_description.get('geometry_degree', 1))

                    facet_tag_nodes = self._materialise(
                        space_description.get('facet_tag_nodes', None))

                    facet_tags = None
                    if facet_tag_nodes is not None:
                        facet_tags = {
                            'nodes': facet_tag_nodes,
                            'values': self._materialise(space_description.facet_tag_values),
                        }

                    space = MeshedSpace(
                        nodes=self._materialise(space_description.nodes),
                        cells=self._materialise(space_description.get('cells', None)),
                        cell_tags=self._materialise(space_description.get('cell_tags', None)),
                        facet_tags=facet_tags,
                        cell_type=cell_type,
                        geometry_degree=int(geometry_degree),
                    )

                else:
                    raise ValueError('Unrecognised space description with keys %s'
                                     % sorted(space_description.keys()))

                self._grid.space = space

            if 'time' in description and self._grid.time is None:
                time = Time(start=description.time.start,
                            stop=description.time.stop,
                            step=description.time.step,
                            num=description.time.num)

                self._grid.time = time

            if 'slow_time' in description and self._grid.slow_time is None:
                slow_time = SlowTime(frame_step=description.slow_time.frame_step,
                                     frame_rate=description.slow_time.frame_rate,
                                     num_frame=description.slow_time.num_frame,
                                     acq_step=description.slow_time.acq_step,
                                     acq_rate=description.slow_time.acq_rate,
                                     num_acq=description.slow_time.num_acq)

                self._grid.slow_time = slow_time

            kwargs['filename'] = kwargs.pop('filename', file.filename)
            self.__set_desc__(description, **kwargs)

    @staticmethod
    def _materialise(value):
        """
        Read a lazily-loaded description entry into memory.

        Loading a description defaults to being lazy, which yields the open dataset with a
        ``load`` attribute attached rather than an ndarray. Anything that has to outlive the
        open file must be materialised explicitly.

        Parameters
        ----------
        value : object
            Entry of a loaded description.

        Returns
        -------
        object
            The entry, read into memory if it was lazy.

        """
        if hasattr(value, 'load'):
            return value.load()

        return value

    def grid_description(self):
        """
        Get a description of the grid of the object.

        Returns
        -------
        dict
            Description of the grid.

        """
        grid_description = dict()

        if self.space is not None:
            space = self.space
            if isinstance(space, Space):
                grid_description['space'] = {
                    'shape': space.shape,
                    'spacing': space.spacing,
                    'extra': space.extra,
                    'absorbing': space.absorbing,
                }

            elif isinstance(space, MeshedSpace):
                # the connectivity travels with the nodes: a node table on its own is not a mesh
                # and cannot be handed back to a solver
                space_description = {'nodes': space.nodes}

                if space.cells is not None:
                    space_description['cells'] = space.cells

                if space.cell_tags is not None:
                    space_description['cell_tags'] = space.cell_tags

                # a facet tag names its facet by the nodes it is made of, so unlike a DOLFINx
                # facet index it means the same thing after the mesh is rebuilt and can be
                # written out. Flat keys rather than a nested dict, to stay within what the
                # HDF5 layer handles
                if space.facet_tags is not None:
                    space_description['facet_tag_nodes'] = space.facet_tags['nodes']
                    space_description['facet_tag_values'] = space.facet_tags['values']

                # the discretisation, without which the mesh cannot be rebuilt: a node and cell
                # table alone does not say what shape a cell is or to what degree it is mapped
                if space.cell_type is not None:
                    space_description['cell_type'] = space.cell_type

                space_description['geometry_degree'] = space.geometry_degree

                grid_description['space'] = space_description

            else:
                raise TypeError('Cannot serialise a grid with space of type %s'
                                % type(space).__name__)

        if self.time is not None:
            time = self.time
            grid_description['time'] = {
                'start': time.start,
                'stop': time.stop,
                'step': time.step,
                'num': time.num,
            }

        if self.slow_time is not None:
            slow_time = self.slow_time
            grid_description['slow_time'] = {
                'start': slow_time.start,
                'stop': slow_time.stop,
                'frame_step': slow_time.frame_step,
                'frame_rate': slow_time.frame_rate,
                'num_frame': slow_time.num_frame,
                'acq_step': slow_time.acq_step,
                'acq_rate': slow_time.acq_rate,
                'num_acq': slow_time.num_acq,
            }

        return grid_description

    def resample(self, grid=None, space=None, time=None, slow_time=None):
        super().resample(grid=grid, space=space, time=time, slow_time=slow_time)


class ProblemBase(GriddedSaved):
    """
    Base class for the different components of the problem that need to have access to it and
    that also create sub-problems.

    Parameters
    ----------
    name : str
        Name of the object.
    problem : Problem
        Problem to which the object belongs.
    grid : Grid or any of Space or Time
        Grid on which the object is defined

    """

    def __init__(self, **kwargs):
        problem = kwargs.pop('problem', None)

        if problem is not None:
            kwargs['space'] = kwargs.get('space', problem.space)
            kwargs['time'] = kwargs.get('time', problem.time)
            kwargs['slow_time'] = kwargs.get('slow_time', problem.slow_time)

        super().__init__(**kwargs)

        self._problem = problem

    @property
    def problem(self):
        """
        Access problem object.

        """
        return self._problem

    def sub_problem(self, shot, sub_problem):
        """
        Create a subset object for a certain shot.

        A SubProblem contains everything that is needed to fully determine how to run a particular shot.
        This method takes care of selecting the portions of the object that are needed
        for a given shot.

        By default, this has no effect.

        Parameters
        ----------
        shot : Shot
            Shot for which the SubProblem is being generated.
        sub_problem : SubProblem
            Container for the sub-problem being generated.

        Returns
        -------
        ProblemBase
            ProblemBase instance.

        """
        return self
