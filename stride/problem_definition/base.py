
from mosaic import h5

from .domain import Space, Time, Grid


__all__ = ['Gridded', 'Saved', 'GriddedSaved', 'ProblemBase']


class Gridded:

    def __init__(self, grid=None, space=None, time=None, slow_time=None, **kwargs):
        if grid is None:
            grid = Grid(space, time, slow_time)

        self._grid = grid

    @property
    def grid(self):
        return self._grid

    @property
    def space(self):
        return self._grid.space

    @property
    def time(self):
        return self._grid.time

    @property
    def slow_time(self):
        return self._grid.slow_time

    def resample(self, grid=None, space=None, time=None, slow_time=None):
        pass


class Saved:

    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = name

    def dump(self, *args, **kwargs):
        description = self.__get_desc__()

        kwargs['parameter'] = kwargs.get('parameter', self.name)
        with h5.HDF5(*args, **kwargs, mode='w') as file:
            file.dump(description)

    def append(self, *args, **kwargs):
        if not h5.file_exists(*args, **kwargs):
            self.dump(*args, **kwargs)
            return

        description = self.__get_desc__()

        kwargs['parameter'] = kwargs.get('parameter', self.name)
        with h5.HDF5(*args, **kwargs, mode='a') as file:
            file.append(description)

    def load(self, *args, **kwargs):
        kwargs['parameter'] = self.name
        with h5.HDF5(*args, **kwargs, mode='r') as file:
            description = file.load()

            self.__set_desc__(description)

    def __get_desc__(self):
        return {}

    def __set_desc__(self, description):
        pass


class GriddedSaved(Saved, Gridded):

    def dump(self, *args, **kwargs):
        grid_description = self.grid_description()

        description = self.__get_desc__()
        grid_description.update(description)

        kwargs['parameter'] = kwargs.get('parameter', self.name)
        with h5.HDF5(*args, **kwargs, mode='w') as file:
            file.dump(grid_description)

    def append(self, *args, **kwargs):
        if not h5.file_exists(*args, **kwargs):
            self.dump(*args, **kwargs)
            return

        grid_description = self.grid_description()

        description = self.__get_desc__()
        grid_description.update(description)

        kwargs['parameter'] = kwargs.get('parameter', self.name)
        with h5.HDF5(*args, **kwargs, mode='a') as file:
            file.append(grid_description)

    def load(self, *args, **kwargs):
        kwargs['parameter'] = self.name
        with h5.HDF5(*args, **kwargs, mode='r') as file:
            description = file.load()

            # TODO If there's already a grid and they don't match, resample instead or overwriting
            if 'space' in description:
                space = Space(shape=description.space.shape,
                              spacing=description.space.spacing,
                              extra=description.space.extra,
                              absorbing=description.space.absorbing)

                self._grid.space = space

            if 'time' in description:
                time = Time(start=description.time.start,
                            stop=description.time.stop,
                            step=description.time.step,
                            num=description.time.num)

                self._grid.time = time

            if 'slow_time' in description:
                pass

            self.__set_desc__(description)

    def grid_description(self):
        grid_description = dict()

        if self.space is not None:
            space = self.space
            grid_description['space'] = {
                'shape': space.shape,
                'spacing': space.spacing,
                'extra': space.extra,
                'absorbing': space.absorbing,
            }

        if self.time is not None:
            time = self.time
            grid_description['time'] = {
                'start': time.start,
                'stop': time.stop,
                'step': time.step,
                'num': time.num,
            }

        if self.slow_time is not None:
            pass

        return grid_description


class ProblemBase(GriddedSaved):

    def __init__(self, name, problem, *args, **kwargs):
        if problem is not None:
            kwargs['space'] = kwargs.get('space', problem.space)
            kwargs['time'] = kwargs.get('time', problem.time)
            kwargs['slow_time'] = kwargs.get('slow_time', problem.slow_time)

        super().__init__(name, *args, **kwargs)

        self._problem = problem

    @property
    def problem(self):
        return self._problem

    def sub_problem(self, shot, sub_problem):
        return self
