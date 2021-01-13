
import numpy as np


__all__ = ['Space', 'Time', 'SlowTime']


class Space:

    def __init__(self, shape=None, spacing=None, extra=None, absorbing=None):
        self.dim = len(shape)
        self.shape = tuple(shape)
        self.spacing = tuple(spacing)
        self.extra = tuple(extra)
        self.absorbing = tuple(absorbing)

        origin = tuple([0] * self.dim)
        pml_origin = tuple([each_origin - each_spacing * each_extra for each_origin, each_spacing, each_extra in
                            zip(origin, spacing, extra)])

        extended_shape = tuple(np.array([dim + 2*added for dim, added in zip(shape, extra)]))
        size = tuple(np.array(spacing) * (np.array(shape) - 1))
        extended_size = tuple([each_origin + each_spacing * (each_shape + each_extra - 1)
                               for each_origin, each_spacing, each_shape, each_extra in zip(origin, spacing, shape, extra)])

        self.origin = origin
        self.pml_origin = pml_origin
        self.extended_shape = extended_shape
        self.limit = size
        self.extended_limit = extended_size

    def resample(self):
        pass

    def pml(self, damping_coefficient=None, mask=False):
        damp = np.ones(self.shape, dtype=np.float32) if mask else np.zeros(self.shape, dtype=np.float32)

        pad_widths = [(extra, extra) for extra in self.extra]
        damp = np.pad(damp, pad_widths, 'edge')

        if damping_coefficient is None:
            damping_coefficient = 2.0 * np.log(1.0 / 0.001) / np.max(self.extra)

        for dimension in range(self.dim):
            for index in range(self.absorbing[dimension]):
                # Damping coefficient
                pos = np.abs((self.absorbing[dimension] - index + 1) / float(self.absorbing[dimension]))
                val = damping_coefficient * (pos - np.sin(2 * np.pi * pos) / (2 * np.pi))

                # : slices
                all_ind = [slice(0, d) for d in damp.shape]
                # Left slice for dampening for dimension
                all_ind[dimension] = slice(index, index + 1)
                damp[tuple(all_ind)] += val / self.spacing[dimension]
                # right slice for dampening for dimension
                all_ind[dimension] = slice(damp.shape[dimension] - index, damp.shape[dimension] - index + 1)
                damp[tuple(all_ind)] += val / self.spacing[dimension]

        return damp

    @property
    def inner(self):
        return tuple([slice(extra, extra + shape) for shape, extra in zip(self.shape, self.extra)])

    def inner_time(self, time_slice):
        return tuple([time_slice] + list(self.inner))

    @property
    def inner_mask(self):
        mask = np.zeros(self.extended_shape, dtype=np.float32)
        pml_slices = self.inner

        mask[pml_slices] = 1.

        return mask

    @property
    def mesh_indices(self):
        grid = [np.arange(0, shape) for shape in self.shape]
        return np.meshgrid(*grid)

    @property
    def extended_mesh_indices(self):
        grid = [np.arange(0, extended_shape) for extended_shape in self.extended_shape]
        return np.meshgrid(*grid)

    @property
    def mesh(self):
        grid = self.grid
        return np.meshgrid(*grid)

    @property
    def extended_mesh(self):
        grid = self.extended_grid
        return np.meshgrid(*grid)

    @property
    def indices(self):
        axes = [np.arange(0, shape) for shape in self.shape]
        return axes

    @property
    def extended_indices(self):
        axes = [np.arange(0, extended_shape) for extended_shape in self.extended_shape]
        return axes

    @property
    def grid(self):
        axes = [np.linspace(self.origin[dim], self.limit[dim], self.shape[dim],
                            endpoint=True, dtype=np.float32)
                for dim in range(self.dim)]
        return axes

    @property
    def extended_grid(self):
        axes = [np.linspace(self.pml_origin[dim], self.extended_limit[dim], self.extended_shape[dim],
                            endpoint=True, dtype=np.float32)
                for dim in range(self.dim)]
        return axes


class Time:

    def __init__(self, start=None, step=None, num=None, stop=None):
        try:
            if start is None:
                start = stop - step*(num - 1)
            elif step is None:
                step = (stop - start)/(num - 1)
            elif num is None:
                num = int(np.ceil((stop - start)/step + 1))
                stop = step*(num - 1) + start
            elif stop is None:
                stop = start + step*(num - 1)

        except:
            raise ValueError('Three of args start, step, num and stop may be set')

        if not isinstance(num, int):
            raise TypeError('"input" argument must be of type int')

        self.start = start
        self.stop = stop
        self.step = step
        self.num = num

        self.extra = 0
        self.extended_start = start
        self.extended_stop = stop
        self.extended_num = num

    def extend(self, freq):
        if not isinstance(freq, float):
            self.extended_start = self.start
            self.extended_stop = self.stop
            self.extended_num = self.num
            self.extra = 0

            return

        extra = int((1/self.step)/freq * 0.75)

        self.extra = extra
        self.extended_start = self.start - (self.extra - 1)*self.step
        self.extended_stop = self.stop + (self.extra - 1)*self.step
        self.extended_num = self.num + 2*self.extra

    def resample(self):
        pass

    @property
    def inner(self):
        return slice(self.extra, self.extra + self.num)

    @property
    def grid(self):
        return np.linspace(self.start, self.stop, self.num, endpoint=True, dtype=np.float32)

    @property
    def extended_grid(self):
        return np.linspace(self.extended_start, self.extended_stop, self.extended_num, endpoint=True, dtype=np.float32)


class SlowTime:
    pass


class Grid:

    def __init__(self, space, time, slow_time):
        self.space = space
        self.time = time
        self.slow_time = slow_time
