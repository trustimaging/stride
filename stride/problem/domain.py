
import numpy as np
from cached_property import cached_property


__all__ = ['Space', 'Time', 'SlowTime', 'Grid']


class Space:
    """
    This defines the spatial grid over which the problem is defined.

    The spatial grid consists of an inner domain defined by ``shape`` and
    an external padding defined by ``extra``. Within this extra region, a
    further sub-region is defined as absorbing for boundary purposes as defined
    by ``absorbing``.

    The ``spacing`` defines the axis-wise spacing of the grid.

    Parameters
    ----------
    shape : tuple
        Shape of the inner domain.
    spacing : tuple or float
        Axis-wise spacing of the grid, in metres.
    extra : tuple
        Amount of axis-wise extra space around the inner domain.
    absorbing : tuple
        Portion of the extra space that corresponds to absorbing boundaries.

    """

    def __init__(self, shape=None, spacing=None, extra=None, absorbing=None):
        self._set_properties(shape=shape, spacing=spacing, extra=extra, absorbing=absorbing)

    def _set_properties(self, shape, spacing, extra, absorbing):
        if isinstance(spacing, float):
            spacing = (spacing,)*len(shape)

        extra = extra or (0,)*len(shape)
        absorbing = absorbing or (0,)*len(shape)

        self.dim = len(shape)
        self.shape = tuple(shape)
        self.spacing = tuple(spacing)
        self.extra = tuple(extra)
        self.absorbing = tuple(absorbing)

        origin = (0,) * self.dim
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

    @property
    def size(self):
        """
        Alias for the domain limit.

        """
        return self.limit

    @property
    def extended_size(self):
        """
        Alias for the extended domain limit.

        """
        return self.extended_limit

    def resample(self, new_spacing, new_extra=None, new_absorbing=None):
        '''
        Method updates Space to the properties of the domain after resampling.

        Parameters
        ----------
        new_spacing: float or tuple(float)
            The new grid spacing.
        new_extra: tuple(int), optional
            The shape of the boundary for the new grid. Defaults to rescaling existing extra.
        new_absorbing: tuple(int), optional
            The shape of the absorbing boundary for the new grid. Defaults to rescaling
            existing absorbing.

        Returns
        -------

        '''

        if isinstance(new_spacing, float):
            new_spacing = (new_spacing,)*self.dim

        # NOTE you must be careful with numerical errors calculating new_shape, using:
        # new_shape = tuple((np.round(np.array(self.size) / np.array(new_spacing)) + 1).astype(int))
        # ... is not compatible with the method in scipy.ndimage.zoom
        old_spacing = self.spacing
        old_shape = self.shape
        resampling_factors = tuple([dx_old/dx_new
                for dx_old, dx_new in zip(old_spacing, new_spacing)])
        new_shape = tuple([int(round(n * m))
                for n, m in zip(old_shape, resampling_factors)])  # method matches scipy zoom

        if new_extra is None:
            new_extra = tuple((np.round(np.array(self.spacing) * (np.array(self.extra) - 1) /
                           np.array(new_spacing)) + 1).astype(int))

        if new_absorbing is None:
            new_absorbing = tuple((np.round(np.array(self.spacing) * (np.array(self.absorbing) - 1) /
                               np.array(new_spacing)) + 1).astype(int))

        self._set_properties(shape=new_shape, spacing=new_spacing, extra=new_extra, absorbing=new_absorbing)
        self._clear_cache('mesh_indices')
        self._clear_cache('extended_mesh_indices')
        self._clear_cache('mesh')
        self._clear_cache('extended_mesh')
        self._clear_cache('indices')
        self._clear_cache('extended_indices')
        self._clear_cache('grid')
        self._clear_cache('extended_grid')

    def _clear_cache(self, cached_property):
        '''
        Clear a cached property

        Parameters
        ----------
        cached_property: str
            The name of the property to remove from the cache.

        Returns
        -------
        '''
        try:
            del self.__dict__[cached_property]
        except:
            pass

    @property
    def inner(self):
        """
        Slices defining the inner domain, as a tuple of slices.

        """
        return tuple([slice(extra, extra + shape) for shape, extra in zip(self.shape, self.extra)])

    @property
    def inner_mask(self):
        """
        Tensor of the shape of the space grid with gridpoints wihtin inner domain set to 1
        and those outside set to 0, as an ndarray.

        """
        mask = np.zeros(self.extended_shape, dtype=np.float32)
        pml_slices = self.inner

        mask[pml_slices] = 1.

        return mask

    @cached_property
    def mesh_indices(self):
        """
        Create the mesh of indices in the inner domain, as a tuple
        of ndarray.

        """
        grid = [np.arange(0, shape) for shape in self.shape]
        return np.meshgrid(*grid)

    @cached_property
    def extended_mesh_indices(self):
        """
        Create the mesh of indices in the extended domain, as a tuple
        of ndarray.

        """
        grid = [np.arange(0, extended_shape) for extended_shape in self.extended_shape]
        return np.meshgrid(*grid)

    @cached_property
    def mesh(self):
        """
        Create the mesh of spatial locations in the inner domain, as a tuple
        of ndarray.

        """
        grid = self.grid
        return np.meshgrid(*grid, indexing='ij')

    @cached_property
    def extended_mesh(self):
        """
        Create the mesh of spatial locations the full, extended domain, as a tuple
        of ndarray.

        """
        grid = self.extended_grid
        return np.meshgrid(*grid, indexing='ij')

    @cached_property
    def indices(self):
        """
        Indices corresponding to the grid of the inner domain, as a tuple of 1d-arrays.

        """
        axes = [np.arange(0, shape) for shape in self.shape]
        return tuple(axes)

    @cached_property
    def extended_indices(self):
        """
        Indices corresponding to the grid of the extended domain, as a tuple of 1d-arrays.

        """
        axes = [np.arange(0, extended_shape) for extended_shape in self.extended_shape]
        return tuple(axes)

    @cached_property
    def grid(self):
        """
        Spatial points corresponding to the grid of the inner domain, as a tuple of 1d-arrays.

        """
        axes = [np.linspace(self.origin[dim], self.limit[dim], self.shape[dim],
                            endpoint=True, dtype=np.float32)
                for dim in range(self.dim)]
        return tuple(axes)

    @cached_property
    def extended_grid(self):
        """
        Spatial points corresponding to the grid of the extended domain, as a tuple of 1d-arrays.


        """
        axes = [np.linspace(self.pml_origin[dim], self.extended_limit[dim], self.extended_shape[dim],
                            endpoint=True, dtype=np.float32)
                for dim in range(self.dim)]
        return tuple(axes)


class Time:
    """
    This defines the temporal grid over which the problem is defined

    A time grid is fully defined by three of its arguments: start, stop, step or num.

    The time grid can be extended with a certain amount of padding, generating an
    inner domain and an extended domain, similar to that seen in the Space.

    Parameters
    ----------
    start : float, optional
        Point at which time starts, in seconds.
    step : float, optional
        Step between time points, in seconds.
    num : int, optional
        Number of time points in the grid.
    stop : float, optional
        Point at which time ends, in seconds.

    """

    def __init__(self, start=None, step=None, num=None, stop=None):
        self._set_properties(start, step, num, stop)

    def _set_properties(self, start=None, step=None, num=None, stop=None):
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

    def extend(self, extra):
        self.extra = extra
        self.extended_start = self.start - (self.extra[0] - 1)*self.step
        self.extended_stop = self.stop + (self.extra[1] - 1)*self.step
        self.extended_num = self.num + self.extra[0] + self.extra[1]

    def resample(self, new_step, new_num):
        """
        Resample a trace.

        Parameters
        ----------
        new_step : float
            The time spacing for the interpolated grid
        new_num : int
            The number of time-points, default is calculated to match input pulse
            length in [s]

        Returns
        -------
        """

        dt_in = self.step  # Extract current parameters
        start = self.start
        stop = self.stop
        num = self.num

        new_start = 0.  # Calculate new parameters

        interp_num = int((num)*(dt_in/new_step))
        interp_stop = new_start + new_step*(interp_num - 1)

        if new_num is not None:  # Do we need to pad the array or not?
            new_stop = new_start + new_step*(new_num - 1)
        else:
            new_num = interp_num
            new_stop = interp_stop

        self._set_properties(start=new_start, step=new_step, num=new_num)  # Update time

        try:
            del self.__dict__['grid']
        except:
            self._clear_cache('grid')

        try:
            del self.__dict__['extended_grid']
        except:
            self._clear_cache('extended_grid')

    def _clear_cache(self, cached_property):
        '''
        Clear a cached property

        Parameters
        ----------
        cached_property: str
            The name of the property to remove from the cache.

        Returns
        -------
        '''
        try:
            del self.__dict__[cached_property]
        except:
            pass

    @property
    def inner(self):
        """
        Slice defining the inner domain.

        """
        return slice(self.extra, self.extra + self.num)

    @cached_property
    def grid(self):
        """
        Time points corresponding to the grid of the inner domain, as a 1d-array.

        """
        return np.linspace(self.start, self.stop, self.num, endpoint=True, dtype=np.float32)

    @cached_property
    def extended_grid(self):
        """
        Time points corresponding to the grid of the extended domain, as a 1d-array.

        """
        return np.linspace(self.extended_start, self.extended_stop, self.extended_num, endpoint=True, dtype=np.float32)


class SlowTime:
    """
    This defines the slow temporal grid over which the problem is defined

    Parameters
    ----------
    frame_rate : float, optional
        Sampling frequency between frames, in Hz.
    acq_rate : float, optional
        Sampling frequency between acquisitions, in Hz.
    frame_step : float, optional
        Time step between frames, in seconds.
    acq_step : float, optional
        Time step between frames, in seconds.
    num_frame : int, optional
        Number of frames in the grid.
    num_acq : int, optional
        Number of acquisitions per frame.

    """

    def __init__(self, frame_rate=None, acq_rate=None,
                 frame_step=None, acq_step=None,
                 num_frame=None, num_acq=None):
        try:
            if frame_step is None:
                frame_step = 1/frame_rate
            else:
                frame_rate = 1/frame_step

        except:
            raise ValueError('Either freq or step has to be defined')

        if not isinstance(num_frame, int):
            raise TypeError('num_frames must be of type int')

        if acq_step is None and acq_rate is None:
            acq_step = 0
            acq_rate = -1
            num_acq = 1
        else:
            if not isinstance(num_acq, int):
                raise TypeError('num_acq must be of type int')

        if acq_step is None:
            acq_step = 1/acq_rate
        elif acq_rate is None:
            acq_rate = 1/acq_step

        if num_acq*acq_step > frame_step:
            raise ValueError('Acquisition step (%e s) too large for frame step (%e s).'
                             % (num_acq*acq_step, frame_step))

        start = 0.
        stop = start + frame_step * (num_frame - 1)

        self.start = start
        self.stop = stop
        self.frame_step = frame_step
        self.frame_rate = frame_rate
        self.num_frame = num_frame
        self.acq_step = acq_step
        self.acq_rate = acq_rate
        self.num_acq = num_acq

    def resample(self):
        raise NotImplementedError('Resampling has not been implemented yet')

    @property
    def num(self):
        """
        Total number of steps.

        """
        return self.num_frame*self.num_acq

    @property
    def extended_num(self):
        """
        Total number of steps.

        """
        return self.num

    @property
    def inner(self):
        """
        Slice defining the inner domain.

        """
        return slice(0, None)

    @cached_property
    def grid(self):
        """
        Time points corresponding to the grid, as a 1d-array.

        """
        if self.acq_rate > 0:
            start = 0.
            stop = start + self.acq_step * (self.num_acq - 1)

            grid = [np.linspace(start + self.frame_step*acq, stop + self.frame_step*acq,
                                self.num_acq, endpoint=True, dtype=np.float32)
                    for acq in range(self.num_frame)]

            return np.concatenate(grid)
        else:
            return np.linspace(self.start, self.stop, self.num_frame, endpoint=True, dtype=np.float32)


class Grid:
    """
    The grid is a container for the spatial and temporal grids.

    Parameters
    ----------
    space : Space
    time : Time
    slow_time : SlowTime
    """

    def __init__(self, space=None, time=None, slow_time=None):
        self.space = space
        self.time = time
        self.slow_time = slow_time
