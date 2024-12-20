
import gc
import functools
import numpy as np
import scipy.ndimage
import scipy.interpolate
import resampy

try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    ENABLED_2D_PLOTTING = True

except ModuleNotFoundError:
    ENABLED_2D_PLOTTING = False

import mosaic
from mosaic.core.tessera import PickleClass
from mosaic.comms.compression import maybe_compress, decompress

from .base import GriddedSaved
from ..core import Variable
from .. import plotting


__all__ = ['Data', 'StructuredData', 'Scalar', 'ScalarField', 'VectorField', 'Traces',
           'SparseField', 'SparseCoordinates']


def inv_transform(x):
    return 1 / x


@mosaic.tessera
class Data(GriddedSaved, Variable):
    """
    Objects of this type represent Data defined over a grid and on which mathematical
    operations might be performed. This data might or might not be structured.

    """

    def __init__(self, **kwargs):
        GriddedSaved.__init__(self, **kwargs)
        Variable.__init__(self, **kwargs)

    def clear_grad(self):
        raise NotImplementedError('Unimplemented Data method clear_grad')

    def process_grad(self):
        raise NotImplementedError('Unimplemented Data method process_grad')

    def __add__(self, other):
        raise NotImplementedError('Operator + has not been implemented for class %s' % self.__class__.__name__)

    def __sub__(self, other):
        raise NotImplementedError('Operator - has not been implemented for class %s' % self.__class__.__name__)

    def __mul__(self, other):
        raise NotImplementedError('Operator * has not been implemented for class %s' % self.__class__.__name__)

    def __pow__(self, power, modulo=None):
        raise NotImplementedError('Operator ** has not been implemented for class %s' % self.__class__.__name__)

    def __truediv__(self, other):
        raise NotImplementedError('Operator / has not been implemented for class %s' % self.__class__.__name__)

    def __floordiv__(self, other):
        raise NotImplementedError('Operator // has not been implemented for class %s' % self.__class__.__name__)

    def __iadd__(self, other):
        raise NotImplementedError('Operator + has not been implemented for class %s' % self.__class__.__name__)

    def __isub__(self, other):
        raise NotImplementedError('Operator - has not been implemented for class %s' % self.__class__.__name__)

    def __imul__(self, other):
        raise NotImplementedError('Operator * has not been implemented for class %s' % self.__class__.__name__)

    def __ipow__(self, power, modulo=None):
        raise NotImplementedError('Operator ** has not been implemented for class %s' % self.__class__.__name__)

    def __itruediv__(self, other):
        raise NotImplementedError('Operator / has not been implemented for class %s' % self.__class__.__name__)

    def __ifloordiv__(self, other):
        raise NotImplementedError('Operator // has not been implemented for class %s' % self.__class__.__name__)

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rtruediv__ = __truediv__
    __rfloordiv__ = __floordiv__


@mosaic.tessera
class StructuredData(Data):
    """
    Objects of this type represent data defined over a structured grid.

    This grid is on which the data lives is fully defined by the ``shape`` parameter. Optionally,
    an ``extended_shape`` may be provided if the data is defined over an inner and extended domain.
    If an extended domain is defined, the ``inner`` parameter can be used to determine the position
    of the inner domain within the larger extended domain.

    Parameters
    ----------
    name : str
        Name of the data.
    shape : tuple
        Shape of the inner domain of the data.
    extended_shape : tuple, optional
        Shape of the extended domain of the data, defaults to the ``shape``.
    inner : tuple, optional
        Tuple of slices defining the location of the inner domain inside the
        extended domain, defaults to the inner domain being centred.
    dtype : data-type, optional
        Data type of the data, defaults to float32.
    data : ndarray, optional
        Data with which to initialise the internal buffer, defaults to a new array. By default,
        no copies of the buffer are made if provided.
    grid : Grid or any of Space or Time
        Grid on which the Problem is defined

    """

    def __init__(self, **kwargs):
        # hacky, but does the trick for now
        name = kwargs.get('name', None)
        if name is not None and 'vp' in name:
            kwargs['transform'] = kwargs.pop('transform', PickleClass(inv_transform))
        super().__init__(**kwargs)

        shape = kwargs.pop('shape', None)
        extended_shape = kwargs.pop('extended_shape', None)
        inner = kwargs.pop('inner', None)
        dtype = kwargs.pop('dtype', np.float32)
        compressed = kwargs.pop('compressed', False)
        compression = kwargs.pop('compression', None)

        data = kwargs.pop('data', None)
        if data is not None:
            shape = shape or data.shape

        if shape is not None:
            extended_shape = extended_shape or shape

            if inner is None:
                extra = [each_extended - each_shape
                         for each_extended, each_shape in zip(extended_shape, shape)]
                inner = tuple([slice(each_extra, each_extra + each_shape)
                               for each_extra, each_shape in zip(extra, shape)])

        self._shape = shape
        self._extended_shape = extended_shape
        self._inner = inner
        self._dtype = dtype
        self._compressed = compressed
        self._compression = compression

        self._data = None
        if data is not None:
            self._set_data(self.pad_data(data))

        self.grad = None
        self.prec = None

    def alike(self, *args, **kwargs):
        """
        Create a data object that shares its characteristics with this object.

        The same parameters as those given to ``__init__`` are valid here. Otherwise the
        new object will be configured to be like this one.

        Returns
        -------
        StructuredData
            Newly created StructuredData.

        """
        kwargs['shape'] = kwargs.pop('shape', self.shape)
        kwargs['extended_shape'] = kwargs.pop('extended_shape', self.extended_shape)
        kwargs['inner'] = kwargs.pop('inner', self.inner)
        kwargs['dtype'] = kwargs.pop('dtype', self.dtype)
        kwargs['grid'] = kwargs.pop('grid', self.grid)
        kwargs['compressed'] = kwargs.pop('compressed', self.compressed)
        kwargs['compression'] = kwargs.pop('compression', self._compression)
        kwargs['propagate_tessera'] = False

        return super().copy(*args, **kwargs)

    def detach(self, *args, **kwargs):
        """
        Create a copy of the variable that is detached from the original
        graph.

        Returns
        -------
        StructuredData
            Detached variable.

        """
        kwargs['shape'] = kwargs.pop('shape', self.shape)
        kwargs['extended_shape'] = kwargs.pop('extended_shape', self.extended_shape)
        kwargs['inner'] = kwargs.pop('inner', self.inner)
        kwargs['dtype'] = kwargs.pop('dtype', self.dtype)
        kwargs['grid'] = kwargs.pop('grid', self.grid)
        kwargs['compressed'] = kwargs.pop('compressed', self.compressed)
        kwargs['compression'] = kwargs.pop('compression', self._compression)
        kwargs['data'] = kwargs.pop('data', self._data)

        return super().detach(*args, **kwargs)

    def as_parameter(self, *args, **kwargs):
        """
        Create a copy of the variable that is detached from the original
        graph and re-initialised as a parameter.

        Returns
        -------
        StructuredData
            Detached variable.

        """
        kwargs['shape'] = kwargs.pop('shape', self.shape)
        kwargs['extended_shape'] = kwargs.pop('extended_shape', self.extended_shape)
        kwargs['inner'] = kwargs.pop('inner', self.inner)
        kwargs['dtype'] = kwargs.pop('dtype', self.dtype)
        kwargs['grid'] = kwargs.pop('grid', self.grid)
        kwargs['compressed'] = kwargs.pop('compressed', self.compressed)
        kwargs['compression'] = kwargs.pop('compression', self._compression)
        kwargs['data'] = kwargs.pop('data', self._data)

        return super().as_parameter(*args, **kwargs)

    def copy(self, **kwargs):
        """
        Create a deep copy of the data object.

        Returns
        -------
        StructuredData
            Newly created StructuredData.

        """
        cpy = self.alike(name=kwargs.pop('name', self._init_name), **kwargs)
        cpy.needs_grad = self.needs_grad
        cpy._set_data(self.extended_data.copy())

        if self.grad is not None:
            cpy.grad = self.grad.copy()

        if self.prec is not None:
            cpy.prec = self.prec.copy()

        return cpy

    def _set_data(self, data):
        if self.compressed:
            compression, data = maybe_compress(data)
            self._compression = compression

        self._data = data

    def _get_data(self):
        if self.compressed:
            data = decompress(self._compression, self._data)
            return np.frombuffer(data, self.dtype).reshape(self.shape)

        return self._data

    @property
    def data(self):
        """
        Data values inside the inner domain, as an ndarray.

        """
        if self._data is None:
            self.allocate()

        return self._get_data()[self._inner]

    @property
    def extended_data(self):
        """
        Data values inside the extended domain, as an ndarray.

        """
        if self._data is None:
            self.allocate()

        return self._get_data()

    @property
    def shape(self):
        """
        Shape of the inner domain, as a tuple.

        """
        return self._shape

    @property
    def extended_shape(self):
        """
        Shape of the extended domain, as a tuple.

        """
        return self._extended_shape

    @property
    def ndim(self):
        """
        Number of data dimensions.

        """
        return len(self._shape)

    @property
    def inner(self):
        """
        Slices that determine the location of the inner domain with respect to the extended domain,
        as a tuple of slices.

        """
        return self._inner

    @property
    def allocated(self):
        """
        Whether or not the data has been allocated.

        """
        return self._data is not None

    @property
    def dtype(self):
        """
        Data-type of the data.

        """
        return self._dtype

    @property
    def compressed(self):
        """
        Whether the data is compressed.

        """
        return self._compressed

    def clear_grad(self):
        """
        Initialise and clear the internal buffers for the gradient and preconditioner.

        Returns
        -------

        """
        if not self.needs_grad:
            return

        if self.grad is None:
            self.grad = self.alike(name='%s_grad' % self.name,
                                   shape=self.shape, extended_shape=self.shape,
                                   inner=None)
            self.grad.prec = self.alike(name='%s_prec' % self.name,
                                        shape=self.shape, extended_shape=self.shape,
                                        inner=None)

        self.grad.fill(0.)
        self.grad.prec.fill(0.)

        if hasattr(self, 'is_proxy') and self.is_proxy:
            self.set('grad', self.grad)

    def release_grad(self):
        """
        Release the internal buffers for the gradient and preconditioner.

        Returns
        -------

        """
        self.grad = None

    def process_grad(self, global_prec=True, **kwargs):
        """
        Process the gradient by applying the pre-conditioner to it.

        Parameters
        ----------
        global_prec : bool, optional
            Whether to apply preconditioner. Defaults to True.
        prec_scale : float, optional
            Condition scaling for the preconditioner.

        Returns
        -------

        """
        if not self.needs_grad:
            return

        if global_prec:
            self.grad.apply_prec(**kwargs)
        return self.grad

    def apply_prec(self, prec_scale=4.0, prec_smooth=None, prec_op=None, prec=None, **kwargs):
        """
        Apply a pre-conditioner to the current field.

        Parameters
        ----------
        prec_scale : float, optional
            Condition scaling for the preconditioner.
        prec_op : callable, optional
            Additional operation to apply to the preconditioner.
        prec_smooth : float, optional
            Smoothing to apply to the preconditioner.
        prec : StructuredData, optional
            Pre-conditioner to apply. Defaults to self.prec.

        Returns
        -------

        """
        prec = self.prec if prec is None else prec

        if prec is not None:
            if prec_smooth is not None:
                prec.data[:] = scipy.ndimage.gaussian_filter(prec.data, prec_smooth)

            prec_factor = np.sum(np.abs(prec.data))

            if prec_factor > 1e-31:
                num_points = np.prod(prec.shape)
                prec_factor = prec_scale * num_points / prec_factor
                prec.data[:] = prec.data * prec_factor + 1
                if prec_op is not None:
                    prec.data[:] = prec_op(prec.data)

                non_zero = np.abs(prec.data) > 0.
                self.data[non_zero] *= 1/prec.data[non_zero]

        return self

    def allocate(self):
        """
        Allocate the data if this has not been allocated yet.

        Returns
        -------

        """
        if self._data is None:
            self._set_data(np.empty(self._extended_shape, dtype=self._dtype))

    def deallocate(self, collect=False):
        """
        Deallocate the data.

        Parameters
        ----------
        collect : bool, optional

        Returns
        -------

        """
        if self._data is not None:
            del self._data
            self._data = None

            if collect:
                gc.collect()

    def fill(self, value):
        """
        Fill the data with a certain value

        Parameters
        ----------
        value : float
            Value with which to fill the data.

        Returns
        -------

        """
        if self._data is None:
            self.allocate()

        data = self.extended_data
        data.fill(value)
        self._set_data(data)

    def pad(self, smooth=False):
        """
        Pad internal data to match the extended shape of the StructuredData.

        Parameters
        ----------
        smooth : bool, optional
            Whether or not to smooth the padding area, defaults to False

        Returns
        -------

        """
        self._set_data(self.pad_data(self.data, smooth=smooth))

    def pad_data(self, data, smooth=False):
        """
        Pad input data to match the extended shape of the StructuredData.

        Parameters
        ----------
        data : ndarray
            Array to pad.
        smooth : bool, optional
            Whether or not to smooth the padding area, defaults to False

        Returns
        -------
        ndarray
            Padded array.

        """
        shape = data.shape
        pad_widths = [each_extended - each_shape for each_extended, each_shape in
                      zip(self._extended_shape, shape)]
        pad_widths = [[each // 2, each // 2] for each in pad_widths]

        if np.asarray(pad_widths).sum() > 0:
            data = np.pad(data, pad_widths, mode='edge')

            if smooth is True:
                for dim, width in zip(range(len(pad_widths)), pad_widths):
                    # : slices
                    all_ind = [slice(0, d) for d in data.shape]

                    for pos in range(0, width[0], 5):
                        sigma = pos / width[0] * 3.0

                        # Left slice for dimension
                        all_ind[dim] = slice(0, width[0] + 1 - pos)
                        data[tuple(all_ind)] = scipy.ndimage.gaussian_filter(data[tuple(all_ind)], sigma=sigma)

                        # right slice for dimension
                        all_ind[dim] = slice(data.shape[dim] - width[1] - 1 + pos, data.shape[dim])
                        data[tuple(all_ind)] = scipy.ndimage.gaussian_filter(data[tuple(all_ind)], sigma=sigma)

            return data

        else:
            return data

    def _prepare_op(self, other):
        res = self.copy()
        other_data = self._prepare_other(other)

        return res, other_data

    def _prepare_other(self, other):
        other_data = other
        if isinstance(other, StructuredData):
            if not isinstance(self, StructuredData):
                raise ValueError('Data of type %s and %s cannot be operated together' %
                                 (type(self), type(other)))

            other_data = other.extended_data

        return other_data

    @staticmethod
    def _op_grad(res, other, op):
        if not hasattr(other, 'grad'):
            return

        if res.grad is not None and other.grad is not None:
            res.grad = getattr(res.grad, op)(other.grad)
        elif other.grad is not None:
            res.grad = other.grad

    @staticmethod
    def _op_prec(res, other, op):
        if not hasattr(other, 'prec'):
            return

        if res.prec is not None and other.prec is not None:
            res.prec = getattr(res.prec, op)(other.prec)
        elif other.prec is not None:
            res.prec = other.prec

    def __add__(self, other):
        res, other_data = self._prepare_op(other)
        res.extended_data[:] = res.extended_data.__add__(other_data)

        self._op_grad(res, other, '__add__')
        self._op_prec(res, other, '__add__')

        return res

    def __sub__(self, other):
        res, other_data = self._prepare_op(other)
        res.extended_data[:] = res.extended_data.__sub__(other_data)

        self._op_grad(res, other, '__sub__')
        self._op_prec(res, other, '__sub__')

        return res

    def __mul__(self, other):
        res, other_data = self._prepare_op(other)
        res.extended_data[:] = res.extended_data.__mul__(other_data)

        self._op_grad(res, other, '__mul__')
        self._op_prec(res, other, '__mul__')

        return res

    def __pow__(self, power, modulo=None):
        res = self.copy()
        res.extended_data[:] = res.extended_data.__pow__(power)

        return res

    def __truediv__(self, other):
        res, other_data = self._prepare_op(other)
        res.extended_data[:] = res.extended_data.__truediv__(other_data)

        self._op_grad(res, other, '__truediv__')
        self._op_prec(res, other, '__truediv__')

        return res

    def __rtruediv__(self, other):
        res, other_data = self._prepare_op(other)
        res.extended_data[:] = res.extended_data.__rtruediv__(other_data)

        self._op_grad(res, other, '__rtruediv__')
        self._op_prec(res, other, '__rtruediv__')

        return res

    def __floordiv__(self, other):
        res, other_data = self._prepare_op(other)
        res.extended_data[:] = res.extended_data.__floordiv__(other_data)

        self._op_grad(res, other, '__floordiv__')
        self._op_prec(res, other, '__floordiv__')

        return res

    def __rfloordiv__(self, other):
        res, other_data = self._prepare_op(other)
        res.extended_data[:] = res.extended_data.__rfloordiv__(other_data)

        self._op_grad(res, other, '__rfloordiv__')
        self._op_prec(res, other, '__rfloordiv__')

        return res

    def __iadd__(self, other):
        res = self
        other_data = self._prepare_other(other)
        res.extended_data.__iadd__(other_data)

        self._op_grad(res, other, '__iadd__')
        self._op_prec(res, other, '__iadd__')

        return res

    def __isub__(self, other):
        res = self
        other_data = self._prepare_other(other)
        res.extended_data.__isub__(other_data)

        self._op_grad(res, other, '__isub__')
        self._op_prec(res, other, '__isub__')

        return res

    def __imul__(self, other):
        res = self
        other_data = self._prepare_other(other)
        res.extended_data.__imul__(other_data)

        self._op_grad(res, other, '__imul__')
        self._op_prec(res, other, '__imul__')

        return res

    def __ipow__(self, power, modulo=None):
        res = self
        res.extended_data.__ipow__(power)

        return res

    def __itruediv__(self, other):
        res = self
        other_data = self._prepare_other(other)
        res.extended_data.__itruediv__(other_data)

        self._op_grad(res, other, '__itruediv__')
        self._op_prec(res, other, '__itruediv__')

        return res

    def __ifloordiv__(self, other):
        res = self
        other_data = self._prepare_other(other)
        res.extended_data.__ifloordiv__(other_data)

        self._op_grad(res, other, '__ifloordiv__')
        self._op_prec(res, other, '__ifloordiv__')

        return res

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__

    def __get_desc__(self, **kwargs):
        if self._data is None:
            self.allocate()

        compression = False
        data = self.data
        if kwargs.pop('maybe_compress', False):
            compression, data = maybe_compress(data)

        inner = []
        for each in self._inner:
            inner.append([
                str(each.start),
                str(each.stop),
                str(each.step),
            ])

        description = {
            'shape': self._shape,
            'extended_shape': self._extended_shape,
            'inner': inner,
            'dtype': str(np.dtype(self._dtype)),
            'data': data,
            'compression': compression if compression is not None else False
        }

        return description

    def __set_desc__(self, description, **kwargs):
        self._shape = description.shape
        self._extended_shape = description.extended_shape
        self._dtype = np.dtype(description.dtype)

        inner = []
        for each in description.inner:
            each = [e.decode() if isinstance(e, bytes) else e for e in each]
            inner.append(slice(
                int(each[0]) if each[0] != 'None' else None,
                int(each[1]) if each[1] != 'None' else None,
                int(each[2]) if each[2] != 'None' else None,
            ))

        self._inner = tuple(inner)

        data = description.data
        if hasattr(data, 'load'):
            data = data.load()

        compression = description.get('compression', None)
        if compression:
            data = decompress(compression, data)
            data = np.frombuffer(data, self.dtype).reshape(self.shape)

        self._set_data(self.pad_data(data))


@mosaic.tessera
class Scalar(StructuredData):

    def __init__(self, **kwargs):
        kwargs['shape'] = (1,)
        super().__init__(**kwargs)


@mosaic.tessera
class ScalarField(StructuredData):
    """
    Objects of this type describe a scalar field defined over the spatial grid. Scalar fields
    can also be time-dependent.

    By default, the domain over which the field is defined is determined by the grid
    provided. This can be overwritten by providing a defined ``shape`` instead.

    Parameters
    ----------
    name : str
        Name of the data.
    time_dependent : bool, optional
        Whether or not the field is time-dependent, defaults to False.
    slow_time_dependent : bool, optional
        Whether or not the field is slow-time dependent, defaults to False.
    shape : tuple, optional
        Shape of the inner domain of the data.
    extended_shape : tuple, optional
        Shape of the extended domain of the data, defaults to the ``shape``.
    inner : tuple, optional
        Tuple of slices defining the location of the inner domain inside the
        extended domain, defaults to the inner domain being centred.
    dtype : data-type, optional
        Data type of the data, defaults to float32.
    grid : Grid or any of Space or Time
        Grid on which the Problem is defined

    """

    def __init__(self, **kwargs):
        data = kwargs.pop('data', None)

        super().__init__(**kwargs)

        time_dependent = kwargs.pop('time_dependent', False)
        slow_time_dependent = kwargs.pop('slow_time_dependent', False)
        self._time_dependent = time_dependent
        self._slow_time_dependent = slow_time_dependent

        if self.space is not None and self._shape is None:
            self._init_shape()
        elif data is not None:
            self._shape = self._extended_shape = data.shape
            self._inner = (slice(0, None),)*data.ndim

        if data is not None:
            self._data = self.pad_data(data)

        self.step_size = None

    def _init_shape(self, fill_shape=True):
        shape = ()
        extended_shape = ()
        inner = ()

        if self._time_dependent:
            shape += (self.time.num,)
            extended_shape += (self.time.extended_num,)
            inner += (self.time.inner,)

        if self._slow_time_dependent:
            shape += (self.slow_time.num,)
            extended_shape += (self.slow_time.extended_num,)
            inner += (self.slow_time.inner,)

        shape += self.space.shape
        extended_shape += self.space.extended_shape
        inner += self.space.inner

        if fill_shape:
            self._shape = shape
        self._extended_shape = extended_shape
        self._inner = inner

    def alike(self, *args, **kwargs):
        """
        Create a data object that shares its characteristics with this object.

        The same parameters as those given to ``__init__`` are valid here. Otherwise the
        new object will be configured to be like this one.

        Returns
        -------
        ScalarField
            Newly created ScalarField.

        """
        kwargs['time_dependent'] = kwargs.pop('time_dependent', self.time_dependent)
        kwargs['slow_time_dependent'] = kwargs.pop('slow_time_dependent', self.slow_time_dependent)

        return super().alike(*args, **kwargs)

    def detach(self, *args, **kwargs):
        """
        Create a copy of the variable that is detached from the original
        graph.

        Returns
        -------
        ScalarField
            Detached variable.

        """
        kwargs['time_dependent'] = kwargs.pop('time_dependent', self.time_dependent)
        kwargs['slow_time_dependent'] = kwargs.pop('slow_time_dependent', self.slow_time_dependent)

        return super().detach(*args, **kwargs)

    def as_parameter(self, *args, **kwargs):
        """
        Create a copy of the variable that is detached from the original
        graph and re-initialised as a parameter.

        Returns
        -------
        ScalarField
            Detached variable.

        """
        kwargs['time_dependent'] = kwargs.pop('time_dependent', self.time_dependent)
        kwargs['slow_time_dependent'] = kwargs.pop('slow_time_dependent', self.slow_time_dependent)

        return super().as_parameter(*args, **kwargs)

    @property
    def time_dependent(self):
        """
        Whether or not the field is time dependent.

        """
        return self._time_dependent

    @property
    def slow_time_dependent(self):
        """
        Whether or not the field is slow-time dependent.

        """
        return self._slow_time_dependent

    def stagger(self, stagger, method='nearest'):
        """
        Resample the internal (non-padded) data given some spatial staggering.

        Parameters
        ----------
        stagger : float or tuple of floats
            Stagger in each dimension.
        method : str, optional
            Method used for resampling, ``linear`` or ``nearest``, defaults
            to ``nearest``.

        Returns
        -------

        """
        self.data[:] = self.stagger_data(self.data, stagger, method=method)

    def stagger_data(self, data, stagger, method='nearest'):
        """
        Resample the data given some spatial staggering.

        Parameters
        ----------
        data : ndarray
            Data to stagger.
        stagger : float or tuple of floats
            Stagger in each dimension.
        method : str, optional
            Method used for resampling, ``linear`` or ``nearest``, defaults
            to ``nearest``.

        Returns
        -------
        ndarray
            Resampled data.

        """
        try:
            iter(stagger)
        except TypeError:
            stagger = (stagger,) * self.space.dim

        staggered_grid = [grid - h for grid, h in zip(self.space.grid, stagger)]
        mesh = np.asarray([each.ravel() for each in np.meshgrid(*self.space.grid, indexing='ij')]).T

        interp = scipy.interpolate.interpn(staggered_grid, data, mesh,
                                           method=method,
                                           bounds_error=False, fill_value=None)
        interp = interp.reshape(self.space.shape)

        return interp

    def resample(self, space=None, **kwargs):
        """
        Resample the internal (non-padded) data given some new space object.

        Parameters
        ----------
        space : Space
            New space.
        order : int, optional
            Order of the interplation, default is 3.
        prefilter : bool, optional
            Determines if the input array is prefiltered
            before interpolation. If downsampling, this defaults to ``False`` as an anti-aliasing filter
            will be applied instead. If upsampling, this defaults to ``True``.
        anti_alias : bool, optional
            Whether a Gaussian filter is applied to smooth the data before interpolation.
            The default is ``True``. This is only applied when downsampling.
        anti_alias_sigma : float or tuple of floats, optional
            Gaussian filter standard deviations used for the anti-aliasing filter.
            The default is (d - 1) / 2 where d is the downsampling factor and d > 1. When upsampling,
            d < 1, and no anti-aliasing filter is applied.

        Returns
        -------

        """

        if self.time_dependent or self.slow_time_dependent:
            data = self.data

            interp = []
            for t in range(data.shape[0]):
                interp.append(self.resample_data(data[t], space, **kwargs))

            interp = np.stack(interp, axis=0)

        else:
            interp = self.resample_data(self.data, space, **kwargs)

        self.grid.space = space
        self._init_shape()
        self._data = self.pad_data(interp)

    def resample_data(self, data, space, **kwargs):
        """
        Resample the data given some new space object.

        Parameters
        ----------
        data : ndarray
            Data to stagger.
        space : Space
            New space.
        order : int, optional
            Order of the interpolation, default is 3.
        prefilter : bool, optional
            Determines if the input array is prefiltered
            before interpolation. If downsampling, this defaults to ``False`` as an anti-aliasing filter
            will be applied instead. If upsampling, this defaults to ``True``.
        anti_alias : bool, optional
            Whether a Gaussian filter is applied to smooth the data before interpolation.
            The default is ``True``. This is only applied when downsampling.
        anti_alias_sigma : float or tuple of floats, optional
            Gaussian filter standard deviations used for the anti-aliasing filter.
            The default is (d - 1) / 2 where d is the downsampling factor and d > 1. When upsampling,
            d < 1, and no anti-aliasing filter is applied.

        Returns
        -------
        ndarray
            Resampled data.

        """
        order = kwargs.pop('order', 3)
        prefilter = kwargs.pop('prefilter', True)

        resampling_factors = np.array([dx_old/dx_new
                             for dx_old, dx_new in zip(self.space.spacing, space.spacing)])

        # Anti-aliasing is only required for down-sampling interpolation
        if any(factor < 1 for factor in resampling_factors):
            anti_alias = kwargs.pop('anti_alias', True)

            if anti_alias:
                anti_alias_sigma = kwargs.pop('anti_alias_sigma', None)

                if anti_alias_sigma is not None:
                    anti_alias_sigma = anti_alias_sigma * np.ones_like(resampling_factors)

                    if np.any(anti_alias_sigma < 0):
                        raise ValueError("Anti-alias standard dev. must be equal to or greater than zero")

                # Estimate anti-alias standard deviations if none provided
                else:
                    anti_alias_sigma = np.maximum(0, (1/resampling_factors - 1) / 2)

                data = scipy.ndimage.gaussian_filter(data, anti_alias_sigma)

                # Prefiltering is not necessary if anti-alias filter used
                prefilter = False

        interp = scipy.ndimage.zoom(data, resampling_factors,
                                    order=order, prefilter=prefilter)

        return interp

    def plot(self, **kwargs):
        """
        Plot the inner domain of the field.

        Parameters
        ----------
        kwargs
            Arguments for plotting.

        Returns
        -------
        axes
            Axes on which the plotting is done.

        """
        plot = kwargs.pop('plot', True)
        try:
            dim = self.space.dim
            origin = kwargs.pop('origin', self.space.origin)
            limit = kwargs.pop('limit', self.space.limit)
        except AttributeError:
            dim = self.ndim
            origin = (0,)*self.ndim
            limit = self.shape

        if (self.time_dependent or self.slow_time_dependent) and dim == 2:
            def update(figure, axis, step):
                if len(axis.images):
                    axis.images[-1].colorbar.remove()
                axis.clear()

                kwargs.pop('time_range', None)
                self._plot(self.data[int(step)], origin=origin, limit=limit, axis=axis,
                           **kwargs)
                axis.set_title(axis.get_title() + ' - time step %d' % step)

                figure.canvas.draw_idle()

            axis = self._plot_time(update, **kwargs)

        elif self.slow_time_dependent:
            axis = self._plot(self.data[0], origin=origin, limit=limit, **kwargs)

        else:
            axis = self._plot(self.data, origin=origin, limit=limit, **kwargs)

        if plot is True:
            plotting.show(axis)

        return axis

    def extended_plot(self, **kwargs):
        """
        Plot the extended domain of the field.

        Parameters
        ----------
        kwargs
            Arguments for plotting.

        Returns
        -------
        axes
            Axes on which the plotting is done.

        """
        plot = kwargs.pop('plot', True)
        try:
            dim = self.space.dim
            origin = kwargs.pop('origin', self.space.pml_origin)
            limit = kwargs.pop('limit', self.space.extended_limit)
        except AttributeError:
            dim = self.ndim
            origin = (0,)*self.ndim
            limit = self.extended_shape

        if (self.time_dependent or self.slow_time_dependent) and dim == 2:
            def update(figure, axis, step):
                if len(axis.images):
                    axis.images[-1].colorbar.remove()
                axis.clear()

                kwargs.pop('time_range', None)
                self._plot(self.extended_data[int(step)], origin=origin, limit=limit, axis=axis,
                           **kwargs)
                axis.set_title(axis.get_title() + ' - time step %d' % step)

                figure.canvas.draw_idle()

            axis = self._plot_time(update, **kwargs)

        elif self.slow_time_dependent:
            axis = self._plot(self.extended_data[0], origin=origin, limit=limit, **kwargs)

        else:
            axis = self._plot(self.extended_data, origin=origin, limit=limit, **kwargs)

        if plot is True:
            plotting.show(axis)

        return axis

    def _plot(self, data, **kwargs):
        title = kwargs.pop('title', self.name)

        axis = plotting.plot_scalar_field(data, title=title, **kwargs)

        return axis

    def _plot_time(self, update, **kwargs):
        if not ENABLED_2D_PLOTTING:
            return None

        figure, axis = plt.subplots(1, 1)
        plt.subplots_adjust(bottom=0.25)
        axis.margins(x=0)

        ax_shot = plt.axes([0.15, 0.1, 0.7, 0.03])
        time_range = kwargs.get('time_range', (0, self._data.shape[0]))
        slider = Slider(ax_shot, 'time',
                        time_range[0], time_range[1]-1,
                        valinit=0, valstep=1)

        update = functools.partial(update, figure, axis)
        update(0)

        slider.on_changed(update)
        axis.slider = slider

        return axis

    def __get_desc__(self, **kwargs):
        description = super().__get_desc__(**kwargs)
        description['time_dependent'] = self._time_dependent
        description['slow_time_dependent'] = self._slow_time_dependent
        description['step_size'] = self.step_size

        return description

    def __set_desc__(self, description, **kwargs):
        self._shape = description.shape
        self._dtype = np.dtype(description.dtype)
        self._time_dependent = description.time_dependent
        self._slow_time_dependent = description.get('slow_time_dependent', False)

        if self.space is not None:
            self._init_shape(fill_shape=False)

        data = description.data
        if hasattr(data, 'load'):
            data = data.load()

        compression = description.get('compression', None)
        if compression:
            data = decompress(compression, data)
            data = np.frombuffer(data, self.dtype).reshape(self.shape)

        self.extended_data[:] = self.pad_data(data)


@mosaic.tessera
class VectorField(ScalarField):
    """
    Objects of this type describe a vector field defined over the spatial grid. Vector fields
    can also be time-dependent.

    By default, the domain over which the field is defined is determined by the grid
    provided. This can be overwritten by providing a defined ``shape`` instead.

    Parameters
    ----------
    name : str
        Name of the data.
    dim : int, optional
        Number of dimensions for the vector field, defaults to the spatial dimensions.
    time_dependent : bool, optional
        Whether or not the field is time-dependent, defaults to False.
    slow_time_dependent : bool, optional
        Whether or not the field is slow-time dependent, defaults to False.
    shape : tuple, optional
        Shape of the inner domain of the data.
    extended_shape : tuple, optional
        Shape of the extended domain of the data, defaults to the ``shape``.
    inner : tuple, optional
        Tuple of slices defining the location of the inner domain inside the
        extended domain, defaults to the inner domain being centred.
    dtype : data-type, optional
        Data type of the data, defaults to float32.
    grid : Grid or any of Space or Time
        Grid on which the Problem is defined

    """

    def __init__(self, **kwargs):
        dim = kwargs.pop('dim', None)

        if 'space' in kwargs and dim is None:
            dim = kwargs['space'].dim
        elif 'grid' in kwargs and dim is None:
            dim = kwargs['grid'].space.dim

        self._dim = dim

        super().__init__(**kwargs)

    def _init_shape(self, fill_shape=True):
        shape = ()
        extended_shape = ()
        inner = ()

        if self._time_dependent:
            shape += (self.time.num,)
            extended_shape += (self.time.extended_num,)
            inner += (self.time.inner,)

        if self._slow_time_dependent:
            shape += (self.slow_time.num,)
            extended_shape += (self.slow_time.extended_num,)
            inner += (self.slow_time.inner,)

        shape += (self._dim,)
        extended_shape += (self._dim,)
        inner += (slice(0, None),)

        shape += self.space.shape
        extended_shape += self.space.extended_shape
        inner += self.space.inner

        if fill_shape:
            self._shape = shape
        self._extended_shape = extended_shape
        self._inner = inner

    def alike(self, *args, **kwargs):
        """
        Create a data object that shares its characteristics with this object.

        The same parameters as those given to ``__init__`` are valid here. Otherwise the
        new object will be configured to be like this one.

        Returns
        -------
        VectorField
            Newly created VectorField.

        """
        kwargs['dim'] = kwargs.pop('dim', self.dim)

        return super().alike(*args, **kwargs)

    def detach(self, *args, **kwargs):
        """
        Create a copy of the variable that is detached from the original
        graph.

        Returns
        -------
        VectorField
            Detached variable.

        """
        kwargs['dim'] = kwargs.pop('dim', self.dim)

        return super().detach(*args, **kwargs)

    def as_parameter(self, *args, **kwargs):
        """
        Create a copy of the variable that is detached from the original
        graph and re-initialised as a parameter.

        Returns
        -------
        VectorField
            Detached variable.

        """
        kwargs['dim'] = kwargs.pop('dim', self.dim)

        return super().as_parameter(*args, **kwargs)

    @property
    def dim(self):
        """
        Number of dimensions of the vector field.

        """
        return self._dim

    def plot_dims(self, **kwargs):
        """
        Plot separated dimensions of the field.

        Parameters
        ----------

        Returns
        -------

        """
        axes = kwargs.pop('axes', None)
        plot = kwargs.pop('plot', True)
        origin = kwargs.pop('origin', self.space.origin)
        limit = kwargs.pop('limit', self.space.limit)

        if axes is None:
            figure, axes = plt.subplots(1, self.dim)

        for dim in range(self.dim):
            title = kwargs.pop('title', '%s_%d' % (self.name, dim))

            super()._plot(self.data[dim], origin=origin, limit=limit,
                          title=title, axis=axes[dim], **kwargs)

        if plot is True:
            plotting.show(axes)

        return axes

    def _plot(self, data, **kwargs):
        title = kwargs.pop('title', self.name)

        axis = plotting.plot_vector_field(data, title=title, **kwargs)

        return axis

    def __get_desc__(self, **kwargs):
        description = super().__get_desc__(**kwargs)
        description['dim'] = self._dim

        return description

    def __set_desc__(self, description, **kwargs):
        self._shape = description.shape
        self._dtype = np.dtype(description.dtype)
        self._time_dependent = description.time_dependent
        self._slow_time_dependent = description.get('slow_time_dependent', False)
        self._dim = description.dim

        if self.space is not None:
            self._init_shape(fill_shape=False)

        data = description.data
        if hasattr(data, 'load'):
            data = data.load()

        compression = description.get('compression', None)
        if compression:
            data = decompress(compression, data)
            data = np.frombuffer(data, self.dtype).reshape(self.shape)

        self.extended_data[:] = self.pad_data(data)


@mosaic.tessera
class Traces(StructuredData):
    """
    Objects of this type describe a set of time traces defined over the time grid.

    By default, the domain over which the field is defined is determined by the time grid
    provided. This can be overwritten by providing a defined ``shape`` instead.

    Parameters
    ----------
    name : str
        Name of the data.
    transducer_ids : list
        List of IDs to which the time traces correspond.
    shape : tuple, optional
        Shape of the inner domain of the data.
    extended_shape : tuple, optional
        Shape of the extended domain of the data, defaults to the ``shape``.
    inner : tuple, optional
        Tuple of slices defining the location of the inner domain inside the
        extended domain, defaults to the inner domain being centred.
    dtype : data-type, optional
        Data type of the data, defaults to float32.
    grid : Grid or any of Space or Time
        Grid on which the Problem is defined

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        data = kwargs.get('data', None)

        transducer_ids = kwargs.pop('transducer_ids',
                                    list(range(data.shape[0])) if data is not None else None)
        self._transducer_ids = transducer_ids

        if self._transducer_ids is not None and self._shape is None:
            shape = (len(self._transducer_ids), self.time.num)
            extended_shape = (len(self._transducer_ids), self.time.extended_num)
            inner = (slice(0, None), self.time.inner)

            self._shape = shape
            self._extended_shape = extended_shape
            self._inner = inner

    def alike(self, *args, **kwargs):
        """
        Create a data object that shares its characteristics with this object.

        The same parameters as those given to ``__init__`` are valid here. Otherwise the
        new object will be configured to be like this one.

        Returns
        -------
        Traces
            Newly created Traces.

        """
        kwargs['transducer_ids'] = kwargs.pop('transducer_ids', self.transducer_ids)

        return super().alike(*args, **kwargs)

    def detach(self, *args, **kwargs):
        """
        Create a copy of the variable that is detached from the original
        graph.

        Returns
        -------
        Traces
            Detached variable.

        """
        kwargs['transducer_ids'] = kwargs.pop('transducer_ids', self.transducer_ids)

        return super().detach(*args, **kwargs)

    def as_parameter(self, *args, **kwargs):
        """
        Create a copy of the variable that is detached from the original
        graph and re-initialised as a parameter.

        Returns
        -------
        Traces
            Detached variable.

        """
        kwargs['transducer_ids'] = kwargs.pop('transducer_ids', self.transducer_ids)

        return super().as_parameter(*args, **kwargs)

    @property
    def transducer_ids(self):
        """
        List of transducer IDs associated with the traces.

        """
        return self._transducer_ids

    @property
    def num_transducers(self):
        """
        Number of transducers.

        """
        return len(self._transducer_ids)

    def get(self, id):
        """
        Get one trace based on a transducer ID, selecting the inner domain.

        Parameters
        ----------
        id : int
            Transducer ID.

        Returns
        -------
        1d-array
            Time trace.

        """
        if self._data is None:
            self.allocate()

        index = list(self._transducer_ids).index(id)
        return self.data[index, :]

    def get_extended(self, id):
        """
        Get one trace based on a transducer ID, selecting the extended domain.

        Parameters
        ----------
        id : int
            Transducer ID.

        Returns
        -------
        1d-array
            Time trace.

        """
        if self._data is None:
            self.allocate()

        index = list(self._transducer_ids).index(id)
        return self.extended_data[index, :]

    def plot(self, **kwargs):
        """
        Plot the inner domain of the traces as a shot gather.

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
        if self.time.num == self.shape[1]:
            time_axis = self.time.grid / 1e-6
        else:
            time_axis = np.arange(self.shape[1])

        axis = plotting.plot_gather(self.transducer_ids, time_axis, self.data,
                                    title=title, **kwargs)

        if plot is True:
            plotting.show(axis)

        return axis

    def plot_one(self, id, **kwargs):
        """
        Plot the the inner domain of one of the traces.

        Parameters
        ----------
        id : int
            Transducer ID.
        kwargs
            Arguments for plotting.

        Returns
        -------
        axes
            Axes on which the plotting is done.

        """
        title = kwargs.pop('title', self.name)
        plot = kwargs.pop('plot', True)
        trace = self.get(id)
        time_axis = self.time.grid / 1e-6

        axis = plotting.plot_trace(time_axis, trace,
                                   title=title, **kwargs)

        if plot is True:
            plotting.show(axis)

        return axis

    def _resample(self, factor, new_num, **kwargs):
        sr_orig = 1
        sr_new = factor

        if self.allocated:
            data = resampy.resample(self.data, sr_orig, sr_new, axis=1)  # resample
            if data.shape[-1] < new_num:
                data = np.pad(data, ((0, 0), (0, new_num-data.shape[-1])), mode='constant', constant_values=0)
            elif data.shape[-1] > new_num:
                data = data[:, :new_num]
            new_traces = Traces(name=self.name, grid=self.grid, transducer_ids=self._transducer_ids, data=data)
        else:
            new_traces = Traces(name=self.name, grid=self.grid, transducer_ids=self._transducer_ids)

        return new_traces

    def __get_desc__(self, **kwargs):
        description = super().__get_desc__(**kwargs)
        description['num_transducers'] = self.num_transducers
        description['transducer_ids'] = self._transducer_ids

        return description

    def __set_desc__(self, description, **kwargs):
        super().__set_desc__(description, **kwargs)

        self._transducer_ids = description.transducer_ids


@mosaic.tessera
class SparseField(StructuredData):
    """
    Objects of this type describe a sparse field defined at discrete points. Sparse fields
    can also be time-dependent.

    Parameters
    ----------
    name : str
        Name of the data.
    num : int
        Number of points in the field.
    dim : int
        Number of dimensions at every point in the field, defaults to 1.
    time_dependent : bool, optional
        Whether or not the field is time-dependent, defaults to False.
    slow_time_dependent : bool, optional
        Whether or not the field is slow-time dependent, defaults to False.
    shape : tuple, optional
        Shape of the inner domain of the data.
    extended_shape : tuple, optional
        Shape of the extended domain of the data, defaults to the ``shape``.
    inner : tuple, optional
        Tuple of slices defining the location of the inner domain inside the
        extended domain, defaults to the inner domain being centred.
    dtype : data-type, optional
        Data type of the data, defaults to float32.
    grid : Grid or any of Space or Time
        Grid on which the Problem is defined

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        num = kwargs.pop('num', 1)
        dim = kwargs.pop('dim', 1)
        time_dependent = kwargs.pop('time_dependent', False)
        slow_time_dependent = kwargs.pop('slow_time_dependent', False)

        self._num = num
        self._dim = dim
        self._time_dependent = time_dependent
        self._slow_time_dependent = slow_time_dependent

        if self._shape is None:
            self._init_shape()

    def _init_shape(self, fill_shape=True):
        shape = ()
        extended_shape = ()
        inner = ()

        if self._time_dependent:
            shape += (self.time.num,)
            extended_shape += (self.time.extended_num,)
            inner += (self.time.inner,)

        if self._slow_time_dependent:
            shape += (self.slow_time.num,)
            extended_shape += (self.slow_time.extended_num,)
            inner += (self.slow_time.inner,)

        if self.dim > 1:
            shape += (self.num, self.dim)
            extended_shape += (self.num, self.dim)
            inner += (slice(0, None), slice(0, None))
        else:
            shape += (self.num,)
            extended_shape += (self.num,)
            inner += (slice(0, None),)

        if fill_shape:
            self._shape = shape
        self._extended_shape = extended_shape
        self._inner = inner

    def alike(self, *args, **kwargs):
        """
        Create a data object that shares its characteristics with this object.

        The same parameters as those given to ``__init__`` are valid here. Otherwise the
        new object will be configured to be like this one.

        Returns
        -------
        SparseCoordinates
            Newly created ScalarField.

        """
        kwargs['num'] = kwargs.pop('num', self.num)
        kwargs['dim'] = kwargs.pop('dim', self.dim)
        kwargs['time_dependent'] = kwargs.pop('time_dependent', self.time_dependent)
        kwargs['slow_time_dependent'] = kwargs.pop('slow_time_dependent', self.time_dependent)

        return super().alike(*args, **kwargs)

    def detach(self, *args, **kwargs):
        """
        Create a copy of the variable that is detached from the original
        graph.

        Returns
        -------
        SparseCoordinates
            Detached variable.

        """
        kwargs['num'] = kwargs.pop('num', self.num)
        kwargs['dim'] = kwargs.pop('dim', self.dim)
        kwargs['time_dependent'] = kwargs.pop('time_dependent', self.time_dependent)
        kwargs['slow_time_dependent'] = kwargs.pop('slow_time_dependent', self.time_dependent)

        return super().detach(*args, **kwargs)

    def as_parameter(self, *args, **kwargs):
        """
        Create a copy of the variable that is detached from the original
        graph and re-initialised as a parameter.

        Returns
        -------
        ScalarField
            Detached variable.

        """
        kwargs['num'] = kwargs.pop('num', self.num)
        kwargs['dim'] = kwargs.pop('dim', self.dim)
        kwargs['time_dependent'] = kwargs.pop('time_dependent', self.time_dependent)
        kwargs['slow_time_dependent'] = kwargs.pop('slow_time_dependent', self.time_dependent)

        return super().as_parameter(*args, **kwargs)

    @property
    def num(self):
        """
        Number of elements in the field.

        """
        return self._num

    @property
    def dim(self):
        """
        Number of elements in the field.

        """
        return self._dim

    @property
    def time_dependent(self):
        """
        Whether or not the field is time dependent.

        """
        return self._time_dependent

    @property
    def slow_time_dependent(self):
        """
        Whether or not the field is slow-time dependent.

        """
        return self._slow_time_dependent

    def plot(self, *args, **kwargs):
        pass

    def __get_desc__(self, **kwargs):
        description = super().__get_desc__(**kwargs)
        description['num'] = self._num
        description['dim'] = self._dim
        description['time_dependent'] = self._time_dependent
        description['slow_time_dependent'] = self._slow_time_dependent

        return description

    def __set_desc__(self, description, **kwargs):
        self._shape = description.shape
        self._dtype = np.dtype(description.dtype)
        self._num = description.num
        self._dim = description.dim
        self._slow_time_dependent = description.get('slow_time_dependent', False)

        if self.space is not None:
            self._init_shape(fill_shape=False)

        data = description.data
        if hasattr(data, 'load'):
            data = data.load()

        self.extended_data[:] = self.pad_data(data)


@mosaic.tessera
class SparseCoordinates(SparseField):
    """
    Objects of this type describe a sparse set of coordinates defined over the spatial grid. Coordinates
    can also be time-dependent.

    Parameters
    ----------
    name : str
        Name of the data.
    num : int
        Number of particles in the field.
    time_dependent : bool, optional
        Whether or not the field is time-dependent, defaults to False.
    slow_time_dependent : bool, optional
        Whether or not the field is slow-time dependent, defaults to True.
    shape : tuple, optional
        Shape of the inner domain of the data.
    extended_shape : tuple, optional
        Shape of the extended domain of the data, defaults to the ``shape``.
    inner : tuple, optional
        Tuple of slices defining the location of the inner domain inside the
        extended domain, defaults to the inner domain being centred.
    dtype : data-type, optional
        Data type of the data, defaults to float32.
    grid : Grid or any of Space or Time
        Grid on which the Problem is defined

    """

    def __init__(self, **kwargs):
        dim = kwargs.pop('dim', None)

        if 'space' in kwargs and dim is None:
            dim = kwargs['space'].dim
        elif 'grid' in kwargs and dim is None:
            dim = kwargs['grid'].space.dim

        super().__init__(dim=dim, **kwargs)

    def plot(self, **kwargs):
        """
        Plot the particle the field.

        Parameters
        ----------
        kwargs
            Arguments for plotting.

        Returns
        -------
        axes
            Axes on which the plotting is done.

        """
        plot = kwargs.pop('plot', True)

        if self.slow_time_dependent:
            if self.space.dim == 2:
                def update(figure, axis, step):
                    axis.clear()

                    self._plot(self.data[int(step)], axis=axis, **kwargs)
                    axis.set_title(axis.get_title() + ' - slow time step %d' % step)

                    figure.canvas.draw_idle()

                axis = self._plot_time(update)

            else:
                slow_t = kwargs.pop('slow_t', 0)
                axis = self._plot(self.data[slow_t], **kwargs)
        else:
            axis = self._plot(self.data, **kwargs)

        if plot is True:
            plotting.show(axis)

        return axis

    def _plot(self, data, **kwargs):
        title = kwargs.pop('title', self.name)
        colour = kwargs.pop('colour', 'k')
        size = kwargs.pop('size', 1)
        origin = kwargs.pop('origin', self.space.origin)
        limit = kwargs.pop('limit', self.space.limit)

        data = data.reshape((-1, self.space.dim))

        axis = plotting.plot_points(data, title=title, colour=colour, size=size,
                                    origin=origin, limit=limit, **kwargs)

        return axis

    def _plot_time(self, update):
        if not ENABLED_2D_PLOTTING:
            return None

        figure, axis = plt.subplots(1, 1)
        plt.subplots_adjust(bottom=0.25)
        axis.margins(x=0)

        ax_shot = plt.axes([0.15, 0.1, 0.7, 0.03])
        slider = Slider(ax_shot, 'time',
                        0, self.slow_time.num-1,
                        valinit=0, valstep=1)

        update = functools.partial(update, figure, axis)
        update(0)

        slider.on_changed(update)
        axis.slider = slider

        return axis
