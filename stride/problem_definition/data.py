
import gc
import copy
import numpy as np

from .base import GriddedSaved
from .. import plotting


__all__ = ['StructuredData', 'ScalarField', 'VectorField', 'Traces']


class Data(GriddedSaved):

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


class StructuredData(Data):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._data = None

        shape = kwargs.pop('shape', None)
        extended_shape = kwargs.pop('extended_shape', None)
        inner = kwargs.pop('inner', None)
        dtype = kwargs.pop('dtype', np.float32)

        if shape is not None:
            extended_shape = extended_shape or shape

            if inner is None:
                extra = [each_extended - each_shape for each_extended, each_shape in zip(extended_shape, shape)]
                inner = [slice(each_extra, each_extra + each_shape) for each_extra, each_shape in zip(extra, shape)]

        self._shape = shape
        self._extended_shape = extended_shape
        self._inner = inner
        self._dtype = dtype

    def alike(self, *args, **kwargs):
        kwargs['shape'] = kwargs.pop('shape', self.shape)
        kwargs['extended_shape'] = kwargs.pop('extended_shape', self.extended_shape)
        kwargs['inner'] = kwargs.pop('inner', self.inner)
        kwargs['dtype'] = kwargs.pop('dtype', self.dtype)
        kwargs['grid'] = kwargs.pop('grid', self.grid)

        return self.__class__(*args, **kwargs)

    @property
    def data(self):
        if self._data is None:
            self.allocate()

        return self._data[tuple(self._inner)]

    @property
    def extended_data(self):
        if self._data is None:
            self.allocate()

        return self._data

    @property
    def shape(self):
        return self._shape

    @property
    def extended_shape(self):
        return self._extended_shape

    @property
    def inner(self):
        return self._inner

    @property
    def allocated(self):
        return self._data is not None

    @property
    def dtype(self):
        return self._dtype

    def allocate(self):
        if self._data is None:
            self._data = np.zeros(self._extended_shape, dtype=self._dtype)

    def deallocate(self):
        if self._data is not None:
            del self._data
            self._data = None
            gc.collect()

    def fill(self, value):
        if self._data is None:
            self.allocate()

        self._data.fill(value)

    def pad_data(self, data):
        shape = data.shape
        pad_widths = [each_extended - each_shape for each_extended, each_shape in
                      zip(self._extended_shape, shape)]
        pad_widths = [[each // 2, each // 2] for each in pad_widths]

        return np.pad(data, pad_widths, mode='edge')

    def _prepare_op(self, other):
        res = self.copy()
        other_data = self._prepare_other(other)

        return res, other_data

    def _prepare_other(self, other):
        other_data = other
        if isinstance(other, StructuredData):
            if not isinstance(other, StructuredData):
                raise ValueError('Data of type %s and %s cannot be operated together' %
                                 (type(self), type(other)))

            other_data = other.extended_data

        return other_data

    def __add__(self, other):
        res, other_data = self._prepare_op(other)
        res.extended_data[:] = res.extended_data.__add__(other_data)

        return res

    def __sub__(self, other):
        res, other_data = self._prepare_op(other)
        res.extended_data[:] = res.extended_data.__sub__(other_data)

        return res

    def __mul__(self, other):
        res, other_data = self._prepare_op(other)
        res.extended_data[:] = res.extended_data.__mul__(other_data)

        return res

    def __pow__(self, power, modulo=None):
        res = self.copy()
        res.extended_data[:] = res.extended_data.__pow__(power)

        return res

    def __truediv__(self, other):
        res, other_data = self._prepare_op(other)
        res.extended_data[:] = res.extended_data.__truediv__(other_data)

        return res

    def __floordiv__(self, other):
        res, other_data = self._prepare_op(other)
        res.extended_data[:] = res.extended_data.__floordiv__(other_data)

        return res

    def __iadd__(self, other):
        res = self
        other_data = self._prepare_other(other)
        res.extended_data.__iadd__(other_data)

        return res

    def __isub__(self, other):
        res = self
        other_data = self._prepare_other(other)
        res.extended_data.__isub__(other_data)

        return res

    def __imul__(self, other):
        res = self
        other_data = self._prepare_other(other)
        res.extended_data.__imul__(other_data)

        return res

    def __ipow__(self, power, modulo=None):
        res = self
        res.extended_data.__ipow__(power)

        return res

    def __itruediv__(self, other):
        res = self
        other_data = self._prepare_other(other)
        res.extended_data.__itruediv__(other_data)

        return res

    def __ifloordiv__(self, other):
        res = self
        other_data = self._prepare_other(other)
        res.extended_data.__ifloordiv__(other_data)

        return res

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rtruediv__ = __truediv__
    __rfloordiv__ = __floordiv__

    def copy(self):
        return copy.deepcopy(self)

    def __get_desc__(self):
        if self._data is None:
            self.allocate()

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
            'data': self.data,
        }

        return description

    def __set_desc__(self, description):
        self._shape = description.shape
        self._extended_shape = description.extended_shape
        self._dtype = np.dtype(description.dtype)

        inner = []
        for each in description.inner:
            inner.append(slice(
                int(each[0]) if each[0] != 'None' else None,
                int(each[1]) if each[1] != 'None' else None,
                int(each[2]) if each[2] != 'None' else None,
            ))

        self._inner = inner

        if hasattr(description.data, 'load'):
            data = description.data.load()
        else:
            data = description.data

        self.extended_data[:] = self.pad_data(data)


class ScalarField(StructuredData):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        time_dependent = kwargs.pop('time_dependent', False)
        self._time_dependent = time_dependent

        if self.space is not None and self._shape is None:
            shape = ()
            extended_shape = ()
            inner = ()
            if self._time_dependent:
                shape += (self.time.num,)
                extended_shape += (self.time.extended_num,)
                inner += (self.time.inner,)

            shape += self.space.shape
            extended_shape += self.space.extended_shape
            inner += self.space.inner

            self._shape = shape
            self._extended_shape = extended_shape
            self._inner = inner

    def alike(self, *args, **kwargs):
        kwargs['time_dependent'] = kwargs.pop('time_dependent', self.time_dependent)

        return super().alike(*args, **kwargs)

    @property
    def time_dependent(self):
        return self._time_dependent

    def __get_desc__(self):
        description = super().__get_desc__()
        description['time_dependent'] = self._time_dependent

        return description

    def __set_desc__(self, description):
        super().__set_desc__(description)

        self._time_dependent = description.time_dependent

    def plot(self, **kwargs):
        title = kwargs.pop('title', self.name)
        axis = plotting.plot_scalar_field(self.data, title=title,
                                          origin=self.space.origin, limit=self.space.limit,
                                          **kwargs)

        return axis

    def extended_plot(self, **kwargs):
        title = kwargs.pop('title', self.name)
        axis = plotting.plot_scalar_field(self.extended_data, title=title,
                                          origin=self.space.pml_origin, limit=self.space.extended_limit,
                                          **kwargs)

        return axis


class VectorField(ScalarField):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        dim = kwargs.pop('dim', False)
        self._dim = dim

        if self.space is not None and self._shape is not None:
            self._dim = dim or self.space.dim
            self._shape = (self._dim,) + self._shape
            self._extended_shape = (self._dim,) + self._extended_shape
            self._inner = (slice(0, None),) + self._inner

    def alike(self, *args, **kwargs):
        kwargs['dim'] = kwargs.pop('dim', self.dim)

        return super().alike(*args, **kwargs)

    @property
    def dim(self):
        return self._dim

    def __get_desc__(self):
        description = super().__get_desc__()
        description['dim'] = self._dim

        return description

    def __set_desc__(self, description):
        super().__set_desc__(description)

        self._dim = description.dim


class Traces(StructuredData):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        transducer_ids = kwargs.pop('transducer_ids', None)
        self._transducer_ids = transducer_ids

        if self._transducer_ids is not None and self._shape is None:
            shape = (len(self._transducer_ids), self.time.num)
            extended_shape = (len(self._transducer_ids), self.time.extended_num)
            inner = (slice(0, None), self.time.inner)

            self._shape = shape
            self._extended_shape = extended_shape
            self._inner = inner

    def alike(self, *args, **kwargs):
        kwargs['transducer_ids'] = kwargs.pop('transducer_ids', self.transducer_ids)

        return super().alike(*args, **kwargs)

    @property
    def transducer_ids(self):
        return self._transducer_ids

    @property
    def num_transducers(self):
        return len(self._transducer_ids)

    def get(self, id):
        if self._data is None:
            self.allocate()

        index = list(self._transducer_ids).index(id)
        return self.data[index, :]

    def get_extended(self, id):
        if self._data is None:
            self.allocate()

        index = list(self._transducer_ids).index(id)
        return self.extended_data[index, :]

    def __get_desc__(self):
        description = super().__get_desc__()
        description['num_transducers'] = self.num_transducers
        description['transducer_ids'] = self._transducer_ids

        return description

    def __set_desc__(self, description):
        super().__set_desc__(description)

        self._transducer_ids = description.transducer_ids

    def plot(self, **kwargs):
        title = kwargs.pop('title', self.name)
        time_axis = self.time.grid / 1e-6

        return plotting.plot_gather(self.transducer_ids, time_axis, self.data,
                                    title=title, **kwargs)

    def plot_one(self, index, **kwargs):
        title = kwargs.pop('title', self.name)
        trace = self.data[index, :]
        time_axis = self.time.grid / 1e-6

        return plotting.plot_trace(time_axis, trace,
                                   title=title, **kwargs)
