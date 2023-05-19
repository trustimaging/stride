
import numpy as np
import mosaic

from ..core import Operator


__all__ = ['Add', 'Mul', 'Concatenate']


class Add(Operator):

    def forward(self, a, b, **kwargs):
        return a + b

    def adjoint(self, d_sum, a, b, **kwargs):
        return d_sum, d_sum


class Mul(Operator):

    def forward(self, a, b, **kwargs):
        return a * b

    def adjoint(self, d_mul, a, b, **kwargs):
        d_a = d_mul * b
        d_b = a * d_mul

        return d_a, d_b


@mosaic.tessera
class Concatenate(Operator):
    """
    Concatenate multiple StructuredData objects.

    Parameters
    ----------
    start_end: 2d-array, optional
        Array containing start and end indices pointing to each objects' position in the concatenated object.
        Shape should be (num_objects, 2)
    new_axis : bool, optional
        Whether to concatenate on a new axis. Defaults to False.
    axis : int, optional
        Axis to perform the concatenation. Defaults to 0.

    """
    def __init__(self, *args, **kwargs):
        self._start_end = kwargs.pop('start_end', None)
        self.new_axis = kwargs.pop('new_axis', False)
        self.axis = kwargs.pop('axis', 0)
        super().__init__(*args, **kwargs)

    async def forward(self, *args, **kwargs):
        """
        Parameters
        ----------
        args : Sequence[StructuredData]
            Sequence of StructuredData objects to be combined.
        start_end : 2d-array, optional
            Array containing start and end indices pointing to each objects' position in the concatenated object.
            Shape should be (num_objects, 2)
        axis : int, optional
            The axis to concatenate on. Defaults to 0.
        new_axis : bool, optional
            Whether to create a new axis when concatenating, or maintain the dimensions. Defaults to False.

        Returns
        -------
        StructuredData
            Concatenated data as a single StructuredData object.

        """
        new_axis = kwargs.pop('new_axis', None)
        axis = kwargs.pop('axis', None)
        start_end = kwargs.pop('start_end', None)

        if new_axis is not None:
            self.new_axis = new_axis

        if axis is not None:
            self.axis = axis

        if start_end is not None:
            # update if required
            self._start_end = start_end
        elif self._start_end is None:
            # build start_end from the arguments
            self.build_start_end(args)

        concat_data = [np.array(each.data) for each in args]

        if self.new_axis:
            concat_data = np.stack(concat_data, axis=self.axis)
        else:
            concat_data = np.vstack(concat_data)

        concat = args[0].alike(name='concat_%s' % args[0].name, data=concat_data, shape=None, extended_shape=None, inner=None)

        return concat

    async def adjoint(self, d_concat, *args, **kwargs):
        d_args = []
        for arg_i in range(len(args)):
            if self.new_axis:
                d_arg_i_data = np.take(d_concat.data, arg_i, axis=self.axis)
            else:
                # get start and end points of original arguments
                start, end = self._start_end[arg_i]
                indices = [i for i in range(start, end)]
                # preallocate shape for data as np.take has no keepdims option
                out_array = np.zeros((end-start, d_concat.data.shape[-1]), dtype=np.float32)
                # extract the data
                d_arg_i_data = np.take(d_concat.data, indices=indices, axis=self.axis, out=out_array)

            # insert into stride object
            d_arg_i = args[arg_i].alike(name='grad_%s' % args[arg_i].name, data=d_arg_i_data,
                                        shape=None, extended_shape=None, inner=None)

            d_args.append(d_arg_i)

        if len(d_args) > 1:
            return tuple(d_args)
        else:
            return d_args[0]

    def build_start_end(self, args):
        """
        Build start and end indices from the Sequence of StructuredData objects.

        Parameters
        ----------
        args : Sequence[StructuredData]
            Sequence of StructuredData objects to be combined.

        """
        start_end = np.zeros((len(args), 2), dtype=np.uint32)
        start = 0
        end = 0
        for idx, each in enumerate(args):
            end += each.data.shape[0]
            start_end[idx] = [start, end]
            start = end

        self._start_end = start_end
