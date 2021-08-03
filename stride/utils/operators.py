
import numpy as np

from ..core import Operator
from ..problem import StructuredData


__all__ = ['Concatenate']


class Concatenate(Operator):

    def forward(self, *args, **kwargs):
        axis = kwargs.pop('axis', 0)

        concat_data = [np.array(each.data) for each in args]
        concat_data = np.stack(concat_data, axis=axis)

        concat = StructuredData(name='concat', data=concat_data)

        return concat

    def adjoint(self, d_concat, *args, **kwargs):
        axis = kwargs.pop('axis', 0)

        d_args = []
        for arg_i in range(len(args)):
            d_arg_i_data = np.take(d_concat.data, arg_i, axis=axis)

            d_arg_i = args[arg_i].alike(name='grad_%s' % args[arg_i].name,
                                        data=d_arg_i_data)
            d_args.append(d_arg_i)

        return tuple(d_args)
