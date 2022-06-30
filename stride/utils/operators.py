
import numpy as np

from ..core import Operator
from ..problem import StructuredData


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


class Concatenate(Operator):

    def forward(self, *args, **kwargs):
        axis = kwargs.pop('axis', 0)

        concat_data = [np.array(each.data) for each in args]
        concat_data = np.stack(concat_data, axis=axis)

        concat = StructuredData(name='concat', data=concat_data, grid=args[0].grid)

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
