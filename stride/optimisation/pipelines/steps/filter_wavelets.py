
from stride.utils import filters

from ....core import Operator


class FilterWavelets(Operator):
    """
    Filter wavelets to 3/4 of the set frequencies.

    Parameters
    ----------
    f_min : float, optional
        Lower value for the frequency filter, defaults to None (no lower filtering).
    f_max : float, optional
        Upper value for the frequency filter, defaults to None (no upper filtering).
    filter_type : str, optional
        Type of filter to apply, from ``butterworth`` (default for band pass and high pass),
        ``fir``, or ``cos`` (default for low pass).

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.f_min = kwargs.pop('f_min', None)
        self.f_max = kwargs.pop('f_max', None)

        self.filter_type = kwargs.pop('filter_type', None)

        self.dilation = kwargs.pop('filter_wavelets_dilation', 0.25)

    def forward(self, wavelets, **kwargs):
        return self._apply(wavelets, **kwargs)

    def adjoint(self, d_wavelets, wavelets, **kwargs):
        return self._apply(d_wavelets, adjoint=True, **kwargs)

    def _apply(self, wavelets, **kwargs):
        time = wavelets.time

        f_min = kwargs.pop('f_min', self.f_min)
        f_max = kwargs.pop('f_max', self.f_max)
        dilation = kwargs.pop('filter_wavelets_dilation', self.dilation)

        f_min_dim_less = dilation*f_min*time.step if f_min is not None else 0
        f_max_dim_less = 1/dilation*f_max*time.step if f_max is not None else 0

        out_wavelets = wavelets.alike(name='filtered_%s' % wavelets.name)

        if f_min is None and f_max is not None:
            pass_type = 'lowpass'
            args = (f_max_dim_less,)
        elif f_min is not None and f_max is None:
            pass_type = 'highpass'
            args = (f_min_dim_less,)
        elif f_min is not None and f_max is not None:
            pass_type = 'bandpass'
            args = (f_min_dim_less, f_max_dim_less)
        else:
            out_wavelets.extended_data[:] = wavelets.extended_data
            return out_wavelets

        default_filter_type = 'cos' if f_min is None else 'butterworth'
        filter_type = kwargs.pop('filter_type', self.filter_type or default_filter_type)

        method_name = '%s_filter_%s' % (pass_type, filter_type)
        method = getattr(filters, method_name)

        filtered = method(wavelets.extended_data, *args, zero_phase=False, **kwargs)

        out_wavelets.extended_data[:] = filtered

        return out_wavelets
