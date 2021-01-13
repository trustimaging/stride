
from stride.utils import filters

from ..pipeline import PipelineStep


class Step(PipelineStep):

    def __init__(self, **kwargs):
        self.f_min = kwargs.pop('f_min', None)
        self.f_max = kwargs.pop('f_max', None)

    def apply(self, wavelets, **kwargs):
        time = wavelets.time

        f_min = self.f_min*time.step / 0.750 if self.f_min is not None else 0
        f_max = self.f_max*time.step / 0.750 if self.f_max is not None else 0

        if self.f_min is None and self.f_max is not None:
            filtered = filters.lowpass_filter_fir(wavelets.extended_data, f_max)
            wavelets.extended_data[:] = filtered

        elif self.f_min is not None and self.f_max is None:
            filtered = filters.highpass_filter_fir(wavelets.extended_data, f_min)
            wavelets.extended_data[:] = filtered

        elif self.f_min is not None and self.f_max is not None:
            filtered = filters.bandpass_filter_fir(wavelets.extended_data, f_min, f_max)
            wavelets.extended_data[:] = filtered

        return wavelets
