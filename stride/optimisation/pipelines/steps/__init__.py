
from .filter_wavelets import FilterWavelets
from .filter_traces import FilterTraces
from .norm_per_shot import NormPerShot
from .norm_per_trace import NormPerTrace
from .norm_field import NormField
from .smooth_field import SmoothField
from .mask import Mask
from .mute_traces import MuteTraces
from .clip import Clip


steps_registry = {
    'filter_wavelets': FilterWavelets,
    'filter_traces': FilterTraces,
    'norm_per_shot': NormPerShot,
    'norm_per_trace': NormPerTrace,
    'norm_field': NormField,
    'smooth_field': SmoothField,
    'mask' : Mask,
    'mute_traces': MuteTraces,
    'clip': Clip,
}
