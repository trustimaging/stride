
from .filter_traces import FilterTraces
from .norm_traces import NormPerShot, NormPerTrace
from .scale_traces import ScalePerShot, ScalePerTrace
from .norm_field import NormField
from .smooth_field import SmoothField
from .mask_field import MaskField
from .mute_traces import MuteTraces
from .clip import Clip
from .check_traces import CheckTraces
from .dump import Dump


steps_registry = {
    'filter_traces': FilterTraces,
    'norm_per_shot': NormPerShot,
    'norm_per_trace': NormPerTrace,
    'scale_per_shot': ScalePerShot,
    'scale_per_trace': ScalePerTrace,
    'norm_field': NormField,
    'smooth_field': SmoothField,
    'mask_field': MaskField,
    'mute_traces': MuteTraces,
    'clip': Clip,
    'check_traces': CheckTraces,
    'dump': Dump,
}
