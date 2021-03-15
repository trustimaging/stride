
from . import devito

try:
    from stride_private.problem_types.acoustic import devito
except ImportError:
    pass
