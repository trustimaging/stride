
import os

if os.environ.get('DISPLAY', None):
    try:
        import matplotlib
        matplotlib.use('TkAgg')
    except ImportError:
        pass

from .plot_scalar_fields import *
from .plot_vector_fields import *
from .plot_points import *
from .plot_traces import *
from .plot_show import *
