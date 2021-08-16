
import os

if os.environ.get('DISPLAY', None):
    import matplotlib
    matplotlib.use('TkAgg')

from .plot_scalar_fields import *
from .plot_vector_fields import *
from .plot_points import *
from .plot_traces import *
from .plot_show import *
