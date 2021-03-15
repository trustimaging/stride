
import os

if os.environ.get('DISPLAY', None) is not None:
    import matplotlib
    matplotlib.use('TkAgg')

    from .plot_fields import *
    from .plot_points import *
    from .plot_traces import *
    from .plot_show import *
