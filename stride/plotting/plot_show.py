
import os

try:
    if not os.environ.get('DISPLAY', None):
        raise ModuleNotFoundError

    from mayavi import mlab
    from mayavi.core.ui.api import MlabSceneModel

    ENABLED_3D_PLOTTING = True

except ModuleNotFoundError:
    ENABLED_3D_PLOTTING = False

try:
    if not os.environ.get('DISPLAY', None):
        raise ModuleNotFoundError

    import matplotlib.pyplot as plt

    ENABLED_2D_PLOTTING = True

except ModuleNotFoundError:
    ENABLED_2D_PLOTTING = False


def show_2d(figure=None):
    """
    Utility function to show a Bokeh figure.

    Parameters
    ----------
    figure : object
        Bokeh figure to show.

    Returns
    -------

    """
    if not ENABLED_2D_PLOTTING:
        return None

    plt.show()


def show_3d(figure):
    """
    Utility function to show a MayaVi figure.

    Parameters
    ----------
    figure : object
        MayaVi figure to show.

    Returns
    -------

    """
    if not ENABLED_3D_PLOTTING:
        return None

    if not isinstance(figure, list):
        figure = [figure]

    for _figure in figure:
        if hasattr(_figure, 'scene3d'):
            _figure.configure_traits()

        else:
            mlab.show()


def show(figure=None):
    """
    Utility function to show a figure regardless of the library being used.

    Parameters
    ----------
    figure : object, optional
        matplotlib or MayaVi figure to show.

    Returns
    -------

    """
    if isinstance(figure, list):
        _figure = figure[0]
    else:
        _figure = figure

    if ENABLED_3D_PLOTTING and (isinstance(_figure, MlabSceneModel) or hasattr(_figure, 'scene3d')):
        show_3d(figure)

    else:
        show_2d(figure)
