
import os


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
    try:
        if not os.environ.get('DISPLAY', None):
            raise ModuleNotFoundError
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
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
    try:
        if not os.environ.get('DISPLAY', None):
            raise ModuleNotFoundError
        from mayavi import mlab
    except ModuleNotFoundError:
        return None

    if not isinstance(figure, list):
        figure = [figure]

    for _figure in figure:
        if hasattr(_figure, 'scene3d'):
            _figure.configure_traits()

        else:
            mlab.show()
        mlab.close(all=True)


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
    plot_3d = True

    try:
        if not os.environ.get('DISPLAY', None):
            raise ModuleNotFoundError
        from mayavi.core.ui.api import MlabSceneModel
    except ModuleNotFoundError:
        plot_3d = False

    if isinstance(figure, list):
        _figure = figure[0]
    else:
        _figure = figure

    if plot_3d and (isinstance(_figure, MlabSceneModel) or hasattr(_figure, 'scene3d')):
        show_3d(figure)

    else:
        show_2d(figure)
