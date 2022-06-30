import os
import functools
import numpy as np

from .volume_slicer import volume_slicer

__all__ = ['plot_vector_field', 'plot_vector_field_2d', 'plot_vector_field_3d']


def prepare_plot_arguments(wrapped):
    @functools.wraps(wrapped)
    def _prepare_plot_arguments(field, data_range=(None, None), origin=None, limit=None,
                                axis=None, palette='viridis', title=None, **kwargs):

        space_scale = 1e-3
        if limit is None:
            limit = field.T.shape

        else:
            limit = tuple(each / space_scale for each in limit)

        if origin is None:
            origin = tuple([0 for _ in range(len(limit))])

        else:
            origin = tuple(each / space_scale for each in origin)

        return wrapped(field,
                       data_range=data_range, limit=limit, origin=origin,
                       axis=axis, palette=palette, title=title, **kwargs)

    return _prepare_plot_arguments


@prepare_plot_arguments
def plot_vector_field_2d(field, data_range=(None, None), origin=None, limit=None,
                         axis=None, palette='viridis', title=None, add_colourbar=True, **kwargs):
    """
    Utility function to plot a 2D vector field using matplotlib.

    Parameters
    ----------
    field : ScalarField or VectorField
        Field to be plotted
    data_range : tuple, optional
        Range of the data, defaults to (min(field), max(field)).
    origin : tuple, optional
        Origin of the axes of the plot, defaults to zero.
    limit : tuple, optional
        Extent of the axes of the plot, defaults to the spatial extent.
    axis : matplotlib axis, optional
        Axis in which to make the plotting, defaults to new empty one.
    palette : str, optional
        Palette to use in the plotting, defaults to plasma.
    title : str, optional
        Figure title, defaults to empty title.
    add_colourbar : bool, optional
        Whether to add colourbar to plot, defaults to ``True``.
    undersampling : int, optional
        Undersampling of the quiver field, defaults to 1.
    equal_arrows : bool, optional
        Whether to make all arrows the same length.

    Returns
    -------
    Axis
        Generated axis.

    """
    try:
        if not os.environ.get('DISPLAY', None):
            raise ModuleNotFoundError
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ModuleNotFoundError:
        return None

    if axis is None:
        figure, axis = plt.subplots(1, 1)

    slice = kwargs.pop('slice', None)
    if slice is not None:
        field = field[slice]
        origin = None
        limit = None

    if origin is None or limit is None:
        start = (0, 0)
        stop = field.shape[1:]
    else:
        start = origin
        stop = limit

    undersampling = kwargs.pop('undersampling', 1)
    equal_arrows = kwargs.pop('equal_arrows', False)

    x = np.linspace(start[0], stop[0], field.shape[1], endpoint=True)
    y = np.linspace(start[1], stop[1], field.shape[2], endpoint=True)
    x = x[::undersampling]
    y = y[::undersampling]
    x, y = np.meshgrid(x, y)

    mag = np.sqrt(field[0].astype(np.float64) ** 2 + field[1].astype(np.float64) ** 2).T
    undersampled_mag = mag[::undersampling, ::undersampling]
    max_mag = np.max(undersampled_mag)

    u = [field[0, ::undersampling, ::undersampling].T,
         field[1, ::undersampling, ::undersampling].T]

    if equal_arrows:
        u[0] = undersampling * u[0] / (undersampled_mag + 1e-31)
        u[1] = undersampling * u[1] / (undersampled_mag + 1e-31)
    else:
        u[0] = undersampling * u[0] / (max_mag + 1e-31)
        u[1] = undersampling * u[1] / (max_mag + 1e-31)

    cmap_name = 'Rainbow_Blended_Black'
    cmap_filename = os.path.join(os.path.dirname(__file__), 'cmaps/' + cmap_name + '.txt')
    cmap_values = []
    with open(cmap_filename, 'r') as file:
        for line in file.readlines():
            line = [float(each) / 255. for each in line.strip().split('\t')]
            line[-1] = 1.
            cmap_values.append(line)

    cmap = ListedColormap(cmap_values, name=cmap_name)

    default_kwargs = dict(cmap=cmap,
                          vmin=data_range[0], vmax=data_range[1],
                          aspect='equal',
                          origin='lower',
                          interpolation='bicubic')

    if origin is not None and limit is not None:
        default_kwargs['extent'] = [origin[0], limit[0], origin[1], limit[1]]

    default_kwargs.update(kwargs)
    im_1 = axis.imshow(mag, **default_kwargs)

    max_dir = np.sqrt(np.max(x) ** 2 + np.max(y) ** 2)
    width = 0.005 / max_dir
    hl = 3.01 * max_dir

    default_kwargs = dict(color='white',
                          angles='xy',
                          scale_units='xy',
                          scale=1.,
                          width=width,
                          headwidth=hl*0.8,
                          headaxislength=hl,
                          headlength=hl,)
    default_kwargs.update(kwargs)
    default_kwargs.pop('vmax', None)
    default_kwargs.pop('vmin', None)
    im_2 = axis.quiver(x, y, u[0], u[1], **default_kwargs)

    if origin is None or limit is None:
        axis.set_xlabel('x')
        axis.set_ylabel('y')

    else:
        axis.set_xlabel('x (mm)')
        axis.set_ylabel('y (mm)')

    if title is not None:
        axis.set_title(title)

    if add_colourbar:
        plt.colorbar(im_1, ax=axis)

    return axis


@prepare_plot_arguments
def plot_vector_field_3d(field, data_range=(None, None), origin=None, limit=None,
                         axis=None, palette='viridis', title=None, **kwargs):
    """
    Utility function to plot a 3D vector field using MayaVi.

    Parameters
    ----------
    field : ScalarField or VectorField
        Field to be plotted
    data_range : tuple, optional
        Range of the data, defaults to (min(field), max(field)).
    origin : tuple, optional
        Origin of the axes of the plot, defaults to zero.
    limit : tuple, optional
        Extent of the axes of the plot, defaults to the spatial extent.
    axis : MayaVi axis, optional
        Axis in which to make the plotting, defaults to new empty one.
    palette : str, optional
        Palette to use in the plotting, defaults to plasma.
    title : str, optional
        Figure title, defaults to empty title.

    Returns
    -------
    MayaVi figure
        Generated MayaVi figure

    """
    try:
        if not os.environ.get('DISPLAY', None):
            raise ModuleNotFoundError
        from mayavi.core.ui.api import MlabSceneModel
    except ModuleNotFoundError:
        return None

    if axis is None:
        axis = MlabSceneModel()

    default_kwargs = dict(colourmap=palette,
                          scene3d=axis,
                          data_range=data_range)
    default_kwargs.update(kwargs)

    window = volume_slicer(data=field,
                           is_vector=True,
                           **default_kwargs)

    return window


def plot_vector_field(field, data_range=(None, None), origin=None, limit=None,
                      axis=None, palette='viridis', title=None, **kwargs):
    """
    Utility function to plot a vector field using matplotib (2D) or MayaVi (3D).

    Parameters
    ----------
    field : VectorField
        Field to be plotted
    data_range : tuple, optional
        Range of the data, defaults to (min(field), max(field)).
    origin : tuple, optional
        Origin of the axes of the plot, defaults to zero.
    limit : tuple, optional
        Extent of the axes of the plot, defaults to the spatial extent.
    axis : MayaVi axis, optional
        Axis in which to make the plotting, defaults to new empty one.
    palette : str, optional
        Palette to use in the plotting, defaults to plasma.
    title : str, optional
        Figure title, defaults to empty title.

    Returns
    -------
    matplotlib or MayaVi figure
        Generated matplotlib or MayaVi figure

    """

    if len(field.shape) > 3:
        axis = plot_vector_field_3d(field,
                                    data_range=data_range, limit=limit, origin=origin,
                                    axis=axis, palette=palette, title=title, **kwargs)

    else:
        axis = plot_vector_field_2d(field,
                                    data_range=data_range, limit=limit, origin=origin,
                                    axis=axis, palette=palette, title=title, **kwargs)

    return axis
