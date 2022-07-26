
import os
import functools


from .volume_slicer import volume_slicer


__all__ = ['plot_scalar_field', 'plot_scalar_field_2d', 'plot_scalar_field_3d']


def prepare_plot_arguments(wrapped):
    @functools.wraps(wrapped)
    def _prepare_plot_arguments(field, data_range=(None, None), origin=None, limit=None,
                                axis=None, palette='viridis', title=None, **kwargs):

        space_scale = 1e-3
        if limit is None:
            limit = field.T.shape

        else:
            limit = tuple(each/space_scale for each in limit)

        if origin is None:
            origin = tuple([0 for _ in range(len(limit))])

        else:
            origin = tuple(each/space_scale for each in origin)

        return wrapped(field,
                       data_range=data_range, limit=limit, origin=origin,
                       axis=axis, palette=palette, title=title, **kwargs)

    return _prepare_plot_arguments


@prepare_plot_arguments
def plot_scalar_field_2d(field, data_range=(None, None), origin=None, limit=None,
                         axis=None, palette='viridis', title=None, add_colourbar=True, **kwargs):
    """
    Utility function to plot a 2D scalar field using matplotlib.

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

    Returns
    -------
    Axis
        Generated axis.

    """
    try:
        if not os.environ.get('DISPLAY', None):
            raise ModuleNotFoundError
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    if axis is None:
        figure, axis = plt.subplots(1, 1)

    slice = kwargs.pop('slice', None)
    if slice is not None:
        field = field[slice]

    default_kwargs = dict(cmap=palette,
                          vmin=data_range[0], vmax=data_range[1],
                          aspect='auto',
                          origin='lower',
                          interpolation='bicubic')

    if slice is None:
        default_kwargs['extent'] = [origin[0], limit[0], origin[1], limit[1]]

    default_kwargs.update(kwargs)

    im = axis.imshow(field.T, **default_kwargs)

    if origin is None or limit is None or slice is not None:
        axis.set_xlabel('x')
        axis.set_ylabel('y')

    else:
        axis.set_xlabel('x (mm)')
        axis.set_ylabel('y (mm)')

    if title is not None:
        axis.set_title(title)

    if add_colourbar:
        plt.colorbar(im, ax=axis)

    return axis


@prepare_plot_arguments
def plot_scalar_field_3d(field, data_range=(None, None), origin=None, limit=None,
                         axis=None, palette='viridis', title=None, **kwargs):
    """
    Utility function to plot a 3D scalar field using MayaVi.

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
                           is_vector=False,
                           **default_kwargs)

    return window


def plot_scalar_field(field, data_range=(None, None), origin=None, limit=None,
                      axis=None, palette='viridis', title=None, **kwargs):
    """
    Utility function to plot a scalar field using matplotib (2D) or MayaVi (3D).

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
    matplotlib or MayaVi figure
        Generated matplotlib or MayaVi figure

    """

    if len(field.shape) > 2:
        axis = plot_scalar_field_3d(field,
                                    data_range=data_range, limit=limit, origin=origin,
                                    axis=axis, palette=palette, title=title, **kwargs)

    else:
        axis = plot_scalar_field_2d(field,
                                    data_range=data_range, limit=limit, origin=origin,
                                    axis=axis, palette=palette, title=title, **kwargs)

    return axis
