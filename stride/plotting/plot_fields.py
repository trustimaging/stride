
import os
import functools
import warnings
try:
    if not os.environ.get('DISPLAY', None):
        raise ModuleNotFoundError

    os.environ['ETS_TOOLKIT'] = 'qt4'
    from wx import wxPyDeprecationWarning
    warnings.simplefilter(action='ignore', category=wxPyDeprecationWarning)

    from traits.api import HasTraits, Instance, Array, on_trait_change
    from traitsui.api import View, Item, HGroup, Group

    from tvtk.api import tvtk
    from tvtk.pyface.scene import Scene

    from mayavi import mlab
    from mayavi.core.api import PipelineBase, Source
    from mayavi.core.ui.api import SceneEditor, MayaviScene, MlabSceneModel

    ENABLED_3D_PLOTTING = True

except (ModuleNotFoundError, RuntimeError):
    ENABLED_3D_PLOTTING = False

try:
    if not os.environ.get('DISPLAY', None):
        raise ModuleNotFoundError

    import matplotlib.pyplot as plt

    ENABLED_2D_PLOTTING = True

except ModuleNotFoundError:
    ENABLED_2D_PLOTTING = False


if ENABLED_3D_PLOTTING:
    class VolumeSlicer(HasTraits):
        # The data to plot
        data = Array()

        # The 4 views displayed
        scene3d = Instance(MlabSceneModel, ())
        scene_x = Instance(MlabSceneModel, ())
        scene_y = Instance(MlabSceneModel, ())
        scene_z = Instance(MlabSceneModel, ())

        # The data source
        data_source = Instance(Source)

        # The image plane widgets of the 3D scene
        plane_widget_3d_x = Instance(PipelineBase)
        plane_widget_3d_y = Instance(PipelineBase)
        plane_widget_3d_z = Instance(PipelineBase)

        _axis_names = dict(x=0, y=1, z=2)

        def __init__(self, is_Vector, data_range, colourmap, **traits):
            self.is_Vector = is_Vector
            self.data_range = data_range
            self.colourmap = colourmap

            super(VolumeSlicer, self).__init__(**traits)

            # Force the creation of the image_plane_widgets:
            self.plane_widget_3d_x
            self.plane_widget_3d_y
            self.plane_widget_3d_z

        # Default values
        def _data_source_default(self):
            if self.is_Vector:
                return mlab.pipeline.vector_field(self.data,
                                                  figure=self.scene3d.mayavi_scene,
                                                  colormap=self.colourmap,
                                                  vmin=self.data_range[0], vmax=self.data_range[1])

            else:
                return mlab.pipeline.scalar_field(self.data,
                                                  figure=self.scene3d.mayavi_scene,
                                                  colormap=self.colourmap,
                                                  vmin=self.data_range[0], vmax=self.data_range[1])

        def make_plane_widget_3d(self, axis_name):
            plane_widget = mlab.pipeline.image_plane_widget(self.data_source,
                                                            figure=self.scene3d.mayavi_scene,
                                                            colormap=self.colourmap,
                                                            plane_orientation='%s_axes' % axis_name)

            return plane_widget

        def _plane_widget_3d_x_default(self):
            return self.make_plane_widget_3d('x')

        def _plane_widget_3d_y_default(self):
            return self.make_plane_widget_3d('y')

        def _plane_widget_3d_z_default(self):
            return self.make_plane_widget_3d('z')

        # Scene activation callbaks
        @on_trait_change('scene3d.activated')
        def display_scene3d(self):
            outline = mlab.pipeline.outline(self.data_source,
                                            figure=self.scene3d.mayavi_scene,
                                            colormap=self.colourmap,
                                            vmin=self.data_range[0], vmax=self.data_range[1])

            try:
                self.scene3d.mlab.view(40, 50)
            except AttributeError:
                return

            # Interaction properties can only be changed after the scene
            # has been created, and thus the interactor exists
            for plane_widget in (self.plane_widget_3d_x, self.plane_widget_3d_y, self.plane_widget_3d_z):
                # Turn the interaction off
                plane_widget.ipw.interaction = 0

            self.scene3d.scene.background = (0, 0, 0)

            # Keep the view always pointing up
            self.scene3d.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()

        def make_side_view(self, axis_name):
            scene = getattr(self, 'scene_%s' % axis_name)

            # To avoid copying the data, we take a reference to the
            # raw VTK dataset, and pass it on to mlab. Mlab will create
            # a Mayavi source from the VTK without copying it.
            # We have to specify the figure so that the data gets
            # added on the figure we are interested in.
            outline = mlab.pipeline.outline(self.data_source.mlab_source.dataset,
                                            figure=scene.mayavi_scene,
                                            colormap=self.colourmap,
                                            vmin=self.data_range[0], vmax=self.data_range[1])

            plane_widget = mlab.pipeline.image_plane_widget(outline,
                                                            plane_orientation='%s_axes' % axis_name,
                                                            colormap=self.colourmap,
                                                            vmin=self.data_range[0], vmax=self.data_range[1])
            setattr(self, 'plane_widget_%s' % axis_name, plane_widget)

            # Synchronize positions between the corresponding image plane
            # widgets on different views.
            plane_widget.ipw.sync_trait('slice_position', getattr(self, 'plane_widget_3d_%s' % axis_name).ipw)

            # Make left-clicking create a crosshair
            plane_widget.ipw.left_button_action = 0

            # Add a callback on the image plane widget interaction to
            # move the others
            def move_view(obj, evt):
                position = obj.GetCurrentCursorPosition()
                for other_axis, axis_number in self._axis_names.items():
                    if other_axis == axis_name:
                        continue
                    ipw3d = getattr(self, 'plane_widget_3d_%s' % other_axis)
                    ipw3d.ipw.slice_position = position[axis_number]

            plane_widget.ipw.add_observer('InteractionEvent', move_view)
            plane_widget.ipw.add_observer('StartInteractionEvent', move_view)

            # Center the image plane widget
            plane_widget.ipw.slice_position = 0.5*self.data.shape[
                        self._axis_names[axis_name]]

            # Position the view for the scene
            views = dict(x=(0, 90), y=(90, 90), z=(0, 0))
            scene.mlab.view(*views[axis_name])

            # 2D interaction: only pan and zoom
            scene.scene.interactor.interactor_style = tvtk.InteractorStyleImage()

            scene.scene.background = (0, 0, 0)

        @on_trait_change('scene_x.activated')
        def display_scene_x(self):
            return self.make_side_view('x')

        @on_trait_change('scene_y.activated')
        def display_scene_y(self):
            return self.make_side_view('y')

        @on_trait_change('scene_z.activated')
        def display_scene_z(self):
            return self.make_side_view('z')

        # The layout of the dialog created
        view = View(HGroup(Group(
                           Item('scene_y',
                                editor=SceneEditor(scene_class=Scene),
                                height=250, width=300),
                           Item('scene_z',
                                editor=SceneEditor(scene_class=Scene),
                                height=250, width=300),
                           show_labels=True), Group(
                           Item('scene_x',
                                editor=SceneEditor(scene_class=Scene),
                                height=250, width=300),
                           Item('scene3d',
                                editor=SceneEditor(scene_class=MayaviScene),
                                height=250, width=300),
                           show_labels=True)), resizable=True, title='VolumeSlicer')


__all__ = ['plot_scalar_field', 'plot_scalar_field_2d', 'plot_scalar_field_3d']


def prepare_plot_arguments(wrapped):
    @functools.wraps(wrapped)
    def _prepare_plot_arguments(field, data_range=(None, None), origin=None, limit=None,
                                axis=None, palette='viridis', title=None):

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
                       axis=axis, palette=palette, title=title)

    return _prepare_plot_arguments


@prepare_plot_arguments
def plot_scalar_field_2d(field, data_range=(None, None), origin=None, limit=None,
                         axis=None, palette='viridis', title=None):
    """
    Utility function to plot a 2D scalar field using matplotlib.

    Parameters
    ----------
    field : ScalarFunction or VectorFunction
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

    Returns
    -------
    Axis
        Generated axis.

    """
    if not ENABLED_2D_PLOTTING:
        return None

    if axis is None:
        figure, axis = plt.subplots(1, 1)

    im = axis.imshow(field.T,
                     cmap=palette,
                     vmin=data_range[0], vmax=data_range[1],
                     aspect='equal',
                     origin='lower',
                     extent=[origin[0], limit[0], origin[1], limit[1]],
                     interpolation='bicubic')

    if origin is None or limit is None:
        axis.set_xlabel('x')
        axis.set_ylabel('y')

    else:
        axis.set_xlabel('x (mm)')
        axis.set_ylabel('y (mm)')

    if title is not None:
        axis.set_title(title)

    plt.colorbar(im, ax=axis)

    return axis


@prepare_plot_arguments
def plot_scalar_field_3d(field, data_range=(None, None), origin=None, limit=None,
                         axis=None, palette='viridis', title=None):
    """
    Utility function to plot a 3D scalar field using MayaVi.

    Parameters
    ----------
    field : ScalarFunction or VectorFunction
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
    if not ENABLED_3D_PLOTTING:
        return None

    if axis is None:
        axis = MlabSceneModel()

    window = VolumeSlicer(data=field,
                          is_Vector=False,
                          colourmap=palette,
                          scene3d=axis,
                          data_range=data_range)

    return window


def plot_scalar_field(field, data_range=(None, None), origin=None, limit=None,
                      axis=None, palette='viridis', title=None):
    """
    Utility function to plot a scalar field using matplotib (2D) or MayaVi (3D).

    Parameters
    ----------
    field : ScalarFunction or VectorFunction
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
                                    axis=axis, palette=palette, title=title)

    else:
        axis = plot_scalar_field_2d(field,
                                    data_range=data_range, limit=limit, origin=origin,
                                    axis=axis, palette=palette, title=title)

    return axis
