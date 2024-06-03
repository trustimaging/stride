
import os
import numpy as np


def volume_slicer(*args, **kwargs):
    try:
        if not os.environ.get('DISPLAY', None):
            raise ModuleNotFoundError

        from traits.api import HasTraits, Instance, Array, on_trait_change
        from traitsui.api import View, Item, HGroup, Group

        from tvtk.api import tvtk
        from tvtk.util import ctf
        from tvtk.pyface.scene import Scene

        from mayavi import mlab
        from mayavi.core.api import PipelineBase, Source
        from mayavi.core.ui.api import SceneEditor, MayaviScene, MlabSceneModel
    except (ModuleNotFoundError, RuntimeError):
        class VolumeSlicer:
            pass

        return VolumeSlicer()

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

        def __init__(self, is_vector, data_range, colourmap, **traits):
            self.is_vector = is_vector
            self.data_range = data_range
            self.colourmap = colourmap

            super(VolumeSlicer, self).__init__(**traits)

            # Force the creation of the image_plane_widgets:
            self.plane_widget_3d_x
            self.plane_widget_3d_y
            self.plane_widget_3d_z

        # Default values
        def _data_source_default(self):
            colourmap = self.colourmap if self.colourmap != 'brain' else 'viridis'
            if self.is_vector:
                return mlab.pipeline.vector_field(self.data,
                                                  figure=self.scene3d.mayavi_scene,
                                                  colormap=colourmap,
                                                  vmin=self.data_range[0], vmax=self.data_range[1])

            else:
                return mlab.pipeline.scalar_field(self.data,
                                                  figure=self.scene3d.mayavi_scene,
                                                  colormap=colourmap,
                                                  vmin=self.data_range[0], vmax=self.data_range[1])

        def make_plane_widget_3d(self, axis_name):
            colourmap = self.colourmap if self.colourmap != 'brain' else 'viridis'
            plane_widget = mlab.pipeline.image_plane_widget(self.data_source,
                                                            figure=self.scene3d.mayavi_scene,
                                                            colormap=colourmap,
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
            colourmap = self.colourmap if self.colourmap != 'brain' else 'viridis'
            outline = mlab.pipeline.outline(self.data_source,
                                            figure=self.scene3d.mayavi_scene,
                                            colormap=colourmap,
                                            vmin=self.data_range[0], vmax=self.data_range[1])

            vmin = self.data_range[0] or self.data.min()
            vmax = self.data_range[1] or self.data.max()

            if vmin == vmax:
                vmax += 0.10 * vmin
                vmin -= 0.10 * vmin

            volume = mlab.pipeline.volume(self.data_source,
                                          figure=self.scene3d.mayavi_scene,
                                          vmin=vmin, vmax=vmax)

            otf = ctf.PiecewiseFunction()

            otf.add_point(vmin, 0.0)
            otf.add_point(vmax, 0.025)

            volume._otf = otf
            volume._volume_property.set_scalar_opacity(otf)

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
            colourmap = self.colourmap if self.colourmap != 'brain' else 'viridis'
            scene = getattr(self, 'scene_%s' % axis_name)

            # To avoid copying the data, we take a reference to the
            # raw VTK dataset, and pass it on to mlab. Mlab will create
            # a Mayavi source from the VTK without copying it.
            # We have to specify the figure so that the data gets
            # added on the figure we are interested in.
            outline = mlab.pipeline.outline(self.data_source.mlab_source.dataset,
                                            figure=scene.mayavi_scene,
                                            colormap=colourmap,
                                            vmin=self.data_range[0], vmax=self.data_range[1])

            plane_widget = mlab.pipeline.image_plane_widget(outline,
                                                            plane_orientation='%s_axes' % axis_name,
                                                            colormap=colourmap,
                                                            vmin=self.data_range[0], vmax=self.data_range[1])
            setattr(self, 'plane_widget_%s' % axis_name, plane_widget)

            # set colour map
            if self.colourmap == 'brain':
                try:
                    import stride_private
                    cmap_filename = os.path.join(os.path.dirname(stride_private.__file__),
                                                 'plotting/cmaps/' + self.colourmap + '.txt')
                    cmap_ = np.genfromtxt(cmap_filename, delimiter=',', dtype=np.float32)

                    lut = plane_widget.module_manager.scalar_lut_manager.lut
                    lut.number_of_colors = cmap_.shape[0]
                    lut.build()

                    for i, v in enumerate(np.linspace(0, 1, cmap_.shape[0])):
                        lut.set_table_value(i, cmap_[i, 0], cmap_[i, 1], cmap_[i, 2])
                except ImportError:
                    pass

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
            plane_widget.ipw.slice_position = 0.5 * self.data.shape[
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

    return VolumeSlicer(*args, **kwargs)
