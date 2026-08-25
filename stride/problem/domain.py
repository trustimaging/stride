
import warnings
import numpy as np
from cached_property import cached_property


__all__ = ['Space', 'MeshedSpace', 'Time', 'SlowTime', 'Grid']

# Topological dimension of every supported cell type. Simplices only for now: adding
# quadrilateral or hexahedron here is most of what supporting them takes.
CELL_TOPOLOGICAL_DIM = {
    'triangle': 2,
    'tetrahedron': 3,
}

# Cell type implied by (nodes per cell, geometry degree). Within simplices this is unique, so a
# cell type that is not given can be inferred rather than demanded. It stops being unique as soon
# as non-simplices are supported -- six nodes at degree one is a prism, not a triangle -- which is
# why anything not in here raises instead of guessing.
CELL_TYPE_BY_NODES = {
    (3, 1): 'triangle',
    (6, 2): 'triangle',
    (4, 1): 'tetrahedron',
    (10, 2): 'tetrahedron',
}


class Space:
    """
    This defines the spatial grid over which the problem is defined.

    The spatial grid consists of an inner domain defined by ``shape`` and
    an external padding defined by ``extra``. Within this extra region, a
    further sub-region is defined as absorbing for boundary purposes as defined
    by ``absorbing``.

    The ``spacing`` defines the axis-wise spacing of the grid.

    Parameters
    ----------
    shape : tuple
        Shape of the inner domain.
    spacing : tuple or float
        Axis-wise spacing of the grid, in metres.
    extra : tuple
        Amount of axis-wise extra space around the inner domain.
    absorbing : tuple
        Portion of the extra space that corresponds to absorbing boundaries.

    """

    def __init__(self, shape=None, spacing=None, extra=None, absorbing=None):
        self.dim = None
        self.shape = None
        self.spacing = None
        self.extra = None
        self.absorbing = None

        self.origin = None
        self.pml_origin = None
        self.extended_shape = None
        self.limit = None
        self.extended_limit = None

        self._set_properties(shape=shape, spacing=spacing, extra=extra, absorbing=absorbing)

    def _set_properties(self, shape, spacing, extra, absorbing):
        if isinstance(spacing, float):
            spacing = (spacing,)*len(shape)

        extra = extra or (0,)*len(shape)
        absorbing = absorbing or (0,)*len(shape)

        self.dim = len(shape)
        self.shape = tuple(shape)
        self.spacing = tuple(spacing)
        self.extra = tuple(extra)
        self.absorbing = tuple(absorbing)

        origin = (0,) * self.dim
        pml_origin = tuple([each_origin - each_spacing * each_extra for each_origin, each_spacing, each_extra in
                            zip(origin, spacing, extra)])

        extended_shape = tuple(np.array([dim + 2*added for dim, added in zip(shape, extra)]))
        size = tuple(np.array(spacing) * (np.array(shape) - 1))
        extended_size = tuple([each_origin + each_spacing * (each_shape + each_extra - 1)
                               for each_origin, each_spacing, each_shape, each_extra in zip(origin, spacing, shape, extra)])

        self.origin = origin
        self.pml_origin = pml_origin
        self.extended_shape = extended_shape
        self.limit = size
        self.extended_limit = extended_size

    @property
    def size(self):
        """
        Alias for the domain limit.

        """
        return self.limit

    @property
    def extended_size(self):
        """
        Alias for the extended domain limit.

        """
        return self.extended_limit

    def resample(self, new_spacing, new_extra=None, new_absorbing=None):
        '''
        Method updates Space to the properties of the domain after resampling.

        Parameters
        ----------
        new_spacing: float or tuple(float)
            The new grid spacing.
        new_extra: tuple(int), optional
            The shape of the boundary for the new grid. Defaults to rescaling existing extra.
        new_absorbing: tuple(int), optional
            The shape of the absorbing boundary for the new grid. Defaults to rescaling
            existing absorbing.

        Returns
        -------

        '''

        if isinstance(new_spacing, float):
            new_spacing = (new_spacing,)*self.dim

        # NOTE you must be careful with numerical errors calculating new_shape, using:
        # new_shape = tuple((np.round(np.array(self.size) / np.array(new_spacing)) + 1).astype(int))
        # ... is not compatible with the method in scipy.ndimage.zoom
        old_spacing = self.spacing
        old_shape = self.shape
        resampling_factors = tuple([dx_old/dx_new
                for dx_old, dx_new in zip(old_spacing, new_spacing)])
        new_shape = tuple([int(round(n * m))
                for n, m in zip(old_shape, resampling_factors)])  # method matches scipy zoom

        if new_extra is None:
            new_extra = tuple((np.round(np.array(self.spacing) * (np.array(self.extra) - 1) /
                           np.array(new_spacing)) + 1).astype(int))

        if new_absorbing is None:
            new_absorbing = tuple((np.round(np.array(self.spacing) * (np.array(self.absorbing) - 1) /
                               np.array(new_spacing)) + 1).astype(int))

        self._set_properties(shape=new_shape, spacing=new_spacing, extra=new_extra, absorbing=new_absorbing)
        self._clear_cache('mesh_indices')
        self._clear_cache('extended_mesh_indices')
        self._clear_cache('mesh')
        self._clear_cache('extended_mesh')
        self._clear_cache('indices')
        self._clear_cache('extended_indices')
        self._clear_cache('grid')
        self._clear_cache('extended_grid')

    def _clear_cache(self, cached_property):
        '''
        Clear a cached property

        Parameters
        ----------
        cached_property: str
            The name of the property to remove from the cache.

        Returns
        -------
        '''
        try:
            del self.__dict__[cached_property]
        except:
            pass

    @property
    def inner(self):
        """
        Slices defining the inner domain, as a tuple of slices.

        """
        return tuple([slice(extra, extra + shape) for shape, extra in zip(self.shape, self.extra)])

    @property
    def inner_mask(self):
        """
        Tensor of the shape of the space grid with gridpoints wihtin inner domain set to 1
        and those outside set to 0, as an ndarray.

        """
        mask = np.zeros(self.extended_shape, dtype=np.float32)
        pml_slices = self.inner

        mask[pml_slices] = 1.

        return mask

    @cached_property
    def mesh_indices(self):
        """
        Create the mesh of indices in the inner domain, as a tuple
        of ndarray.

        """
        grid = [np.arange(0, shape) for shape in self.shape]
        return np.meshgrid(*grid)

    @cached_property
    def extended_mesh_indices(self):
        """
        Create the mesh of indices in the extended domain, as a tuple
        of ndarray.

        """
        grid = [np.arange(0, extended_shape) for extended_shape in self.extended_shape]
        return np.meshgrid(*grid)

    @cached_property
    def mesh(self):
        """
        Create the mesh of spatial locations in the inner domain, as a tuple
        of ndarray.

        """
        grid = self.grid
        return np.meshgrid(*grid, indexing='ij')

    @cached_property
    def extended_mesh(self):
        """
        Create the mesh of spatial locations the full, extended domain, as a tuple
        of ndarray.

        """
        grid = self.extended_grid
        return np.meshgrid(*grid, indexing='ij')

    @cached_property
    def indices(self):
        """
        Indices corresponding to the grid of the inner domain, as a tuple of 1d-arrays.

        """
        axes = [np.arange(0, shape) for shape in self.shape]
        return tuple(axes)

    @cached_property
    def extended_indices(self):
        """
        Indices corresponding to the grid of the extended domain, as a tuple of 1d-arrays.

        """
        axes = [np.arange(0, extended_shape) for extended_shape in self.extended_shape]
        return tuple(axes)

    @cached_property
    def grid(self):
        """
        Spatial points corresponding to the grid of the inner domain, as a tuple of 1d-arrays.

        """
        axes = [np.linspace(self.origin[dim], self.limit[dim], self.shape[dim],
                            endpoint=True, dtype=np.float32)
                for dim in range(self.dim)]
        return tuple(axes)

    @cached_property
    def extended_grid(self):
        """
        Spatial points corresponding to the grid of the extended domain, as a tuple of 1d-arrays.


        """
        axes = [np.linspace(self.pml_origin[dim], self.extended_limit[dim], self.extended_shape[dim],
                            endpoint=True, dtype=np.float32)
                for dim in range(self.dim)]
        return tuple(axes)


class MeshedSpace:
    """
    This defines an unstructured spatial mesh over which the problem is defined.

    Where a :class:`Space` is fully determined by a ``shape`` and a ``spacing``, a MeshedSpace is
    determined by an explicit table of ``nodes`` and, optionally, the ``cells`` that connect them.
    It is the counterpart used by finite-element physics, and is a sibling of Space rather than a
    subclass of it.

    A MeshedSpace deliberately has no ``shape``, ``extended_shape``, ``extra``, ``absorbing``,
    ``spacing``, ``inner`` or ``grid``. Those describe a regular grid and a mesh has no analogue of
    any of them, so code that requires a structured grid fails immediately when handed a mesh
    instead of silently producing a plausible but meaningless result.

    Note also that, unlike Space, ``size`` is not an alias for ``limit``: the mesh origin need not
    be at zero, so the extent and the upper bound differ.

    Parameters
    ----------
    nodes : ndarray
        Node coordinates, of shape ``(num_nodes, dim)``, in metres.
    cells : ndarray, optional
        Node indices making up each cell, of shape ``(num_cells, nodes_per_cell)``. The number of
        nodes per cell is not fixed: linear tetrahedra have 4, quadratic tetrahedra have 10.
    cell_tags : ndarray, optional
        Material tag for every cell, of shape ``(num_cells,)``. Tags are opaque integers as far as
        the MeshedSpace is concerned, and are only meaningful against the label table of whoever
        generated the mesh.
    facet_tags : optional
        Boundary tags, stored as given and not interpreted. These are not serialised, because a
        facet tag is meaningless without the facet connectivity, which is not stored either.
    cell_type : str, optional
        Cell type of the mesh, either ``triangle`` or ``tetrahedron``. Only simplices are
        supported. The spelling is the one DOLFINx and basix use, so that it can be handed to
        either without translation. This fixes the *topological* dimension, which need not equal
        ``dim``: a triangle mesh with three-dimensional nodes is a surface embedded in 3D, so a
        cell may live in a space of higher dimension than its own but never a lower one. If not
        given, it is inferred from the nodes per cell and the geometry degree, and a combination
        that is not unambiguous raises rather than being guessed at.
    geometry_degree : int, optional
        Degree of the coordinate element: 1 for straight-sided cells, 2 for curved ones.
        Defaults to 1. This describes the mesh geometry only, and is unrelated to the degree of
        any function space later defined over it.

    """

    def __init__(self, nodes=None, cells=None, cell_tags=None, facet_tags=None, cell_type=None, geometry_degree=1):
        nodes = np.asarray(nodes, dtype=np.float64)

        if nodes.ndim != 2:
            raise ValueError('Nodes must be a (num_nodes, dim) array, got shape %s'
                             % (nodes.shape,))

        dim = nodes.shape[1]
        if dim not in (2, 3):
            raise ValueError('Only 2 or 3 dimensions are supported, got %d' % dim)

        if cells is not None:
            cells = np.asarray(cells, dtype=np.int32)

            if cells.ndim != 2:
                raise ValueError('Cells must be a (num_cells, nodes_per_cell) array, got shape %s'
                                 % (cells.shape,))

            if cells.size and (cells.min() < 0 or cells.max() >= nodes.shape[0]):
                raise ValueError('Cells reference node indices outside [0, %d)' % nodes.shape[0])

            if cell_type is None:
                cell_type = CELL_TYPE_BY_NODES.get((cells.shape[1], geometry_degree))

                if cell_type is None:
                    raise ValueError('Cannot infer the cell type from %d nodes per cell at '
                                     'geometry degree %s. Pass cell_type explicitly.'
                                     % (cells.shape[1], geometry_degree))

        if cell_tags is not None:
            cell_tags = np.asarray(cell_tags)

            if cells is None:
                raise ValueError('Cell tags were given without any cells')

            if cell_tags.shape != (cells.shape[0],):
                raise ValueError('Cell tags must have one entry per cell, expected %d '
                                 'but got shape %s' % (cells.shape[0], (cell_tags.shape,)))

        if not isinstance(geometry_degree, int):
            raise TypeError('geometry_degree must be an int')
        if geometry_degree < 1:
            raise ValueError('geometry degree must be a positive int')

        if cell_type is not None:
            if not isinstance(cell_type, str):
                raise TypeError('cell_type must be str')
            elif cell_type not in CELL_TOPOLOGICAL_DIM:
                raise ValueError('Only simplex cells are supported (%s), got %r'
                                 % (', '.join(sorted(CELL_TOPOLOGICAL_DIM)), cell_type))

            # a cell can be embedded in a space of higher dimension than its own, but not lower:
            # a triangle mesh may be a surface in 3D, a tetrahedron cannot live in 2D
            if dim < CELL_TOPOLOGICAL_DIM[cell_type]:
                raise ValueError('A %s is %d-dimensional and cannot be embedded in %d dimensions'
                                 % (cell_type, CELL_TOPOLOGICAL_DIM[cell_type], dim))

        self.dim = dim
        self.nodes = nodes
        self.cells = cells
        self.cell_tags = cell_tags
        self.facet_tags = facet_tags
        self.cell_type = cell_type
        self.geometry_degree = geometry_degree

    @property
    def num_nodes(self):
        """
        Number of nodes in the mesh.

        """
        return int(self.nodes.shape[0])

    @property
    def num_cells(self):
        """
        Number of cells in the mesh, zero if no connectivity is defined.

        """
        return 0 if self.cells is None else int(self.cells.shape[0])

    @property
    def size(self):
        """
        Axis-wise extent of the mesh, as a tuple.

        """
        return tuple(each_limit - each_origin
                     for each_limit, each_origin in zip(self.limit, self.origin))

    @cached_property
    def origin(self):
        return tuple(self.nodes.min(axis=0))

    @cached_property
    def limit(self):
        return tuple(self.nodes.max(axis=0))

    def contains_box(self, lower, upper, atol=1e-9):
        """
        Whether the mesh covers an axis-aligned box.

        This is the check needed before evaluating a finite-element solution onto a structured
        grid: every point of the grid has to be owned by some cell of the mesh.

        Parameters
        ----------
        lower : tuple or ndarray
            Lower corner of the box, in metres.
        upper : tuple or ndarray
            Upper corner of the box, in metres.
        atol : float, optional
            Absolute tolerance, to absorb round-off in the node coordinates, defaults to 1e-9.

        Returns
        -------
        bool
            Whether the mesh covers the box.

        """
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)

        return bool((self.nodes.min(axis=0) <= lower + atol).all()
                    and (self.nodes.max(axis=0) >= upper - atol).all())

    def sample_labels(self, volume, affine=None):
        """
        Sample a voxelised label volume at the mesh nodes, using nearest-neighbour lookup.

        This is how a segmentation is transferred onto a mesh in order to build a medium. Nodes
        that fall outside the volume are clipped to the nearest in-range voxel rather than raising.

        Parameters
        ----------
        volume : ndarray
            Label volume, with as many dimensions as the mesh.
        affine : ndarray, optional
            Homogeneous ``(dim+1, dim+1)`` matrix mapping voxel indices to node coordinates. If
            not given, the node coordinates are taken to be voxel indices already.

        Returns
        -------
        ndarray
            Label at every node, of shape ``(num_nodes,)``.

        """
        volume = np.asarray(volume)

        if volume.ndim != self.dim:
            raise ValueError('Volume has %d dimensions but the mesh has %d'
                             % (volume.ndim, self.dim))

        if affine is None:
            indices = self.nodes
        else:
            affine = np.asarray(affine, dtype=np.float64)

            if affine.shape != (self.dim + 1, self.dim + 1):
                raise ValueError('Affine must have shape (%d, %d), got %s'
                                 % (self.dim + 1, self.dim + 1, (affine.shape,)))

            homogeneous = np.hstack([self.nodes, np.ones((self.num_nodes, 1))])
            indices = (np.linalg.inv(affine) @ homogeneous.T).T[:, :self.dim]

        indices = np.rint(indices).astype(np.int64)
        for axis in range(self.dim):
            indices[:, axis] = np.clip(indices[:, axis], 0, volume.shape[axis] - 1)

        # rint before astype: label volumes read from NIfTI are floats, and truncating turns a
        # stored 2.9999 into 2
        return np.rint(volume[tuple(indices.T)]).astype(np.int64)

    def resample(self, *args, **kwargs):
        """
        Not available for a MeshedSpace.

        A mesh has no spacing to resample onto, and generating a new mesh is a separate operation
        that must not be silently approximated here.

        Returns
        -------

        """
        raise NotImplementedError('A MeshedSpace cannot be resampled, it has no spacing. '
                                  'Generate a new mesh instead.')

    @classmethod
    def from_dolfinx(cls, mesh, cell_tags=None, facet_tags=None):
        """
        Create a MeshedSpace from an in-memory DOLFINx mesh.

        This is only correct for a serial mesh. Under MPI, DOLFINx node coordinates and the cell
        dofmap are rank-local and include ghost entities, so the resulting MeshedSpace describes
        one partition rather than the whole mesh: node and cell counts under-report, ``origin`` and
        ``limit`` bound a sub-box, and nodes shared between ranks appear more than once. A warning
        is issued in that case rather than an error, so that experimentation is still possible.

        Parameters
        ----------
        mesh : dolfinx.mesh.Mesh
            Mesh to adapt.
        cell_tags : dolfinx.mesh.MeshTags, optional
            Cell tags, which are sparse and get densified to one entry per cell. Cells that carry
            no tag are filled with -1.
        facet_tags : optional
            Facet tags, stored as given.

        Returns
        -------
        MeshedSpace
            Newly created MeshedSpace.

        """
        if mesh.comm.size > 1:
            warnings.warn('MeshedSpace.from_dolfinx is building from rank-local arrays, so the '
                          'resulting space describes this rank\'s partition and not the whole '
                          'mesh. Build the mesh on MPI.COMM_SELF, or gather it, to avoid this.')

        # geometry.x is always padded to three columns, whatever the geometric dimension
        dim = mesh.geometry.dim
        nodes = np.asarray(mesh.geometry.x)[:, :dim]

        topology_dim = mesh.topology.dim
        num_cells = mesh.topology.index_map(topology_dim).size_local

        # the geometry dofmap, not the topology connectivity, is what indexes geometry.x
        dofmap = mesh.geometry.dofmap

        if hasattr(dofmap, 'offsets'):
            # DOLFINx <= 0.7 exposes a flat array plus offsets. Cells of a single mesh are all
            # of the same type, so the first offset step gives the nodes per cell
            offsets = np.asarray(dofmap.offsets)
            nodes_per_cell = int(offsets[1] - offsets[0])
            cells = np.asarray(dofmap.array).reshape(-1, nodes_per_cell)

        else:
            cells = np.asarray(dofmap).reshape(num_cells, -1)

        cells = cells[:num_cells]

        tags = None
        if cell_tags is not None:
            # MeshTags are a sparse (indices, values) pair, so densify to one entry per cell
            tags = np.full(num_cells, -1, dtype=np.int32)
            indices = np.asarray(cell_tags.indices)
            values = np.asarray(cell_tags.values)

            owned = indices < num_cells
            tags[indices[owned]] = values[owned]

        return cls(nodes=nodes, cells=cells, cell_tags=tags, facet_tags=facet_tags,
                   cell_type=mesh.topology.cell_name(),
                   geometry_degree=mesh.geometry.cmap.degree)

    def to_dolfinx(self, comm=None):
        """
        Create an in-memory DOLFINx mesh from this MeshedSpace.

        This is the inverse of :meth:`from_dolfinx`, and is only correct in serial. The node and
        cell tables stored here describe a whole mesh, so building on a communicator of more
        than one rank would partition it into something this space does not describe. Unlike
        ``from_dolfinx``, which warns, that case raises.

        DOLFINx is an optional dependency of stride, so it is imported here rather than at
        module level.

        Parameters
        ----------
        comm : MPI.Intracomm, optional
            Communicator on which to build the mesh, defaults to ``MPI.COMM_SELF``.

        Returns
        -------
        dolfinx.mesh.Mesh
            Newly created mesh.
        dolfinx.mesh.MeshTags or None
            Cell tags on the new cell ordering, or None if this space carries no tags. Cells
            marked -1, which is what ``from_dolfinx`` fills in for untagged cells, are left out
            rather than handed back as a label of -1.

        """
        try:
            import ufl
            import basix.ufl
            import dolfinx
            from mpi4py import MPI

        except ImportError:
            raise ImportError('to_dolfinx needs dolfinx, basix, ufl and mpi4py, which are '
                              'optional dependencies of stride. Install them into the '
                              'environment, for instance with the fenics-dolfinx conda '
                              'package.') from None

        if comm is None:
            comm = MPI.COMM_SELF

        if self.cells is None:
            raise ValueError('Cannot build a mesh without cells, a node table is not a mesh')

        if comm.size > 1:
            raise ValueError('to_dolfinx is serial only, got a communicator of size %d. Build '
                             'on MPI.COMM_SELF and partition afterwards if needed.' % comm.size)

        topology_dim = CELL_TOPOLOGICAL_DIM[self.cell_type]

        element = ufl.Mesh(basix.ufl.element('Lagrange', self.cell_type, self.geometry_degree,
                                             shape=(self.dim,)))

        mesh = dolfinx.mesh.create_mesh(comm, self.cells, element, self.nodes)

        num_cells = mesh.topology.index_map(topology_dim).size_local
        if num_cells != self.num_cells:
            raise RuntimeError('The rebuilt mesh has %d cells but this space has %d, so the '
                               'cell ordering cannot be recovered' % (num_cells, self.num_cells))

        cell_tags = None
        if self.cell_tags is not None:
            original_index = np.asarray(mesh.topology.original_cell_index)
            values = np.asarray(self.cell_tags)[original_index]

            tagged = values != -1

            cell_tags = dolfinx.mesh.meshtags(mesh, topology_dim,
                                              np.arange(num_cells, dtype=np.int32)[tagged],
                                              values[tagged])

        return mesh, cell_tags


class Time:
    """
    This defines the temporal grid over which the problem is defined

    A time grid is fully defined by three of its arguments: start, stop, step or num.

    The time grid can be extended with a certain amount of padding, generating an
    inner domain and an extended domain, similar to that seen in the Space.

    Parameters
    ----------
    start : float, optional
        Point at which time starts, in seconds.
    step : float, optional
        Step between time points, in seconds.
    num : int, optional
        Number of time points in the grid.
    stop : float, optional
        Point at which time ends, in seconds.

    """

    def __init__(self, start=None, step=None, num=None, stop=None):
        self.start = None
        self.stop = None
        self.step = None
        self.num = None

        self.extra = None
        self.extended_start = None
        self.extended_stop = None
        self.extended_num = None

        self._set_properties(start, step, num, stop)

    def _set_properties(self, start=None, step=None, num=None, stop=None):
        try:
            if start is None:
                start = stop - step*(num - 1)
            elif step is None:
                step = (stop - start)/(num - 1)
            elif num is None:
                num = int(np.ceil((stop - start)/step + 1))
                stop = step*(num - 1) + start
            elif stop is None:
                stop = start + step*(num - 1)

        except:
            raise ValueError('Three of args start, step, num and stop may be set')

        if not isinstance(num, int):
            raise TypeError('"input" argument must be of type int')

        self.start = start
        self.stop = stop
        self.step = step
        self.num = num

        self.extra = 0
        self.extended_start = start
        self.extended_stop = stop
        self.extended_num = num

    def extend(self, extra):
        self.extra = extra
        self.extended_start = self.start - (self.extra[0] - 1)*self.step
        self.extended_stop = self.stop + (self.extra[1] - 1)*self.step
        self.extended_num = self.num + self.extra[0] + self.extra[1]

    def resample(self, new_step, new_num):
        """
        Resample a trace.

        Parameters
        ----------
        new_step : float
            The time spacing for the interpolated grid
        new_num : int
            The number of time-points, default is calculated to match input pulse
            length in [s]

        Returns
        -------
        """

        dt_in = self.step  # Extract current parameters
        start = self.start
        stop = self.stop
        num = self.num

        new_start = 0.  # Calculate new parameters

        interp_num = int((num)*(dt_in/new_step))
        interp_stop = new_start + new_step*(interp_num - 1)

        if new_num is not None:  # Do we need to pad the array or not?
            new_stop = new_start + new_step*(new_num - 1)
        else:
            new_num = interp_num
            new_stop = interp_stop

        self._set_properties(start=new_start, step=new_step, num=new_num)  # Update time

        self._clear_cache('mesh_indices')
        self._clear_cache('extended_mesh_indices')
        self._clear_cache('mesh')
        self._clear_cache('extended_mesh')
        self._clear_cache('indices')
        self._clear_cache('extended_indices')
        self._clear_cache('grid')
        self._clear_cache('extended_grid')

    def _clear_cache(self, cached_property):
        '''
        Clear a cached property

        Parameters
        ----------
        cached_property: str
            The name of the property to remove from the cache.

        Returns
        -------
        '''
        try:
            del self.__dict__[cached_property]
        except:
            pass

    @property
    def inner(self):
        """
        Slice defining the inner domain.

        """
        return slice(self.extra, self.extra + self.num)

    @cached_property
    def grid(self):
        """
        Time points corresponding to the grid of the inner domain, as a 1d-array.

        """
        return np.linspace(self.start, self.stop, self.num, endpoint=True, dtype=np.float32)

    @cached_property
    def extended_grid(self):
        """
        Time points corresponding to the grid of the extended domain, as a 1d-array.

        """
        return np.linspace(self.extended_start, self.extended_stop, self.extended_num, endpoint=True, dtype=np.float32)


class SlowTime:
    """
    This defines the slow temporal grid over which the problem is defined

    Parameters
    ----------
    frame_rate : float, optional
        Sampling frequency between frames, in Hz.
    acq_rate : float, optional
        Sampling frequency between acquisitions, in Hz.
    frame_step : float, optional
        Time step between frames, in seconds.
    acq_step : float, optional
        Time step between frames, in seconds.
    num_frame : int, optional
        Number of frames in the grid.
    num_acq : int, optional
        Number of acquisitions per frame.

    """

    def __init__(self, frame_rate=None, acq_rate=None,
                 frame_step=None, acq_step=None,
                 num_frame=None, num_acq=None):
        try:
            if frame_step is None:
                frame_step = 1/frame_rate
            else:
                frame_rate = 1/frame_step

        except:
            raise ValueError('Either freq or step has to be defined')

        if not isinstance(num_frame, int):
            raise TypeError('num_frames must be of type int')

        if acq_step is None and acq_rate is None:
            acq_step = 0
            acq_rate = -1
            num_acq = 1
        else:
            if not isinstance(num_acq, int):
                raise TypeError('num_acq must be of type int')

        if acq_step is None:
            acq_step = 1/acq_rate
        elif acq_rate is None:
            acq_rate = 1/acq_step

        if num_acq*acq_step > frame_step:
            raise ValueError('Acquisition step (%e s) too large for frame step (%e s).'
                             % (num_acq*acq_step, frame_step))

        start = 0.
        stop = start + frame_step * (num_frame - 1)

        self.start = start
        self.stop = stop
        self.frame_step = frame_step
        self.frame_rate = frame_rate
        self.num_frame = num_frame
        self.acq_step = acq_step
        self.acq_rate = acq_rate
        self.num_acq = num_acq

    def resample(self):
        raise NotImplementedError('Resampling has not been implemented yet')

    @property
    def num(self):
        """
        Total number of steps.

        """
        return self.num_frame*self.num_acq

    @property
    def extended_num(self):
        """
        Total number of steps.

        """
        return self.num

    @property
    def inner(self):
        """
        Slice defining the inner domain.

        """
        return slice(0, None)

    @cached_property
    def grid(self):
        """
        Time points corresponding to the grid, as a 1d-array.

        """
        if self.acq_rate > 0:
            start = 0.
            stop = start + self.acq_step * (self.num_acq - 1)

            grid = [np.linspace(start + self.frame_step*acq, stop + self.frame_step*acq,
                                self.num_acq, endpoint=True, dtype=np.float32)
                    for acq in range(self.num_frame)]

            return np.concatenate(grid)
        else:
            return np.linspace(self.start, self.stop, self.num_frame, endpoint=True, dtype=np.float32)


class Grid:
    """
    The grid is a container for the spatial and temporal grids.

    Parameters
    ----------
    space : Space
    time : Time
    slow_time : SlowTime
    """

    def __init__(self, space=None, time=None, slow_time=None):
        self.space = space
        self.time = time
        self.slow_time = slow_time
