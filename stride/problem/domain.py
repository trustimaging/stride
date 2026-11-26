
import warnings
import collections
import numpy as np
from cached_property import cached_property


__all__ = ['Space', 'MeshedSpace', 'Time', 'SlowTime', 'Grid']


CELL_TOPOLOGICAL_DIM = {
    'triangle': 2,
    'tetrahedron': 3,
}

DofIndex = collections.namedtuple('DofIndex', ('node', 'edge'))


def _row_keys(table, rows):
    """
    Reduce both tables to one sortable value per row.

    Packing the columns into a single integer lets numpy sort and search a flat array, which is
    several times faster than comparing rows through a structured view. It only works while the
    whole row fits in 63 bits, so the view is kept as the general case.

    Parameters
    ----------
    table : ndarray
        Rows to search, of shape ``(n, width)``, non-negative.
    rows : ndarray
        Rows to find, of shape ``(m, width)``, non-negative.

    Returns
    -------
    ndarray
        One key per row of ``table``.
    ndarray
        One key per row of ``rows``.

    """
    width = table.shape[1]
    largest = max(int(table.max()) if table.size else 0, int(rows.max()) if rows.size else 0)
    smallest = min(int(table.min()) if table.size else 0, int(rows.min()) if rows.size else 0)

    bits = max(largest.bit_length(), 1)

    if smallest >= 0 and width*bits <= 63:
        shifts = np.int64(bits)*np.arange(width - 1, -1, -1, dtype=np.int64)

        return (np.bitwise_or.reduce(table << shifts, axis=1),
                np.bitwise_or.reduce(rows << shifts, axis=1))

    as_rows = [('', np.int64)]*width

    return table.view(as_rows).ravel(), rows.view(as_rows).ravel()


def _match_rows(table, rows):
    """
    Find each of ``rows`` in ``table``, matching whole rows rather than single values.

    A mesh entity is identified by the nodes it is made of, which survives the renumbering that
    its own index does not. That makes matching entities a matter of looking up one integer row
    in another table of integer rows.

    A structured view lets numpy sort and search whole rows at once. The obvious alternative, a
    dict keyed on tuples, costs about 165 bytes an entry against 8, which for the edges of a
    mesh of a million nodes is the difference between a gigabyte and a few tens of megabytes.

    Parameters
    ----------
    table : ndarray
        Rows to search, of shape ``(n, width)``.
    rows : ndarray
        Rows to find, of shape ``(m, width)``.

    Returns
    -------
    ndarray
        Index into ``table`` for every row of ``rows``, of shape ``(m,)``.

    """
    table = np.ascontiguousarray(table, dtype=np.int64)
    rows = np.ascontiguousarray(rows, dtype=np.int64)

    if table.ndim != 2 or rows.ndim != 2 or table.shape[1] != rows.shape[1]:
        raise ValueError('Both tables must be 2D and the same width, got %s and %s'
                         % (table.shape, rows.shape))

    key, needle = _row_keys(table, rows)

    order = np.argsort(key)

    # a row that sorts past the end of the table would index out of bounds, and is a miss
    position = np.clip(np.searchsorted(key[order], needle), 0, max(key.size - 1, 0))
    found = order[position]

    missing = key[found] != needle

    if missing.any():
        raise KeyError('%d of %d rows are not in the table, the first being %s'
                       % (int(missing.sum()), missing.size, rows[missing][0]))

    return found


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
    facet_tags : dict, optional
        Boundary tags, as ``{'nodes': ndarray, 'values': ndarray}``. Every facet is named by the
        nodes it is made of, of shape ``(num_tagged, nodes_per_facet)``, rather than by an index
        DOLFINx assigns and does not preserve, and ``values`` carries one opaque integer per
        tagged facet. Naming a facet this way is what lets the tags be stored and rebuilt:
        ``from_dolfinx`` converts DOLFINx MeshTags into this form and ``to_dolfinx`` converts
        them back.
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

        if facet_tags is not None:
            try:
                facet_nodes = np.asarray(facet_tags['nodes'], dtype=np.int32)
                facet_values = np.asarray(facet_tags['values'])

            except (TypeError, KeyError, IndexError):
                raise ValueError('Facet tags must be a mapping with "nodes" and "values", got '
                                 '%s. DOLFINx MeshTags name a facet by an index that does not '
                                 'survive a rebuild, so pass them through from_dolfinx rather '
                                 'than directly' % type(facet_tags).__name__) from None

            if facet_nodes.ndim != 2:
                raise ValueError('Facet tag nodes must be a (num_tagged, nodes_per_facet) '
                                 'array, got shape %s' % (facet_nodes.shape,))

            if facet_values.shape != (facet_nodes.shape[0],):
                raise ValueError('There are %d tagged facets but %s values'
                                 % (facet_nodes.shape[0], facet_values.shape))

            if facet_nodes.size and (facet_nodes.min() < 0
                                     or facet_nodes.max() >= nodes.shape[0]):
                raise ValueError('Facet tags reference node indices outside [0, %d)'
                                 % nodes.shape[0])

            facet_tags = {'nodes': facet_nodes, 'values': facet_values}

        self.dim = dim
        self.nodes = nodes
        self.cells = cells
        self.cell_tags = cell_tags
        self.facet_tags = facet_tags
        self.cell_type = cell_type
        self.geometry_degree = geometry_degree

        # local only
        self._dolfinx_cache = None

    # pickle hooks
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_dolfinx_cache', None)

        # derived from the cells and large enough to matter -- for a mesh of a million nodes the
        # edge table is some 56 MB, which is not worth sending when a worker can rebuild it
        state.pop('edges', None)

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._dolfinx_cache = None

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

    @cached_property
    def edges(self):
        """
        Node pairs making up every edge of the mesh, of shape ``(num_edges, 2)``.

        A mesh generator does not write an edge table and DOLFINx does not preserve one, but
        neither has to: an edge *is* its pair of endpoints. Sorting each pair and then the table
        gives a numbering that this space, a worker rebuilding it and DOLFINx all derive
        identically from ``cells`` alone, with nothing agreed in advance and nothing shipped.

        This is the ordering a field on the edge dofs of a Lagrange element of degree 2 or more
        is indexed by, which is what lets such a field live in a MeshedField at all.

        Derived rather than stored, so it neither enters a file nor travels between workers.

        """
        if self.cells is None:
            raise ValueError('Cannot derive edges without cells, a node table is not a mesh')

        # on a curved cell the columns past the vertices hold the nodes added along the edges,
        # and pairing those would invent edges that do not exist
        vertices = self.cells[:, :CELL_TOPOLOGICAL_DIM[self.cell_type] + 1]

        pairs = [(first, second)
                 for first in range(vertices.shape[1])
                 for second in range(first + 1, vertices.shape[1])]

        edges = np.concatenate([vertices[:, [first, second]] for first, second in pairs])
        edges = np.sort(edges, axis=1).astype(np.int64)

        if self.num_nodes > 2**31:
            raise ValueError('Meshes of more than 2^31 nodes are not supported, this one has %d'
                             % self.num_nodes)

        # packing each pair into one integer lets np.unique sort a flat array rather than rows,
        # which is the same answer some nine times faster -- worth having, because every worker
        # derives this table for itself rather than being sent it
        keys = np.unique((edges[:, 0] << np.int64(32)) | edges[:, 1])

        edges = np.stack([keys >> np.int64(32), keys & np.int64(0xffffffff)], axis=1)

        # unique sorts, so the numbering is a function of the cell table and nothing else
        return edges.astype(self.cells.dtype, copy=False)

    @property
    def num_edges(self):
        """
        Number of edges in the mesh, zero if no connectivity is defined.

        """
        return 0 if self.cells is None else int(self.edges.shape[0])

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
    def from_dolfinx(cls, mesh, cell_tags=None, facet_tags=None, keep_mesh=False):
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
            Facet tags.

        Returns
        -------
        MeshedSpace
            Newly created MeshedSpace.

        """
        try:
            from dolfinx import mesh as dmesh

        except ImportError:
            raise ImportError('from_dolfinx needs dolfinx.mesh, which is an '
                                'optional dependencies of stride. Install them into the '
                                'environment, for instance with the fenics-dolfinx conda '
                                'package.') from None

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

        # Store facet tags using the geometry-node indices defining each facet,
        # rather than DOLFINx facet indices. DOLFINx may renumber facets when the
        # mesh is reconstructed.
        serialised_facet_tags = None
        if facet_tags is not None:

            facet_dim = topology_dim - 1

            mesh.topology.create_entities(facet_dim)
            mesh.topology.create_connectivity(facet_dim, topology_dim)

            indices = np.asarray(facet_tags.indices, dtype=np.int32)
            values = np.asarray(facet_tags.values, dtype=np.int32)

            facet_nodes = dmesh.entities_to_geometry(
                mesh, facet_dim, indices, False
            )

            # Node ordering/orientation within a facet is irrelevant when identifying it.
            facet_nodes = np.sort(np.asarray(facet_nodes, dtype=np.int32), axis=1)

            serialised_facet_tags = {
                'nodes': facet_nodes,
                'values': values,
            }

        space_cls = cls(nodes=nodes, cells=cells, cell_tags=tags,
                facet_tags=serialised_facet_tags,
                cell_type=mesh.topology.cell_name(),
                geometry_degree=mesh.geometry.cmap.degree)

        if keep_mesh:
            space_cls._dolfinx_cache = (mesh, cell_tags, facet_tags)

        return space_cls

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
        dolfinx.mesh.MeshTags or None
            Facet tags on the new facet ordering, or None if this space carries no facet tags.

        """
        try:
            import ufl
            import basix.ufl
            from dolfinx import mesh as dmesh
            from mpi4py import MPI

        except ImportError:
            raise ImportError('to_dolfinx needs dolfinx, basix, ufl and mpi4py, which are '
                              'optional dependencies of stride. Install them into the '
                              'environment, for instance with the fenics-dolfinx conda '
                              'package.') from None

        # check for local run
        cached = getattr(self, '_dolfinx_cache', None)

        if cached is not None:

            mesh, cell_tags, facet_tags = cached

            # Preserve the existing serial-only contract.
            if mesh.comm.size > 1:
                raise ValueError('to_dolfinx is serial only')

            if comm is not None:
                if comm.size > 1:
                    raise ValueError('to_dolfinx is serial only')

                # Reuse only when the explicitly requested communicator
                # is the same communicator as the cached mesh's.
                if MPI.Comm.Compare(mesh.comm, comm) == MPI.IDENT:
                    return cached
            else:
                return cached

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

        # recreate mesh
        mesh = dmesh.create_mesh(comm, cells=self.cells, x=self.nodes, e=element)

        num_cells = mesh.topology.index_map(topology_dim).size_local
        if num_cells != self.num_cells:
            raise RuntimeError('The rebuilt mesh has %d cells but this space has %d, so the '
                               'cell ordering cannot be recovered' % (num_cells, self.num_cells))

        # DOLFINx renumbers the nodes as it builds, and publishes the inverse of that
        # permutation. node_index checks it rather than taking it on trust
        node_index = self.node_index(mesh)

        cell_tags = None
        if self.cell_tags is not None:
            original_index = np.asarray(mesh.topology.original_cell_index)
            values = np.asarray(self.cell_tags)[original_index]

            tagged = values != -1

            cell_tags = dmesh.meshtags(mesh, topology_dim,
                                       np.arange(num_cells, dtype=np.int32)[tagged],
                                       values[tagged])

        facet_tags = None
        if self.facet_tags is not None:
            facet_dim = topology_dim - 1

            # Facet entities are not necessarily created yet, needs cell connectivity
            mesh.topology.create_entities(facet_dim)
            mesh.topology.create_connectivity(facet_dim, topology_dim)

            num_facets = mesh.topology.index_map(facet_dim).size_local
            facets = np.arange(num_facets, dtype=np.int32)

            # Find the geometry nodes belonging to every reconstructed facet.
            rebuilt_facet_nodes = dmesh.entities_to_geometry(
                mesh, facet_dim, facets, False
            )

            # these come back in the rebuilt node numbering, so they have to go through
            # node_index before they mean anything to this space. Sorting each row drops the
            # orientation, which identifies the same facet either way round
            rebuilt_facet_nodes = np.sort(node_index[np.asarray(rebuilt_facet_nodes)], axis=1)

            stored_facet_nodes = np.sort(np.asarray(self.facet_tags['nodes']), axis=1)
            stored_facet_values = np.asarray(self.facet_tags['values'], dtype=np.int32)

            try:
                rebuilt_indices = facets[_match_rows(rebuilt_facet_nodes, stored_facet_nodes)]

            except KeyError as error:
                raise RuntimeError('A tagged facet is not in the rebuilt mesh, so the facet '
                                   'tags cannot be recovered (%s)' % error) from None

            # meshtags expects sorted entity indices.
            order = np.argsort(rebuilt_indices)
            rebuilt_indices = rebuilt_indices[order]
            rebuilt_values = stored_facet_values[order]

            facet_tags = dmesh.meshtags(
                mesh,
                facet_dim,
                rebuilt_indices,
                rebuilt_values,
            )

        return mesh, cell_tags, facet_tags

    def node_index(self, mesh):
        """
        Permutation taking this space's node order into a DOLFINx mesh's.

        ``node_index[i]`` is the index this space gives the mesh's node ``i``, so
        ``self.nodes[node_index]`` is the mesh's node table and ``np.argsort(node_index)`` takes
        a value in this space's order into the mesh's.

        Which permutation that is depends on where the mesh came from, and the mesh cannot say.
        A mesh rebuilt by ``to_dolfinx`` was renumbered from this space's nodes, and
        ``input_global_indices`` is the inverse of that renumbering. A mesh this space was built
        *from* already holds the nodes in this space's order, and its ``input_global_indices``
        refers to whatever built it -- gmsh, say -- which is a different permutation entirely.
        Guessing wrong there is silent and total, so the candidate is checked against the
        coordinates rather than assumed.

        Parameters
        ----------
        mesh : dolfinx.mesh.Mesh
            Mesh to index against.

        Returns
        -------
        ndarray
            Permutation of shape ``(num_nodes,)``.

        """
        coordinates = np.asarray(mesh.geometry.x)[:, :self.dim]

        if coordinates.shape[0] != self.num_nodes:
            raise RuntimeError('The mesh has %d nodes and this space has %d, so no permutation '
                               'relates them' % (coordinates.shape[0], self.num_nodes))

        rebuilt = np.asarray(mesh.geometry.input_global_indices)
        identity = np.arange(self.num_nodes)

        for candidate in (rebuilt, identity):
            if candidate.shape == (self.num_nodes,) \
                    and np.allclose(self.nodes[candidate], coordinates):
                return candidate

        raise RuntimeError('Neither the node mapping DOLFINx reports nor the identity '
                           'reproduces the mesh node coordinates from this space, so the two '
                           'do not describe the same mesh')

    def dof_index(self, mesh, function_space):
        """
        Map this space's entity ordering onto the dofs of a function space over ``mesh``.

        A Lagrange dof is a point value, so no reconstruction is involved: the coefficient at a
        vertex dof is the potential at that vertex, and the one at an edge dof is the potential
        at that edge's midpoint. All that is missing is a name for each dof that both this space
        and DOLFINx agree on, and every arrow in the chain that provides one is an exact integer
        map -- the element reports which of its dofs sit on which entity, the topology reports
        which entity is which, and ``input_global_indices`` undoes the renumbering.

        This is what lets a field cross between a MeshedField and a Function without an
        interpolator. An interpolator would be approximate where this is exact, could not be
        shipped between workers any more easily, since it needs the mesh to exist first, and
        would turn a mismatched mesh into a plausible field rather than an error.

        Parameters
        ----------
        mesh : dolfinx.mesh.Mesh
            Mesh the function space is defined over, as returned by ``to_dolfinx``.
        function_space : dolfinx.fem.FunctionSpace
            Space whose dofs are to be indexed.

        Returns
        -------
        DofIndex
            ``node``, of shape ``(num_nodes,)``, holding the dof that carries the value at each
            of this space's nodes, and ``edge``, of shape ``(num_edges,)``, holding the dof at
            each edge midpoint. ``edge`` is None for an element with no dofs on edges.

        """
        topology_dim = CELL_TOPOLOGICAL_DIM[self.cell_type]
        num_cells = mesh.topology.index_map(topology_dim).size_local

        entity_dofs = function_space.ufl_element().basix_element.entity_dofs

        if any(len(dofs) != 1 for dofs in entity_dofs[0]):
            raise ValueError('Needs an element with exactly one dof per vertex, which is what a '
                             'Lagrange element of degree 1 or more has. A discontinuous element '
                             'has none, and its values belong on cells rather than nodes')

        if any(len(dofs) > 1 for dofs in entity_dofs[1]):
            raise NotImplementedError('Needs at most one dof per edge, which holds up to degree '
                                      '2. Above that an edge carries several and a MeshedField '
                                      'has no ordering for them')

        for dimension in range(2, topology_dim + 1):
            if any(len(dofs) for dofs in entity_dofs[dimension]):
                raise NotImplementedError('Elements with dofs on faces or cell interiors are '
                                          'not supported, only nodes and edges have a place in '
                                          'a MeshedField')

        # local vertex i of a cell is local geometry node i for a simplex, and for a curved cell
        # the nodes past the first topology_dim + 1 are the ones added along the edges
        node_index = self.node_index(mesh)
        geometry = np.asarray(mesh.geometry.dofmap).reshape(num_cells, -1)
        vertices = np.asarray(mesh.topology.connectivity(topology_dim, 0).array)
        vertices = vertices.reshape(num_cells, -1)

        node_of_vertex = np.empty(mesh.topology.index_map(0).size_local, dtype=np.int64)
        node_of_vertex[vertices.reshape(-1)] = \
            node_index[geometry[:, :topology_dim + 1].reshape(-1)]

        dofs = np.asarray(function_space.dofmap.list).reshape(num_cells, -1)

        node = np.empty(self.num_nodes, dtype=np.int64)

        for local_vertex, local_dofs in enumerate(entity_dofs[0]):
            node[node_of_vertex[vertices[:, local_vertex]]] = dofs[:, local_dofs[0]]

        if not any(len(local_dofs) for local_dofs in entity_dofs[1]):
            return DofIndex(node=node, edge=None)

        mesh.topology.create_entities(1)
        mesh.topology.create_connectivity(topology_dim, 1)

        # an edge is its pair of endpoints, which is the one name for it that this space and the
        # mesh derive identically. Sorting each pair drops the direction, which the two number
        # independently and neither needs
        edge_vertices = np.asarray(mesh.topology.connectivity(1, 0).array).reshape(-1, 2)
        edge_nodes = np.sort(node_of_vertex[edge_vertices], axis=1)

        edge_of_mesh_edge = _match_rows(self.edges, edge_nodes)

        cell_edges = np.asarray(mesh.topology.connectivity(topology_dim, 1).array)
        cell_edges = cell_edges.reshape(num_cells, -1)

        edge = np.empty(self.num_edges, dtype=np.int64)

        for local_edge, local_dofs in enumerate(entity_dofs[1]):
            edge[edge_of_mesh_edge[cell_edges[:, local_edge]]] = dofs[:, local_dofs[0]]

        return DofIndex(node=node, edge=edge)


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
