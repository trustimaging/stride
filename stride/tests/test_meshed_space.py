"""
Tests for MeshedSpace (stride/problem/domain.py).

MeshedSpace is the unstructured counterpart to Space: instead of a shape and a
spacing it is defined by an explicit node list and cell connectivity.

Contract under test:

- ``MeshedSpace(nodes, cells=None, cell_tags=None, facet_tags=None)``
- ``dim`` is inferred from ``nodes.shape[1]`` and must be 2 or 3
- ``num_nodes`` / ``num_cells``
- ``origin`` / ``limit`` / ``size`` come from the node bounding box, the
  analogue of Space's origin/limit/size
- ``shape`` is ``(num_nodes,)`` -- the shape of a scalar node field -- with
  ``extended_shape == shape``, ``extra`` and ``absorbing`` all-zero and
  ``inner == (slice(0, None),)``, so that the StructuredData and GriddedSaved
  machinery that reads those attributes keeps working
- ``contains_box(lower, upper)`` is the mesh-covers-the-grid check performed
  when a mesh is attached to a problem grid
- ``resample`` raises, because a mesh has no spacing to resample onto
- ``cell_type`` / ``geometry_degree`` describe the discretisation, are inferred
  from the nodes per cell when not given, and fix the topological dimension,
  which may be lower than ``dim`` for a surface mesh
- ``to_dolfinx`` is the inverse of ``from_dolfinx``: it rebuilds a mesh, maps
  the cell tags onto whatever ordering ``create_mesh`` chose, and re-sparsifies
  the ``-1`` fill that ``from_dolfinx`` introduced
- ``from_dolfinx`` adapts an in-memory DOLFINx mesh (skipped without DOLFINx)
"""

import numpy as np
import pytest

from .conftest import box_tetra_mesh

from stride.problem.domain import MeshedSpace
from stride.problem.data import MeshedField, ScalarField

try:
    import dolfinx  # noqa: F401
    from mpi4py import MPI
    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False


class TestMeshedSpaceConstruction:

    def test_nodes_and_cells_are_stored(self, tetra_mesh, meshed_space):
        nodes, cells = tetra_mesh

        np.testing.assert_array_equal(meshed_space.nodes, nodes)
        np.testing.assert_array_equal(meshed_space.cells, cells)

    def test_dim_inferred_from_nodes(self, meshed_space, meshed_space_2d):
        assert meshed_space.dim == 3
        assert meshed_space_2d.dim == 2

    def test_counts(self, meshed_space):
        assert meshed_space.num_nodes == 27
        assert meshed_space.num_cells == 48

    def test_num_cells_is_zero_without_connectivity(self, tetra_mesh):
        nodes, _ = tetra_mesh
        space = MeshedSpace(nodes=nodes)

        assert space.cells is None
        assert space.num_cells == 0
        assert space.num_nodes == 27

    def test_nodes_stored_as_float64(self, tetra_mesh):
        nodes, cells = tetra_mesh
        space = MeshedSpace(nodes=nodes.astype(np.float32), cells=cells)

        # DOLFINx geometry is double precision; the node table must not be
        # silently downcast, or the mesh-vs-grid bounds checks lose precision.
        assert space.nodes.dtype == np.float64

    def test_cell_tags_are_stored(self, tagged_meshed_space):
        tags = tagged_meshed_space.cell_tags

        assert tags is not None
        assert tags.shape == (tagged_meshed_space.num_cells,)
        assert set(np.unique(tags)) == {1, 2}

    def test_cell_tags_default_to_none(self, meshed_space):
        assert meshed_space.cell_tags is None
        assert meshed_space.facet_tags is None

    def test_ragged_nodes_rejected(self):
        with pytest.raises(ValueError):
            MeshedSpace(nodes=np.zeros(10))

    def test_unsupported_dimensionality_rejected(self):
        # Only 2D and 3D meshes are supported.
        with pytest.raises(ValueError):
            MeshedSpace(nodes=np.zeros((10, 4)))

    def test_out_of_range_cell_indices_rejected(self, tetra_mesh):
        nodes, cells = tetra_mesh
        broken = cells.copy()
        broken[0, 0] = len(nodes)

        with pytest.raises(ValueError):
            MeshedSpace(nodes=nodes, cells=broken)

    def test_cell_tags_length_must_match_cells(self, tetra_mesh):
        nodes, cells = tetra_mesh

        with pytest.raises(ValueError):
            MeshedSpace(nodes=nodes, cells=cells, cell_tags=np.ones(3, dtype=np.int32))


class TestMeshedSpaceGeometry:

    def test_origin_and_limit_from_node_bounds(self, meshed_space):
        np.testing.assert_allclose(meshed_space.origin, (0., 0., 0.))
        np.testing.assert_allclose(meshed_space.limit, (2e-3, 2e-3, 2e-3))

    def test_size_is_the_extent(self, meshed_space):
        np.testing.assert_allclose(meshed_space.size, (2e-3, 2e-3, 2e-3))

    def test_offset_origin_is_respected(self):
        nodes, cells = box_tetra_mesh(shape=(3, 3, 3), spacing=(1e-3, 1e-3, 1e-3),
                                      origin=(-5e-3, 1e-3, 0.))
        space = MeshedSpace(nodes=nodes, cells=cells)

        np.testing.assert_allclose(space.origin, (-5e-3, 1e-3, 0.))
        np.testing.assert_allclose(space.limit, (-3e-3, 3e-3, 2e-3))
        np.testing.assert_allclose(space.size, (2e-3, 2e-3, 2e-3))

    def test_geometry_is_per_axis(self, meshed_space_2d):
        assert len(meshed_space_2d.origin) == 2
        assert len(meshed_space_2d.limit) == 2
        assert len(meshed_space_2d.size) == 2


class TestMeshedSpaceIsNotAGrid:
    """
    A MeshedSpace must not masquerade as a structured grid.

    An earlier version of this class asserted the opposite: that MeshedSpace exposed
    ``shape``/``extended_shape``/``extra``/``absorbing``/``inner`` as a compatibility surface. That
    was wrong. Nothing meshed reads them — MeshedData derives its shape from ``num_nodes``, the way
    SparseField uses ``num`` — and their only effect was to let a structured field accept a mesh and
    then produce plausible nonsense when plotted or resampled.
    """

    @pytest.mark.parametrize('attribute', ['shape', 'extended_shape', 'extra',
                                           'absorbing', 'inner', 'spacing', 'grid'])
    def test_has_no_grid_attributes(self, meshed_space, attribute):
        assert not hasattr(meshed_space, attribute)

    def test_structured_field_rejects_a_mesh(self, meshed_space):

        # Fails on the first grid attribute it reaches for, rather than silently constructing.
        with pytest.raises(AttributeError):
            ScalarField(name='alpha', space=meshed_space)

    def test_meshed_field_rejects_a_structured_space(self, structured_space):
        # The reverse direction needs an explicit check: MeshedData sizes itself behind an
        # isinstance test, which would otherwise skip and leave a field with no shape that
        # still allocates and fills without complaint.
        with pytest.raises(ValueError, match='MeshedSpace'):
            MeshedField(name='alpha', space=structured_space)

    def test_no_space_is_still_allowed(self):
        # This is what an instance about to be loaded from file looks like.
        field = MeshedField(name='alpha')

        assert field.space is None


class TestMeshedSpaceBounds:
    """The mesh-covers-the-grid assertions made when attaching a mesh to a grid."""

    def test_contains_its_own_bounds(self, meshed_space):
        assert meshed_space.contains_box(meshed_space.origin, meshed_space.limit)

    def test_contains_an_inset_box(self, meshed_space):
        assert meshed_space.contains_box((5e-4, 5e-4, 5e-4), (15e-4, 15e-4, 15e-4))

    def test_rejects_a_box_that_overhangs(self, meshed_space):
        assert not meshed_space.contains_box((0., 0., 0.), (3e-3, 2e-3, 2e-3))
        assert not meshed_space.contains_box((-1e-3, 0., 0.), (2e-3, 2e-3, 2e-3))

    def test_tolerance_absorbs_round_off(self, meshed_space):
        # A box that overhangs by less than atol counts as covered, which is
        # what makes the check survive float round-off on gmsh node coordinates.
        upper = tuple(each + 1e-12 for each in meshed_space.limit)

        assert meshed_space.contains_box(meshed_space.origin, upper, atol=1e-9)
        assert not meshed_space.contains_box(meshed_space.origin, upper, atol=0.)


class TestMeshedSpaceResample:

    def test_resample_is_not_supported(self, meshed_space):
        # Unlike Space, a mesh has no spacing to resample onto; remeshing is a
        # separate operation and must not be silently approximated here.
        with pytest.raises(NotImplementedError):
            meshed_space.resample(5e-4)


@pytest.mark.skipif(not HAS_DOLFINX, reason='DOLFINx not available')
class TestMeshedSpaceFromDolfinx:

    def _dolfinx_box(self):

        return dolfinx.mesh.create_box(
            MPI.COMM_WORLD,
            [np.array([0., 0., 0.]), np.array([2e-3, 2e-3, 2e-3])],
            [2, 2, 2],
        )

    def test_nodes_come_from_mesh_geometry(self):
        mesh = self._dolfinx_box()
        space = MeshedSpace.from_dolfinx(mesh)

        np.testing.assert_allclose(space.nodes, mesh.geometry.x[:, :3])
        assert space.dim == 3
        assert space.num_nodes == mesh.geometry.x.shape[0]

    def test_bounds_match_the_dolfinx_mesh(self):
        mesh = self._dolfinx_box()
        space = MeshedSpace.from_dolfinx(mesh)

        np.testing.assert_allclose(space.origin, mesh.geometry.x.min(axis=0))
        np.testing.assert_allclose(space.limit, mesh.geometry.x.max(axis=0))

    def test_cells_come_from_topology(self):
        mesh = self._dolfinx_box()
        space = MeshedSpace.from_dolfinx(mesh)

        tdim = mesh.topology.dim
        num_cells = mesh.topology.index_map(tdim).size_local

        assert space.num_cells == num_cells
        assert space.cells.shape[1] == 4
        assert space.cells.max() < space.num_nodes


class TestMeshedSpaceCellType:
    """cell_type / geometry_degree, and the inference that fills them in."""

    def test_triangle_inferred_from_three_nodes(self, meshed_space_2d):
        assert meshed_space_2d.cell_type == 'triangle'
        assert meshed_space_2d.geometry_degree == 1

    def test_tetrahedron_inferred_from_four_nodes(self, meshed_space):
        assert meshed_space.cell_type == 'tetrahedron'
        assert meshed_space.geometry_degree == 1

    def test_explicit_cell_type_is_kept(self, tri_mesh):
        nodes, cells = tri_mesh
        space = MeshedSpace(nodes=nodes, cells=cells, cell_type='triangle')

        assert space.cell_type == 'triangle'

    def test_cell_type_is_none_without_cells(self, tri_mesh):
        nodes, _ = tri_mesh

        assert MeshedSpace(nodes=nodes).cell_type is None

    def test_unsupported_cell_type_names_the_limit(self, tri_mesh):
        nodes, cells = tri_mesh

        with pytest.raises(ValueError, match='simplex'):
            MeshedSpace(nodes=nodes, cells=cells, cell_type='hexahedron')

    def test_uninferrable_cell_asks_for_an_explicit_type(self, tri_mesh):
        nodes, _ = tri_mesh
        # five nodes per cell is not a simplex at any supported degree
        cells = np.zeros((2, 5), dtype=np.int32)

        with pytest.raises(ValueError, match='infer'):
            MeshedSpace(nodes=nodes, cells=cells)

    def test_a_triangle_may_be_a_surface_in_three_dimensions(self):
        nodes = np.array([[0., 0., 0.], [1e-3, 0., 0.],
                          [0., 1e-3, 1e-3], [1e-3, 1e-3, 1e-3]])
        cells = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        space = MeshedSpace(nodes=nodes, cells=cells, cell_type='triangle')

        assert space.cell_type == 'triangle'
        assert space.dim == 3

    def test_a_tetrahedron_cannot_live_in_two_dimensions(self, tri_mesh):
        nodes, _ = tri_mesh
        cells = np.zeros((2, 4), dtype=np.int32)

        with pytest.raises(ValueError, match='cannot be embedded'):
            MeshedSpace(nodes=nodes, cells=cells, cell_type='tetrahedron')

    def test_degree_must_be_a_positive_int(self, tri_mesh):
        nodes, cells = tri_mesh

        with pytest.raises(ValueError):
            MeshedSpace(nodes=nodes, cells=cells, cell_type='triangle', geometry_degree=0)

        with pytest.raises(TypeError):
            MeshedSpace(nodes=nodes, cells=cells, cell_type='triangle', geometry_degree=1.0)


@pytest.mark.skipif(not HAS_DOLFINX, reason='DOLFINx not available')
class TestMeshedSpaceToDolfinx:
    """
    to_dolfinx has to reproduce a mesh that a solver can use, which is a stronger
    claim than the individual arrays matching. The checks below are chosen so that
    none of them can be satisfied by a plausible-but-wrong reconstruction.
    """

    def _tagged_square(self, n=4):
        """A unit-square mesh tagged 1 left of x=0.5 and 2 right of it."""

        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_SELF, n, n)
        tdim = mesh.topology.dim
        cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
        midpoints = dolfinx.mesh.compute_midpoints(mesh, tdim, cells)
        tags = self._predicate(midpoints).astype(np.int32)

        return mesh, dolfinx.mesh.meshtags(mesh, tdim, cells, tags)

    @staticmethod
    def _predicate(midpoints):
        return np.where(midpoints[:, 0] < 0.5, 1, 2)

    @staticmethod
    def _total_volume(mesh):
        """Sum of triangle areas, straight from the geometry."""
        coordinates = mesh.geometry.x
        dofmap = mesh.geometry.dofmap
        num_cells = mesh.topology.index_map(mesh.topology.dim).size_local

        total = 0.
        for cell in range(num_cells):
            points = coordinates[dofmap[cell]][:, :2]
            first, second = points[1] - points[0], points[2] - points[0]
            total += 0.5*abs(first[0]*second[1] - first[1]*second[0])

        return total

    def test_rebuilds_a_usable_mesh(self):
        mesh, tags = self._tagged_square()
        space = MeshedSpace.from_dolfinx(mesh, cell_tags=tags)

        rebuilt, _ = space.to_dolfinx()

        assert rebuilt.topology.dim == 2
        assert rebuilt.geometry.dim == 2
        assert rebuilt.topology.index_map(2).size_local == space.num_cells

    def test_node_coordinates_survive_as_a_set(self):
        mesh, tags = self._tagged_square()
        space = MeshedSpace.from_dolfinx(mesh, cell_tags=tags)

        rebuilt, _ = space.to_dolfinx()

        # ordering need not survive, the set of positions must
        before = set(map(tuple, np.round(space.nodes, 12)))
        after = set(map(tuple, np.round(rebuilt.geometry.x[:, :2], 12)))

        assert before == after
        assert len(before) == space.num_nodes

    def test_total_volume_is_preserved(self):
        """Catches scrambled connectivity, which per-node coordinate checks do not."""
        mesh, tags = self._tagged_square()
        space = MeshedSpace.from_dolfinx(mesh, cell_tags=tags)

        rebuilt, _ = space.to_dolfinx()

        np.testing.assert_allclose(self._total_volume(rebuilt), self._total_volume(mesh))
        np.testing.assert_allclose(self._total_volume(rebuilt), 1.)

    def test_tags_land_on_the_same_cells(self):
        """
        The load-bearing test. create_mesh may reorder cells, so the tags have to be
        permuted. Rather than trusting the permutation, this re-derives what the tag
        of every rebuilt cell ought to be from its own midpoint, so it holds
        regardless of which way round the permutation was applied.
        """

        mesh, tags = self._tagged_square()
        space = MeshedSpace.from_dolfinx(mesh, cell_tags=tags)

        rebuilt, rebuilt_tags = space.to_dolfinx()

        num_cells = rebuilt.topology.index_map(2).size_local
        midpoints = dolfinx.mesh.compute_midpoints(
            rebuilt, 2, np.arange(num_cells, dtype=np.int32))

        expected = self._predicate(midpoints)
        actual = np.full(num_cells, -1, dtype=np.int32)
        actual[rebuilt_tags.indices] = rebuilt_tags.values

        np.testing.assert_array_equal(actual, expected)

    def test_untagged_cells_do_not_come_back_as_minus_one(self):
        """
        from_dolfinx densifies sparse tags with -1. Handing that back would turn
        'untagged' into a material label of -1, which a lookup table would then
        either fail on or silently honour.
        """

        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_SELF, 4, 4)
        tagged = np.array([0, 1, 2], dtype=np.int32)
        tags = dolfinx.mesh.meshtags(mesh, 2, tagged, np.full(3, 7, dtype=np.int32))

        space = MeshedSpace.from_dolfinx(mesh, cell_tags=tags)
        assert (space.cell_tags == -1).any(), 'fixture should leave most cells untagged'

        _, rebuilt_tags = space.to_dolfinx()

        assert len(rebuilt_tags.indices) == 3
        assert -1 not in rebuilt_tags.values
        np.testing.assert_array_equal(np.unique(rebuilt_tags.values), [7])

    def test_no_tags_gives_no_tags(self):
        mesh, _ = self._tagged_square()
        space = MeshedSpace.from_dolfinx(mesh)

        _, rebuilt_tags = space.to_dolfinx()

        assert rebuilt_tags is None

    def test_round_trips_twice(self):
        mesh, tags = self._tagged_square()
        space = MeshedSpace.from_dolfinx(mesh, cell_tags=tags)

        rebuilt, rebuilt_tags = space.to_dolfinx()
        again = MeshedSpace.from_dolfinx(rebuilt, cell_tags=rebuilt_tags)

        assert again.cell_type == space.cell_type
        assert again.geometry_degree == space.geometry_degree
        assert again.num_nodes == space.num_nodes
        assert again.num_cells == space.num_cells
        np.testing.assert_array_equal(np.sort(again.cell_tags), np.sort(space.cell_tags))

    def test_a_surface_mesh_keeps_its_dimensions(self):
        """tdim 2 with gdim 3 has to survive, which is why cell_type is stored."""
        nodes = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 1.], [1., 1., 1.]])
        cells = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        space = MeshedSpace(nodes=nodes, cells=cells, cell_type='triangle')

        rebuilt, _ = space.to_dolfinx()

        assert rebuilt.topology.dim == 2
        assert rebuilt.geometry.dim == 3

    def test_without_cells_it_refuses(self, tri_mesh):
        nodes, _ = tri_mesh

        with pytest.raises(ValueError, match='not a mesh'):
            MeshedSpace(nodes=nodes).to_dolfinx()

    def test_a_parallel_communicator_is_refused(self):

        mesh, _ = self._tagged_square()
        space = MeshedSpace.from_dolfinx(mesh)

        class _Parallel:
            size = 4

        with pytest.raises(ValueError, match='serial only'):
            space.to_dolfinx(comm=_Parallel())

    def test_an_explicit_communicator_is_honoured(self):
        """comm=None means 'pick a default', not 'ignore what you were given'."""

        mesh, _ = self._tagged_square()
        space = MeshedSpace.from_dolfinx(mesh)

        rebuilt, _ = space.to_dolfinx(comm=MPI.COMM_SELF)

        assert rebuilt.comm.size == 1
