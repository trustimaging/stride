"""
Tests for MeshedSpace (stride/problem/domain.py).

MeshedSpace is the unstructured counterpart to Space: instead of a shape and a
spacing it is defined by an explicit node list and cell connectivity, ported
from ``ae_modelling.fem.mesh.MeshDomain`` and ``ae_modelling.fem.space``.

Contract under test:

- ``MeshedSpace(nodes, cells=None, cell_tags=None, facet_tags=None)``
- ``dim`` is inferred from ``nodes.shape[1]`` and must be 2 or 3
- ``num_nodes`` / ``num_cells``
- ``origin`` / ``limit`` / ``size`` come from the node bounding box, the
  analogue of Space's origin/limit/size
- ``shape`` is ``(num_nodes,)`` -- the shape of a scalar nodal field -- with
  ``extended_shape == shape``, ``extra`` and ``absorbing`` all-zero and
  ``inner == (slice(0, None),)``, so that the StructuredData and GriddedSaved
  machinery that reads those attributes keeps working
- ``contains_box(lower, upper)`` is the mesh-covers-the-grid check that
  ``ae_modelling.fem.mesh.attach_mesh`` performs with assertions
- ``resample`` raises, because a mesh has no spacing to resample onto
- ``from_dolfinx`` adapts an in-memory DOLFINx mesh (skipped without DOLFINx)
"""

import numpy as np
import pytest

from .conftest import box_tetra_mesh


try:
    import dolfinx  # noqa: F401
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
        from stride.problem.domain import MeshedSpace

        nodes, _ = tetra_mesh
        space = MeshedSpace(nodes=nodes)

        assert space.cells is None
        assert space.num_cells == 0
        assert space.num_nodes == 27

    def test_nodes_stored_as_float64(self, tetra_mesh):
        from stride.problem.domain import MeshedSpace

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
        from stride.problem.domain import MeshedSpace

        with pytest.raises(ValueError):
            MeshedSpace(nodes=np.zeros(10))

    def test_unsupported_dimensionality_rejected(self):
        from stride.problem.domain import MeshedSpace

        # ae_modelling.fem.mesh.make_mesh only handles dim 2 and 3.
        with pytest.raises(ValueError):
            MeshedSpace(nodes=np.zeros((10, 4)))

    def test_out_of_range_cell_indices_rejected(self, tetra_mesh):
        from stride.problem.domain import MeshedSpace

        nodes, cells = tetra_mesh
        broken = cells.copy()
        broken[0, 0] = len(nodes)

        with pytest.raises(ValueError):
            MeshedSpace(nodes=nodes, cells=broken)

    def test_cell_tags_length_must_match_cells(self, tetra_mesh):
        from stride.problem.domain import MeshedSpace

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
        from stride.problem.domain import MeshedSpace

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
        from stride.problem.data import ScalarField
        from stride.problem.domain import Grid

        # Fails on the first grid attribute it reaches for, rather than silently constructing.
        with pytest.raises(AttributeError):
            ScalarField(name='sigma', grid=Grid(meshed_space, None, None))


class TestMeshedSpaceBounds:
    """Port of the mesh-covers-the-grid assertions in ae_modelling attach_mesh."""

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
        from mpi4py import MPI

        return dolfinx.mesh.create_box(
            MPI.COMM_WORLD,
            [np.array([0., 0., 0.]), np.array([2e-3, 2e-3, 2e-3])],
            [2, 2, 2],
        )

    def test_nodes_come_from_mesh_geometry(self):
        from stride.problem.domain import MeshedSpace

        mesh = self._dolfinx_box()
        space = MeshedSpace.from_dolfinx(mesh)

        np.testing.assert_allclose(space.nodes, mesh.geometry.x[:, :3])
        assert space.dim == 3
        assert space.num_nodes == mesh.geometry.x.shape[0]

    def test_bounds_match_the_dolfinx_mesh(self):
        from stride.problem.domain import MeshedSpace

        mesh = self._dolfinx_box()
        space = MeshedSpace.from_dolfinx(mesh)

        np.testing.assert_allclose(space.origin, mesh.geometry.x.min(axis=0))
        np.testing.assert_allclose(space.limit, mesh.geometry.x.max(axis=0))

    def test_cells_come_from_topology(self):
        from stride.problem.domain import MeshedSpace

        mesh = self._dolfinx_box()
        space = MeshedSpace.from_dolfinx(mesh)

        tdim = mesh.topology.dim
        num_cells = mesh.topology.index_map(tdim).size_local

        assert space.num_cells == num_cells
        assert space.cells.shape[1] == 4
        assert space.cells.max() < space.num_nodes
