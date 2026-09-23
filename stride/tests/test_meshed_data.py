"""
Tests for MeshedData and MeshedField (stride/problem/data.py).

MeshedData is to MeshedSpace what StructuredData is to Space: a buffer with an
explicit shape and the arithmetic/gradient machinery. MeshedField is to
MeshedData what ScalarField is to StructuredData: it derives its shape from the
grid, and optionally prepends time and slow-time axes.

Because a mesh has no padding, the buffer is flat and ``extended_shape ==
shape``, exactly as for SparseField. That is the existing class these two
follow most closely.

Contract under test:

- ``MeshedData`` takes an explicit ``shape`` or ``data``, like StructuredData
- ``MeshedField`` derives ``(num_nodes,)`` from ``space``, with the same
  ``time_dependent`` / ``slow_time_dependent`` / ``dim`` options as SparseField
- padding is a no-op on both
- ``alike`` / ``copy`` / ``detach`` carry the meshed grid across
- inherited arithmetic operates on the node buffer
- ``clear_grad`` allocates a node gradient
"""

import numpy as np
import pytest
from stride.problem.data import MeshedData, MeshedField
from stride.problem.domain import Grid, MeshedSpace, Time


class TestMeshedDataShape:

    def test_explicit_shape(self, meshed_space):
        data = MeshedData(name='raw', shape=(27,), space=meshed_space)

        assert tuple(data.shape) == (27,)
        assert tuple(data.extended_shape) == (27,)

    def test_shape_inferred_from_space(self, meshed_space):
        data = MeshedData(name='raw', space=meshed_space)

        assert tuple(data.shape) == (meshed_space.num_nodes,)

    def test_shape_inferred_from_data(self, meshed_space):
        values = np.arange(27, dtype=np.float32)
        data = MeshedData(name='raw', data=values, space=meshed_space)

        assert tuple(data.shape) == (27,)
        np.testing.assert_array_equal(data.data, values)

    @pytest.mark.parametrize('num_values', [25, 26, 29])
    def test_mismatched_data_length_rejected(self, meshed_space, num_values):
        # A node field with the wrong number of values cannot be interpolated
        # onto the mesh, so this has to fail loudly rather than at solve time.
        # 25 is the case that matters: the inherited StructuredData.pad_data floors its
        # pad widths, so an off-by-two would otherwise be silently edge-padded to 27.
        with pytest.raises(ValueError):
            MeshedData(name='raw', data=np.zeros(num_values, dtype=np.float32),
                       space=meshed_space)

    def test_compression_is_rejected(self, meshed_space):
        # maybe_compress crashes outright for buffers of 2.5k-10k float32 elements, and
        # above that window the decompressed buffer is read-only.
        with pytest.raises(ValueError, match='[Cc]ompression'):
            MeshedData(name='raw', compressed=True, space=meshed_space)

    def test_default_dtype_is_float32(self, meshed_space):
        data = MeshedData(name='raw', space=meshed_space)

        assert data.dtype == np.float32

    def test_complex_dtype_is_honoured(self, meshed_space):
        data = MeshedData(name='raw', dtype=np.complex128, space=meshed_space)
        data.fill(1 + 2j)

        assert data.data.dtype == np.complex128
        np.testing.assert_allclose(data.data, 1 + 2j)


class TestMeshedDataPadding:

    def test_inner_is_the_whole_buffer(self, meshed_space):
        data = MeshedData(name='raw', space=meshed_space)

        assert data.inner == (slice(0, None),)

    def test_pad_data_is_a_noop(self, meshed_space):
        data = MeshedData(name='raw', space=meshed_space)
        values = np.arange(27, dtype=np.float32)

        np.testing.assert_array_equal(data.pad_data(values), values)

    def test_data_and_extended_data_agree(self, meshed_space):
        data = MeshedData(name='raw', space=meshed_space)
        data.fill(3.)

        np.testing.assert_array_equal(data.data, data.extended_data)


class TestMeshedFieldShape:

    def test_scalar_node_field(self, meshed_space):
        field = MeshedField(name='alpha', space=meshed_space)

        assert tuple(field.shape) == (27,)
        assert tuple(field.extended_shape) == (27,)
        assert field.inner == (slice(0, None),)

    def test_vector_node_field(self, meshed_space):
        field = MeshedField(name='vector_field', dim=3, space=meshed_space)

        assert tuple(field.shape) == (27, 3)
        assert field.inner == (slice(0, None), slice(0, None))

    def test_time_dependent_field(self, meshed_space):
        time = Time(start=0., step=1e-6, num=11)
        field = MeshedField(name='transient', time_dependent=True,
                            grid=Grid(meshed_space, time, None))

        assert tuple(field.shape) == (11, 27)

    def test_time_dependent_vector_field(self, meshed_space):
        time = Time(start=0., step=1e-6, num=11)
        field = MeshedField(name='vector_field', dim=3, time_dependent=True,
                            grid=Grid(meshed_space, time, None))

        assert tuple(field.shape) == (11, 27, 3)

    def test_dim_defaults_to_scalar(self, meshed_space):
        field = MeshedField(name='alpha', space=meshed_space)

        assert field.dim == 1

    def test_num_nodes_is_exposed(self, meshed_space):
        field = MeshedField(name='alpha', space=meshed_space)

        assert field.num_nodes == meshed_space.num_nodes

    def test_explicit_shape_overrides_the_grid(self, meshed_space):
        field = MeshedField(name='alpha', shape=(5,), space=meshed_space)

        assert tuple(field.shape) == (5,)


class TestMeshedFieldLocation:
    """
    A meshed field lives on nodes, edges or cells, and `shape` alone does not say which,
    so `location` is what keeps them apart.
    """

    def test_defaults_to_node(self, meshed_space):
        field = MeshedField(name='alpha', space=meshed_space)

        assert field.location == 'node'
        assert field.num_entities == meshed_space.num_nodes

    def test_cell_location_sizes_from_cells(self, tagged_meshed_space):
        field = MeshedField(name='alpha', location='cell',
                            space=tagged_meshed_space)

        assert field.location == 'cell'
        assert field.num_entities == tagged_meshed_space.num_cells
        assert tuple(field.shape) == (48,)

    def test_from_cell_tags_reports_cell_location(self, tagged_meshed_space):
        field = MeshedField.from_cell_tags({1: 0.1, 2: 0.5}, name='alpha',
                                           space=tagged_meshed_space)

        assert field.location == 'cell'

    def test_from_cell_tags_values_are_consistent(self, tagged_meshed_space):
        field = MeshedField.from_cell_tags({1: 0.1, 2: 0.5}, name='alpha',
                                           space=tagged_meshed_space)

        np.testing.assert_allclose(field.data,
                                   np.where(tagged_meshed_space.cell_tags == 1, 0.1, 0.5))

    def test_invalid_location_rejected(self, meshed_space):
        with pytest.raises(ValueError):
            MeshedField(name='alpha', location='facet', space=meshed_space)

    def test_alike_preserves_location(self, tagged_meshed_space):
        field = MeshedField.from_cell_tags({1: 0.1, 2: 0.5}, name='alpha',
                                           space=tagged_meshed_space)
        other = field.alike(name='beta')

        assert other.location == 'cell'
        assert tuple(other.shape) == (48,)


class TestMeshedFieldEdgeLocation:
    """
    A Lagrange element of degree 2 keeps half its dofs on edges, and they have to live
    somewhere to cross between a worker and the head. Edges are the third entity dimension a
    simplex mesh has, so they sit alongside nodes and cells rather than replacing either.
    """

    def test_an_edge_field_sizes_from_the_edges(self, meshed_space):
        field = MeshedField(name='alpha', location='edge', space=meshed_space)

        assert field.location == 'edge'
        assert field.num_entities == meshed_space.num_edges
        assert tuple(field.shape) == (meshed_space.num_edges,)

    def test_edges_outnumber_nodes_so_the_shape_is_not_the_node_one(self, meshed_space):
        """A location that silently fell back to nodes would pass every test above."""

        field = MeshedField(name='alpha', location='edge', space=meshed_space)

        assert meshed_space.num_edges != meshed_space.num_nodes
        assert field.num_entities != meshed_space.num_nodes

    def test_a_vector_edge_field_keeps_its_components(self, meshed_space):
        field = MeshedField(name='alpha', location='edge', dim=3, space=meshed_space)

        assert tuple(field.shape) == (meshed_space.num_edges, 3)

    def test_the_wrong_length_is_still_rejected(self, meshed_space):
        with pytest.raises(ValueError, match='edge entities'):
            MeshedField(name='alpha', location='edge', space=meshed_space,
                        data=np.zeros(meshed_space.num_nodes))

    def test_alike_preserves_the_edge_location(self, meshed_space):
        field = MeshedField(name='alpha', location='edge', space=meshed_space)

        assert field.alike(name='beta').location == 'edge'

    def test_num_edges_is_exposed(self, meshed_space):
        field = MeshedField(name='alpha', location='edge', space=meshed_space)

        assert field.num_edges == meshed_space.num_edges


class TestMeshedFieldCopying:

    def test_alike_keeps_the_meshed_grid(self, meshed_space):

        field = MeshedField(name='alpha', space=meshed_space)
        other = field.alike(name='beta')

        assert isinstance(other.space, MeshedSpace)
        assert other.space is meshed_space
        assert tuple(other.shape) == tuple(field.shape)
        assert other.dtype == field.dtype

    def test_copy_duplicates_the_buffer(self, meshed_space):
        field = MeshedField(name='alpha', space=meshed_space)
        field.fill(2.)

        cpy = field.copy()
        cpy.data[:] = 5.

        np.testing.assert_allclose(field.data, 2.)
        np.testing.assert_allclose(cpy.data, 5.)

    def test_copy_keeps_vector_shape(self, meshed_space):
        field = MeshedField(name='vector_field', dim=3, space=meshed_space)
        field.fill(1.)

        assert tuple(field.copy().shape) == (27, 3)

    def test_detach_keeps_shape_and_data(self, meshed_space):
        field = MeshedField(name='alpha', space=meshed_space)
        field.fill(4.)

        detached = field.detach()

        assert tuple(detached.shape) == (27,)
        np.testing.assert_allclose(detached.data, 4.)


class TestMeshedFieldArithmetic:

    def test_add(self, meshed_space):
        a = MeshedField(name='a', space=meshed_space)
        a.fill(1.)
        b = a.copy()
        b.fill(2.)

        np.testing.assert_allclose((a + b).data, 3.)

    def test_multiply_by_scalar(self, meshed_space):
        field = MeshedField(name='a', space=meshed_space)
        field.fill(3.)

        np.testing.assert_allclose((field * 2).data, 6.)

    def test_in_place_add(self, meshed_space):
        field = MeshedField(name='a', space=meshed_space)
        field.fill(1.)
        field += 1.

        np.testing.assert_allclose(field.data, 2.)

    def test_operations_preserve_node_count(self, meshed_space):
        a = MeshedField(name='a', space=meshed_space)
        a.fill(1.)

        assert tuple((a * 2 + a).shape) == (27,)

    def test_elementwise_over_nodes(self, meshed_space):
        a = MeshedField(name='a', space=meshed_space)
        a.allocate()
        a.data[:] = np.arange(27)

        np.testing.assert_allclose((a * 2).data, np.arange(27) * 2)


class TestMeshedFieldGradient:

    def test_clear_grad_allocates_a_node_gradient(self, meshed_space):
        field = MeshedField(name='alpha', space=meshed_space,
                            needs_grad=True)
        field.clear_grad()

        assert field.grad is not None
        assert tuple(field.grad.shape) == (27,)
        np.testing.assert_allclose(field.grad.data, 0.)

    def test_gradient_has_a_preconditioner(self, meshed_space):
        field = MeshedField(name='alpha', space=meshed_space,
                            needs_grad=True)
        field.clear_grad()

        assert field.grad.prec is not None
        assert tuple(field.grad.prec.shape) == (27,)

    def test_clear_grad_is_a_noop_without_needs_grad(self, meshed_space):
        field = MeshedField(name='alpha', space=meshed_space)
        field.clear_grad()

        assert field.grad is None
