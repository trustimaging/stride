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
- inherited arithmetic operates on the nodal buffer
- ``clear_grad`` allocates a nodal gradient
"""

import numpy as np
import pytest

from stride.problem.domain import Grid


def nodal_grid(space):
    return Grid(space, None, None)


class TestMeshedDataShape:

    def test_explicit_shape(self, meshed_space):
        from stride.problem.data import MeshedData

        data = MeshedData(name='raw', shape=(27,), grid=nodal_grid(meshed_space))

        assert tuple(data.shape) == (27,)
        assert tuple(data.extended_shape) == (27,)

    def test_shape_inferred_from_space(self, meshed_space):
        from stride.problem.data import MeshedData

        data = MeshedData(name='raw', grid=nodal_grid(meshed_space))

        assert tuple(data.shape) == (meshed_space.num_nodes,)

    def test_shape_inferred_from_data(self, meshed_space):
        from stride.problem.data import MeshedData

        values = np.arange(27, dtype=np.float32)
        data = MeshedData(name='raw', data=values, grid=nodal_grid(meshed_space))

        assert tuple(data.shape) == (27,)
        np.testing.assert_array_equal(data.data, values)

    def test_mismatched_data_length_rejected(self, meshed_space):
        from stride.problem.data import MeshedData

        # A nodal field with the wrong number of values cannot be interpolated
        # onto the mesh, so this has to fail loudly rather than at solve time.
        with pytest.raises(ValueError):
            MeshedData(name='raw', data=np.zeros(26, dtype=np.float32),
                       grid=nodal_grid(meshed_space))

    def test_default_dtype_is_float32(self, meshed_space):
        from stride.problem.data import MeshedData

        data = MeshedData(name='raw', grid=nodal_grid(meshed_space))

        assert data.dtype == np.float32

    def test_complex_dtype_is_honoured(self, meshed_space):
        from stride.problem.data import MeshedData

        data = MeshedData(name='raw', dtype=np.complex128, grid=nodal_grid(meshed_space))
        data.fill(1 + 2j)

        assert data.data.dtype == np.complex128
        np.testing.assert_allclose(data.data, 1 + 2j)


class TestMeshedDataPadding:

    def test_inner_is_the_whole_buffer(self, meshed_space):
        from stride.problem.data import MeshedData

        data = MeshedData(name='raw', grid=nodal_grid(meshed_space))

        assert data.inner == (slice(0, None),)

    def test_pad_data_is_a_noop(self, meshed_space):
        from stride.problem.data import MeshedData

        data = MeshedData(name='raw', grid=nodal_grid(meshed_space))
        values = np.arange(27, dtype=np.float32)

        np.testing.assert_array_equal(data.pad_data(values), values)

    def test_data_and_extended_data_agree(self, meshed_space):
        from stride.problem.data import MeshedData

        data = MeshedData(name='raw', grid=nodal_grid(meshed_space))
        data.fill(3.)

        np.testing.assert_array_equal(data.data, data.extended_data)


class TestMeshedFieldShape:

    def test_scalar_nodal_field(self, meshed_space):
        from stride.problem.data import MeshedField

        field = MeshedField(name='sigma', grid=nodal_grid(meshed_space))

        assert tuple(field.shape) == (27,)
        assert tuple(field.extended_shape) == (27,)
        assert field.inner == (slice(0, None),)

    def test_vector_nodal_field(self, meshed_space):
        from stride.problem.data import MeshedField

        # The electric field E = -grad(phi) is the motivating case.
        field = MeshedField(name='e_field', dim=3, grid=nodal_grid(meshed_space))

        assert tuple(field.shape) == (27, 3)
        assert field.inner == (slice(0, None), slice(0, None))

    def test_time_dependent_field(self, meshed_space):
        from stride.problem.data import MeshedField
        from stride.problem.domain import Time

        time = Time(start=0., step=1e-6, num=11)
        field = MeshedField(name='phi', time_dependent=True,
                            grid=Grid(meshed_space, time, None))

        assert tuple(field.shape) == (11, 27)

    def test_time_dependent_vector_field(self, meshed_space):
        from stride.problem.data import MeshedField
        from stride.problem.domain import Time

        time = Time(start=0., step=1e-6, num=11)
        field = MeshedField(name='e_field', dim=3, time_dependent=True,
                            grid=Grid(meshed_space, time, None))

        assert tuple(field.shape) == (11, 27, 3)

    def test_dim_defaults_to_scalar(self, meshed_space):
        from stride.problem.data import MeshedField

        field = MeshedField(name='sigma', grid=nodal_grid(meshed_space))

        assert field.dim == 1

    def test_num_nodes_is_exposed(self, meshed_space):
        from stride.problem.data import MeshedField

        field = MeshedField(name='sigma', grid=nodal_grid(meshed_space))

        assert field.num_nodes == meshed_space.num_nodes

    def test_explicit_shape_overrides_the_grid(self, meshed_space):
        from stride.problem.data import MeshedField

        field = MeshedField(name='sigma', shape=(5,), grid=nodal_grid(meshed_space))

        assert tuple(field.shape) == (5,)


class TestMeshedFieldCopying:

    def test_alike_keeps_the_meshed_grid(self, meshed_space):
        from stride.problem.data import MeshedField
        from stride.problem.domain import MeshedSpace

        field = MeshedField(name='sigma', grid=nodal_grid(meshed_space))
        other = field.alike(name='eps')

        assert isinstance(other.space, MeshedSpace)
        assert other.space is meshed_space
        assert tuple(other.shape) == tuple(field.shape)
        assert other.dtype == field.dtype

    def test_copy_duplicates_the_buffer(self, meshed_space):
        from stride.problem.data import MeshedField

        field = MeshedField(name='sigma', grid=nodal_grid(meshed_space))
        field.fill(2.)

        cpy = field.copy()
        cpy.data[:] = 5.

        np.testing.assert_allclose(field.data, 2.)
        np.testing.assert_allclose(cpy.data, 5.)

    def test_copy_keeps_vector_shape(self, meshed_space):
        from stride.problem.data import MeshedField

        field = MeshedField(name='e_field', dim=3, grid=nodal_grid(meshed_space))
        field.fill(1.)

        assert tuple(field.copy().shape) == (27, 3)

    def test_detach_keeps_shape_and_data(self, meshed_space):
        from stride.problem.data import MeshedField

        field = MeshedField(name='sigma', grid=nodal_grid(meshed_space))
        field.fill(4.)

        detached = field.detach()

        assert tuple(detached.shape) == (27,)
        np.testing.assert_allclose(detached.data, 4.)


class TestMeshedFieldArithmetic:

    def test_add(self, meshed_space):
        from stride.problem.data import MeshedField

        a = MeshedField(name='a', grid=nodal_grid(meshed_space))
        a.fill(1.)
        b = a.copy()
        b.fill(2.)

        np.testing.assert_allclose((a + b).data, 3.)

    def test_multiply_by_scalar(self, meshed_space):
        from stride.problem.data import MeshedField

        field = MeshedField(name='a', grid=nodal_grid(meshed_space))
        field.fill(3.)

        np.testing.assert_allclose((field * 2).data, 6.)

    def test_in_place_add(self, meshed_space):
        from stride.problem.data import MeshedField

        field = MeshedField(name='a', grid=nodal_grid(meshed_space))
        field.fill(1.)
        field += 1.

        np.testing.assert_allclose(field.data, 2.)

    def test_operations_preserve_node_count(self, meshed_space):
        from stride.problem.data import MeshedField

        a = MeshedField(name='a', grid=nodal_grid(meshed_space))
        a.fill(1.)

        assert tuple((a * 2 + a).shape) == (27,)

    def test_elementwise_over_nodes(self, meshed_space):
        from stride.problem.data import MeshedField

        a = MeshedField(name='a', grid=nodal_grid(meshed_space))
        a.allocate()
        a.data[:] = np.arange(27)

        np.testing.assert_allclose((a * 2).data, np.arange(27) * 2)


class TestMeshedFieldGradient:

    def test_clear_grad_allocates_a_nodal_gradient(self, meshed_space):
        from stride.problem.data import MeshedField

        field = MeshedField(name='sigma', grid=nodal_grid(meshed_space),
                            needs_grad=True)
        field.clear_grad()

        assert field.grad is not None
        assert tuple(field.grad.shape) == (27,)
        np.testing.assert_allclose(field.grad.data, 0.)

    def test_gradient_has_a_preconditioner(self, meshed_space):
        from stride.problem.data import MeshedField

        field = MeshedField(name='sigma', grid=nodal_grid(meshed_space),
                            needs_grad=True)
        field.clear_grad()

        assert field.grad.prec is not None
        assert tuple(field.grad.prec.shape) == (27,)

    def test_clear_grad_is_a_noop_without_needs_grad(self, meshed_space):
        from stride.problem.data import MeshedField

        field = MeshedField(name='sigma', grid=nodal_grid(meshed_space))
        field.clear_grad()

        assert field.grad is None
