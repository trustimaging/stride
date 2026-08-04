"""
Tests for the meshed branch of GriddedSaved (stride/problem/base.py).

``GriddedSaved.grid_description`` has to emit a different payload per space
type, and ``GriddedSaved.load`` has to reconstruct the matching space subclass
from what it finds on disk. These tests pin both directions, plus the
structured-Space regressions, since the skeleton edited that code path.

Contract under test:

- a structured Space serialises as ``shape``/``spacing``/``extra``/``absorbing``
  and loads back as a Space (unchanged behaviour)
- a MeshedSpace serialises as ``nodes``/``cells`` (and ``cell_tags`` when
  present) and loads back as a MeshedSpace
- the branch is chosen from the keys present under ``description.space``, not
  from the top-level description
- an unrecognised space payload raises a clear, typed error
- a MeshedField round-trips its data and its mesh through HDF5
"""

import numpy as np
import pytest

from stride.problem.domain import Grid, MeshedSpace, Space


@pytest.fixture
def project(tmp_path):
    """Path/project_name pair for the HDF5 helpers."""
    return {'path': str(tmp_path), 'project_name': 'meshed'}


class TestGridDescriptionStructured:
    """The pre-existing Space path must keep working untouched."""

    def test_structured_space_keys(self, structured_space, project):
        from stride.problem.data import ScalarField

        field = ScalarField(name='vp_field', grid=Grid(structured_space, None, None))
        description = field.grid_description()

        assert set(description['space']) == {'shape', 'spacing', 'extra', 'absorbing'}
        assert tuple(description['space']['shape']) == (6, 8)
        assert tuple(description['space']['extra']) == (2, 2)

    def test_structured_space_has_no_mesh_keys(self, structured_space):
        from stride.problem.data import ScalarField

        field = ScalarField(name='vp_field', grid=Grid(structured_space, None, None))

        assert 'nodes' not in field.grid_description()['space']

    def test_structured_round_trip(self, structured_space, project):
        from stride.problem.data import ScalarField

        field = ScalarField(name='vp_field', grid=Grid(structured_space, None, None))
        field.fill(1500.)
        field.dump(**project)

        loaded = ScalarField(name='vp_field')
        loaded.load(**project)

        assert isinstance(loaded.space, Space)
        assert tuple(loaded.space.shape) == (6, 8)
        np.testing.assert_allclose(loaded.space.spacing, (1e-3, 1e-3))
        np.testing.assert_allclose(loaded.space.extra, (2, 2))
        np.testing.assert_allclose(loaded.space.absorbing, (1, 1))
        np.testing.assert_allclose(loaded.data, 1500.)


class TestGridDescriptionMeshed:

    def test_meshed_space_keys(self, meshed_space):
        from stride.problem.data import MeshedField

        field = MeshedField(name='sigma', grid=Grid(meshed_space, None, None))
        space_description = field.grid_description()['space']

        assert 'nodes' in space_description
        assert 'cells' in space_description

    def test_meshed_space_omits_structured_keys(self, meshed_space):
        from stride.problem.data import MeshedField

        field = MeshedField(name='sigma', grid=Grid(meshed_space, None, None))
        space_description = field.grid_description()['space']

        # 'shape' under a meshed space would make the load-time branch pick the
        # structured Space reconstruction.
        assert 'shape' not in space_description
        assert 'spacing' not in space_description

    def test_nodes_are_the_node_table(self, meshed_space):
        from stride.problem.data import MeshedField

        field = MeshedField(name='sigma', grid=Grid(meshed_space, None, None))
        nodes = np.asarray(field.grid_description()['space']['nodes'])

        assert nodes.shape == (27, 3)
        np.testing.assert_allclose(nodes, meshed_space.nodes)

    def test_cell_tags_are_included_when_present(self, tagged_meshed_space):
        from stride.problem.data import MeshedField

        field = MeshedField(name='sigma', grid=Grid(tagged_meshed_space, None, None))
        space_description = field.grid_description()['space']

        assert 'cell_tags' in space_description
        np.testing.assert_array_equal(
            np.asarray(space_description['cell_tags']),
            tagged_meshed_space.cell_tags,
        )

    def test_unknown_space_type_raises(self, monkeypatch):
        from stride.problem.data import MeshedField

        class NotASpace:
            pass

        field = MeshedField(name='sigma', shape=(4,))
        field.grid.space = NotASpace()

        # A bare `raise Exception` here would be indistinguishable from a bug
        # anywhere else in the dump path.
        with pytest.raises((TypeError, ValueError)):
            field.grid_description()


class TestMeshedRoundTrip:

    def test_space_type_survives(self, meshed_space, project):
        from stride.problem.data import MeshedField

        field = MeshedField(name='sigma', grid=Grid(meshed_space, None, None))
        field.fill(0.152)
        field.dump(**project)

        loaded = MeshedField(name='sigma')
        loaded.load(**project)

        assert isinstance(loaded.space, MeshedSpace)
        assert not isinstance(loaded.space, Space)

    def test_nodes_survive(self, meshed_space, project):
        from stride.problem.data import MeshedField

        field = MeshedField(name='sigma', grid=Grid(meshed_space, None, None))
        field.fill(0.152)
        field.dump(**project)

        loaded = MeshedField(name='sigma')
        loaded.load(**project)

        assert loaded.space.num_nodes == 27
        assert loaded.space.dim == 3
        np.testing.assert_allclose(loaded.space.nodes, meshed_space.nodes)

    def test_cells_survive(self, meshed_space, project):
        from stride.problem.data import MeshedField

        field = MeshedField(name='sigma', grid=Grid(meshed_space, None, None))
        field.fill(0.152)
        field.dump(**project)

        loaded = MeshedField(name='sigma')
        loaded.load(**project)

        # Connectivity is what makes the node table a mesh; without it the
        # reloaded space cannot be handed back to a FEM solver.
        assert loaded.space.num_cells == 48
        np.testing.assert_array_equal(loaded.space.cells, meshed_space.cells)

    def test_cell_tags_survive(self, tagged_meshed_space, project):
        from stride.problem.data import MeshedField

        field = MeshedField(name='sigma', grid=Grid(tagged_meshed_space, None, None))
        field.fill(1.)
        field.dump(**project)

        loaded = MeshedField(name='sigma')
        loaded.load(**project)

        np.testing.assert_array_equal(loaded.space.cell_tags,
                                      tagged_meshed_space.cell_tags)

    def test_data_survives(self, meshed_space, project):
        from stride.problem.data import MeshedField

        field = MeshedField(name='sigma', grid=Grid(meshed_space, None, None))
        field.allocate()
        field.data[:] = np.arange(27)
        field.dump(**project)

        loaded = MeshedField(name='sigma')
        loaded.load(**project)

        assert tuple(loaded.shape) == (27,)
        np.testing.assert_allclose(loaded.data, np.arange(27))

    def test_bounds_survive(self, meshed_space, project):
        from stride.problem.data import MeshedField

        field = MeshedField(name='sigma', grid=Grid(meshed_space, None, None))
        field.fill(1.)
        field.dump(**project)

        loaded = MeshedField(name='sigma')
        loaded.load(**project)

        np.testing.assert_allclose(loaded.space.origin, meshed_space.origin)
        np.testing.assert_allclose(loaded.space.limit, meshed_space.limit)

    def test_vector_field_round_trip(self, meshed_space, project):
        from stride.problem.data import MeshedField

        field = MeshedField(name='e_field', dim=3, grid=Grid(meshed_space, None, None))
        field.allocate()
        field.data[:] = np.arange(27 * 3).reshape(27, 3)
        field.dump(**project)

        loaded = MeshedField(name='e_field', dim=3)
        loaded.load(**project)

        assert tuple(loaded.shape) == (27, 3)
        np.testing.assert_allclose(loaded.data, np.arange(27 * 3).reshape(27, 3))

    def test_two_dimensional_mesh_round_trip(self, meshed_space_2d, project):
        from stride.problem.data import MeshedField

        field = MeshedField(name='sigma', grid=Grid(meshed_space_2d, None, None))
        field.fill(1.)
        field.dump(**project)

        loaded = MeshedField(name='sigma')
        loaded.load(**project)

        assert loaded.space.dim == 2
        assert loaded.space.num_nodes == 9
        np.testing.assert_allclose(loaded.space.nodes, meshed_space_2d.nodes)

    def test_existing_grid_is_not_overwritten(self, meshed_space, project):
        from stride.problem.data import MeshedField

        field = MeshedField(name='sigma', grid=Grid(meshed_space, None, None))
        field.fill(1.)
        field.dump(**project)

        loaded = MeshedField(name='sigma', grid=Grid(meshed_space, None, None))
        loaded.load(**project)

        # GriddedSaved.load only builds a space when the instance has none.
        assert loaded.space is meshed_space

    def test_cell_location_round_trip(self, tagged_meshed_space, project):
        from stride.problem.data import MeshedField

        field = MeshedField.from_cell_tags({1: 0.1, 2: 0.5}, name='sigma',
                                           grid=Grid(tagged_meshed_space, None, None))
        field.dump(**project)

        loaded = MeshedField(name='sigma')
        loaded.load(**project)

        # Without `location` on the description the reloaded field would default to nodal
        # and disagree with its own shape.
        assert loaded.location == 'cell'
        assert tuple(loaded.shape) == (48,)
        np.testing.assert_allclose(loaded.data, field.data)

    def test_time_dependent_round_trip(self, meshed_space, project):
        from stride.problem.data import MeshedField
        from stride.problem.domain import Time

        time = Time(start=0., step=1e-6, num=5)
        field = MeshedField(name='phi', time_dependent=True,
                            grid=Grid(meshed_space, time, None))
        field.fill(2.)
        field.dump(**project)

        loaded = MeshedField(name='phi', time_dependent=True)
        loaded.load(**project)

        assert isinstance(loaded.space, MeshedSpace)
        assert loaded.time.num == 5
        assert tuple(loaded.shape) == (5, 27)
        np.testing.assert_allclose(loaded.data, 2.)
