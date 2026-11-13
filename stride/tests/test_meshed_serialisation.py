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
- ``cell_type``/``geometry_degree`` survive the round trip, so that a space read
  back from disk can still be handed to ``to_dolfinx``; a file written before
  those fields existed still loads
"""

import numpy as np
import pytest

from stride.problem.domain import Grid, Time, Space, MeshedSpace
from stride.problem.data import ScalarField, MeshedField


@pytest.fixture
def project(tmp_path):
    """Path/project_name pair for the HDF5 helpers."""
    return {'path': str(tmp_path), 'project_name': 'meshed'}


class TestGridDescriptionStructured:
    """The pre-existing Space path must keep working untouched."""

    def test_structured_space_keys(self, structured_space, project):
        field = ScalarField(name='vp_field', space=structured_space)
        description = field.grid_description()

        assert set(description['space']) == {'shape', 'spacing', 'extra', 'absorbing'}
        assert tuple(description['space']['shape']) == (6, 8)
        assert tuple(description['space']['extra']) == (2, 2)

    def test_structured_space_has_no_mesh_keys(self, structured_space):
        field = ScalarField(name='vp_field', space=structured_space)

        assert 'nodes' not in field.grid_description()['space']

    def test_structured_round_trip(self, structured_space, project):
        field = ScalarField(name='vp_field', space=structured_space)
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
        field = MeshedField(name='alpha', space=meshed_space)
        space_description = field.grid_description()['space']

        assert 'nodes' in space_description
        assert 'cells' in space_description

    def test_meshed_space_omits_structured_keys(self, meshed_space):
        field = MeshedField(name='alpha', space=meshed_space)
        space_description = field.grid_description()['space']

        # 'shape' under a meshed space would make the load-time branch pick the
        # structured Space reconstruction.
        assert 'shape' not in space_description
        assert 'spacing' not in space_description

    def test_nodes_are_the_node_table(self, meshed_space):
        field = MeshedField(name='alpha', space=meshed_space)
        nodes = np.asarray(field.grid_description()['space']['nodes'])

        assert nodes.shape == (27, 3)
        np.testing.assert_allclose(nodes, meshed_space.nodes)

    def test_cell_tags_are_included_when_present(self, tagged_meshed_space):
        field = MeshedField(name='alpha', space=tagged_meshed_space)
        space_description = field.grid_description()['space']

        assert 'cell_tags' in space_description
        np.testing.assert_array_equal(
            np.asarray(space_description['cell_tags']),
            tagged_meshed_space.cell_tags,
        )

    def test_unknown_space_type_raises(self, monkeypatch):
        class NotASpace:
            pass

        field = MeshedField(name='alpha', shape=(4,))
        field.grid.space = NotASpace()

        # A bare `raise Exception` here would be indistinguishable from a bug
        # anywhere else in the dump path.
        with pytest.raises((TypeError, ValueError)):
            field.grid_description()


class TestMeshedRoundTrip:

    def test_space_type_survives(self, meshed_space, project):
        field = MeshedField(name='alpha', space=meshed_space)
        field.fill(0.152)
        field.dump(**project)

        loaded = MeshedField(name='alpha')
        loaded.load(**project)

        assert isinstance(loaded.space, MeshedSpace)
        assert not isinstance(loaded.space, Space)

    def test_nodes_survive(self, meshed_space, project):
        field = MeshedField(name='alpha', space=meshed_space)
        field.fill(0.152)
        field.dump(**project)

        loaded = MeshedField(name='alpha')
        loaded.load(**project)

        assert loaded.space.num_nodes == 27
        assert loaded.space.dim == 3
        np.testing.assert_allclose(loaded.space.nodes, meshed_space.nodes)

    def test_cells_survive(self, meshed_space, project):
        field = MeshedField(name='alpha', space=meshed_space)
        field.fill(0.152)
        field.dump(**project)

        loaded = MeshedField(name='alpha')
        loaded.load(**project)

        # Connectivity is what makes the node table a mesh; without it the
        # reloaded space cannot be handed back to a FEM solver.
        assert loaded.space.num_cells == 48
        np.testing.assert_array_equal(loaded.space.cells, meshed_space.cells)

    def test_cell_tags_survive(self, tagged_meshed_space, project):
        field = MeshedField(name='alpha', space=tagged_meshed_space)
        field.fill(1.)
        field.dump(**project)

        loaded = MeshedField(name='alpha')
        loaded.load(**project)

        np.testing.assert_array_equal(loaded.space.cell_tags,
                                      tagged_meshed_space.cell_tags)

    def test_data_survives(self, meshed_space, project):
        field = MeshedField(name='alpha', space=meshed_space)
        field.allocate()
        field.data[:] = np.arange(27)
        field.dump(**project)

        loaded = MeshedField(name='alpha')
        loaded.load(**project)

        assert tuple(loaded.shape) == (27,)
        np.testing.assert_allclose(loaded.data, np.arange(27))

    def test_bounds_survive(self, meshed_space, project):
        field = MeshedField(name='alpha', space=meshed_space)
        field.fill(1.)
        field.dump(**project)

        loaded = MeshedField(name='alpha')
        loaded.load(**project)

        np.testing.assert_allclose(loaded.space.origin, meshed_space.origin)
        np.testing.assert_allclose(loaded.space.limit, meshed_space.limit)

    def test_vector_field_round_trip(self, meshed_space, project):
        field = MeshedField(name='vector_field', dim=3, space=meshed_space)
        field.allocate()
        field.data[:] = np.arange(27 * 3).reshape(27, 3)
        field.dump(**project)

        loaded = MeshedField(name='vector_field', dim=3)
        loaded.load(**project)

        assert tuple(loaded.shape) == (27, 3)
        np.testing.assert_allclose(loaded.data, np.arange(27 * 3).reshape(27, 3))

    def test_two_dimensional_mesh_round_trip(self, meshed_space_2d, project):
        field = MeshedField(name='alpha', space=meshed_space_2d)
        field.fill(1.)
        field.dump(**project)

        loaded = MeshedField(name='alpha')
        loaded.load(**project)

        assert loaded.space.dim == 2
        assert loaded.space.num_nodes == 9
        np.testing.assert_allclose(loaded.space.nodes, meshed_space_2d.nodes)

    def test_existing_grid_is_not_overwritten(self, meshed_space, project):
        field = MeshedField(name='alpha', space=meshed_space)
        field.fill(1.)
        field.dump(**project)

        loaded = MeshedField(name='alpha', space=meshed_space)
        loaded.load(**project)

        # GriddedSaved.load only builds a space when the instance has none.
        assert loaded.space is meshed_space

    def test_cell_location_round_trip(self, tagged_meshed_space, project):
        field = MeshedField.from_cell_tags({1: 0.1, 2: 0.5}, name='alpha',
                                           space=tagged_meshed_space)
        field.dump(**project)

        loaded = MeshedField(name='alpha')
        loaded.load(**project)

        # Without `location` on the description the reloaded field would default to node
        # and disagree with its own shape.
        assert loaded.location == 'cell'
        assert tuple(loaded.shape) == (48,)
        np.testing.assert_allclose(loaded.data, field.data)

    def test_time_dependent_round_trip(self, meshed_space, project):
        time = Time(start=0., step=1e-6, num=5)
        field = MeshedField(name='transient', time_dependent=True,
                            grid=Grid(meshed_space, time, None))
        field.fill(2.)
        field.dump(**project)

        loaded = MeshedField(name='transient', time_dependent=True)
        loaded.load(**project)

        assert isinstance(loaded.space, MeshedSpace)
        assert loaded.time.num == 5
        assert tuple(loaded.shape) == (5, 27)
        np.testing.assert_allclose(loaded.data, 2.)


class TestDiscretisationSurvivesTheRoundTrip:
    """
    cell_type and geometry_degree are what make a stored space rebuildable. Without
    them a node and cell table does not say what shape a cell is, so to_dolfinx has
    nothing to hand basix.
    """

    def test_description_carries_the_discretisation(self, meshed_space):
        field = MeshedField(name='alpha', space=meshed_space)
        space_description = field.grid_description()['space']

        assert space_description['cell_type'] == 'tetrahedron'
        assert space_description['geometry_degree'] == 1

    def test_round_trip_keeps_the_cell_type(self, meshed_space, project):
        field = MeshedField(name='alpha', space=meshed_space, dtype=np.float64)
        field.fill(1.)
        field.dump(**project)

        loaded = MeshedField(name='alpha')
        loaded.load(**project)

        assert isinstance(loaded.space, MeshedSpace)
        assert loaded.space.cell_type == 'tetrahedron'
        assert loaded.space.geometry_degree == 1

    def test_cell_type_survives_as_str_not_bytes(self, meshed_space, project):
        """HDF5 hands strings back as bytes, which would break the isinstance check."""
        field = MeshedField(name='alpha', space=meshed_space, dtype=np.float64)
        field.dump(**project)

        loaded = MeshedField(name='alpha')
        loaded.load(**project)

        assert isinstance(loaded.space.cell_type, str)
        assert isinstance(loaded.space.geometry_degree, int)

    def test_a_loaded_space_can_still_be_rebuilt(self, meshed_space, project):
        """The point of storing them at all."""
        pytest.importorskip('dolfinx')

        field = MeshedField(name='alpha', space=meshed_space, dtype=np.float64)
        field.dump(**project)

        loaded = MeshedField(name='alpha')
        loaded.load(**project)

        mesh, _, _ = loaded.space.to_dolfinx()

        assert mesh.topology.dim == 3
        assert mesh.topology.index_map(3).size_local == meshed_space.num_cells

    def test_a_file_without_the_new_fields_still_loads(self, meshed_space, project):
        """
        Backward compatibility: the fields are read with defaults, so a space written
        before they existed reconstructs, falling back on inference for the cell type.
        """
        field = MeshedField(name='alpha', space=meshed_space, dtype=np.float64)
        description = field.grid_description()

        del description['space']['cell_type']
        del description['space']['geometry_degree']

        assert 'cell_type' not in description['space']

        # what the load branch does with what it finds, minus the HDF5 layer
        space = MeshedSpace(
            nodes=description['space']['nodes'],
            cells=description['space']['cells'],
            cell_type=description['space'].get('cell_type', None),
            geometry_degree=int(description['space'].get('geometry_degree', 1)),
        )

        assert space.cell_type == 'tetrahedron'
        assert space.geometry_degree == 1


class TestFacetTagsSurviveTheRoundTrip:
    """
    A facet tag used to be dropped on the way to a file, because a DOLFINx facet index means
    nothing once the mesh is rebuilt. Naming the facet by its nodes instead makes it storable:
    the name is the connectivity, so nothing else has to travel with it.

    Without this, electrodes named by facet tag work in process and vanish through
    ``dump``/``load``, which is a surprise that only shows up once a Problem is reloaded.
    """

    @staticmethod
    def _tagged(meshed_space):
        """Tag the two faces at the extremes of x, by the nodes that make them up."""

        nodes = meshed_space.nodes
        faces = []

        for cell in meshed_space.cells:
            for drop in range(cell.shape[0]):
                facet = np.delete(cell, drop)

                if np.allclose(nodes[facet][:, 0], nodes[facet][0, 0]):
                    faces.append((np.sort(facet), nodes[facet][0, 0]))

        lower, upper = nodes[:, 0].min(), nodes[:, 0].max()
        tagged = [(facet, 2 if np.isclose(x, lower) else 3)
                  for facet, x in faces if np.isclose(x, lower) or np.isclose(x, upper)]

        assert tagged, 'fixture should find some faces to tag'

        return MeshedSpace(
            nodes=nodes, cells=meshed_space.cells, cell_type=meshed_space.cell_type,
            facet_tags={'nodes': np.array([facet for facet, _ in tagged]),
                        'values': np.array([value for _, value in tagged], dtype=np.int32)})

    def test_the_tags_come_back(self, meshed_space, project):
        space = self._tagged(meshed_space)

        field = MeshedField(name='alpha', space=space, dtype=np.float64)
        field.dump(**project)

        loaded = MeshedField(name='alpha')
        loaded.load(**project)

        assert loaded.space.facet_tags is not None

        np.testing.assert_array_equal(loaded.space.facet_tags['nodes'],
                                      space.facet_tags['nodes'])
        np.testing.assert_array_equal(loaded.space.facet_tags['values'],
                                      space.facet_tags['values'])

    def test_a_space_without_them_still_round_trips(self, meshed_space, project):
        field = MeshedField(name='alpha', space=meshed_space, dtype=np.float64)
        field.dump(**project)

        loaded = MeshedField(name='alpha')
        loaded.load(**project)

        assert loaded.space.facet_tags is None

    def test_a_loaded_space_can_still_rebuild_its_tags(self, meshed_space, project):
        """The point of storing them: the rebuilt mesh has to carry the same boundaries."""

        pytest.importorskip('dolfinx')

        space = self._tagged(meshed_space)

        field = MeshedField(name='alpha', space=space, dtype=np.float64)
        field.dump(**project)

        loaded = MeshedField(name='alpha')
        loaded.load(**project)

        _, _, facet_tags = loaded.space.to_dolfinx()

        assert facet_tags is not None
        np.testing.assert_array_equal(np.sort(facet_tags.values),
                                      np.sort(space.facet_tags['values']))

    def test_meshtags_are_refused_with_an_explanation(self, meshed_space):
        """
        The obvious mistake is handing the DOLFINx object straight over. It has indices rather
        than nodes, so it would store something that means nothing after a rebuild.
        """
        class _MeshTags:
            indices = np.array([0, 1])
            values = np.array([2, 3])

        with pytest.raises(ValueError, match='mapping with "nodes" and "values"'):
            MeshedSpace(nodes=meshed_space.nodes, cells=meshed_space.cells,
                        facet_tags=_MeshTags())
