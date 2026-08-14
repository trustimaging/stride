"""
Tests for loading a medium onto a MeshedSpace.
"""

import numpy as np
import pytest

from stride.problem.domain import Grid, MeshedSpace
from stride.problem.data import MeshedField
from stride.problem.medium import Medium

def value_lut(material_properties, prop='alpha'):
    """
    One property of the material table, as an array indexed by label.

    This is the array form of a label -> value lookup: ``lut[label]`` is the
    value of ``prop`` for that material.

    """
    lut = np.zeros(max(material_properties) + 1)
    for label, properties in material_properties.items():
        lut[label] = properties[prop]
    return lut


class TestValueLookupTables:
    """Guards on the fixture itself, so the sampling tests read unambiguously."""

    def test_lut_is_indexed_by_label(self, material_properties):
        lut = value_lut(material_properties)

        assert lut.shape == (6,)
        assert lut[1] == pytest.approx(1e-1)
        assert lut[3] == pytest.approx(5e-1)

    def test_lut_selects_the_requested_property(self, material_properties):
        lut = value_lut(material_properties, prop='beta')

        assert lut[1] == pytest.approx(2e3)
        assert lut[3] == pytest.approx(1e2)


class TestSampleLabelsAtNodes:

    def test_one_label_per_node(self, meshed_space, label_volume):
        volume, affine = label_volume

        labels = meshed_space.sample_labels(volume, affine=affine)

        assert labels.shape == (meshed_space.num_nodes,)
        assert labels.dtype.kind == 'i'

    def test_labels_follow_the_volume(self, meshed_space, label_volume):
        volume, affine = label_volume

        labels = meshed_space.sample_labels(volume, affine=affine)

        # The fixture volume is 1 for z < 2 mm and 3 above, on a 1 mm voxel
        # grid; the mesh nodes sit at z = 0, 1 and 2 mm.
        z = meshed_space.nodes[:, 2]
        np.testing.assert_array_equal(labels[z < 1.5e-3], 1)
        np.testing.assert_array_equal(labels[z > 1.5e-3], 3)

    def test_affine_translation_is_applied(self, tetra_mesh, label_volume):
        volume, _ = label_volume
        nodes, cells = tetra_mesh
        space = MeshedSpace(nodes=nodes, cells=cells)

        # Shift the volume 2 mm down in z, so every mesh node now lands in the
        # volume's upper, label-3 half.
        affine = np.diag([1e-3, 1e-3, 1e-3, 1.])
        affine[2, 3] = -2e-3

        labels = space.sample_labels(volume, affine=affine)

        np.testing.assert_array_equal(labels, 3)

    def test_identity_affine_treats_nodes_as_voxel_indices(self, tetra_mesh, label_volume):
        volume, _ = label_volume
        nodes, cells = tetra_mesh

        # Nodes at integer voxel positions, no affine given.
        space = MeshedSpace(nodes=nodes * 1e3, cells=cells)
        labels = space.sample_labels(volume)

        z = space.nodes[:, 2]
        np.testing.assert_array_equal(labels[z < 1.5], 1)
        np.testing.assert_array_equal(labels[z > 1.5], 3)

    def test_out_of_volume_nodes_are_clipped(self, tetra_mesh, label_volume):
        volume, affine = label_volume
        nodes, cells = tetra_mesh

        # Push the mesh well past the 4x4x4 voxel volume.
        space = MeshedSpace(nodes=nodes + 1e-1, cells=cells)
        labels = space.sample_labels(volume, affine=affine)

        assert labels.shape == (space.num_nodes,)
        np.testing.assert_array_equal(labels, 3)

    def test_2d_volume_sampling(self, meshed_space_2d):
        volume = np.zeros((4, 4), dtype=np.int64)
        volume[:, 2:] = 5
        affine = np.diag([1e-3, 1e-3, 1.])

        labels = meshed_space_2d.sample_labels(volume, affine=affine)

        y = meshed_space_2d.nodes[:, 1]
        np.testing.assert_array_equal(labels[y < 1.5e-3], 0)
        np.testing.assert_array_equal(labels[y > 1.5e-3], 5)


class TestFieldFromLabels:

    def test_field_from_labels(self, meshed_space, label_volume, material_properties):
        volume, affine = label_volume
        labels = meshed_space.sample_labels(volume, affine=affine)

        field = MeshedField.from_labels(labels, value_lut(material_properties),
                                        name='alpha', space=meshed_space)

        assert tuple(field.shape) == (meshed_space.num_nodes,)

        z = meshed_space.nodes[:, 2]
        np.testing.assert_allclose(field.data[z < 1.5e-3], 1e-1, rtol=1e-6)
        np.testing.assert_allclose(field.data[z > 1.5e-3], 5e-1, rtol=1e-6)

    def test_a_second_property_maps_over_the_same_labels(self, meshed_space, label_volume,
                                                         material_properties):
        volume, affine = label_volume
        labels = meshed_space.sample_labels(volume, affine=affine)

        field = MeshedField.from_labels(labels, value_lut(material_properties, prop='beta'),
                                        name='beta', space=meshed_space)

        z = meshed_space.nodes[:, 2]
        np.testing.assert_allclose(field.data[z < 1.5e-3], 2e3, rtol=1e-6)
        np.testing.assert_allclose(field.data[z > 1.5e-3], 1e2, rtol=1e-6)

    def test_lut_may_be_a_mapping(self, meshed_space, label_volume):
        volume, affine = label_volume
        labels = meshed_space.sample_labels(volume, affine=affine)

        # Sparse or non-contiguous label sets are common in labelled volumes, so
        # a dict has to work as well as an indexable array.
        field = MeshedField.from_labels(labels, {1: 10., 3: 30.}, name='alpha',
                                        space=meshed_space)

        z = meshed_space.nodes[:, 2]
        np.testing.assert_allclose(field.data[z < 1.5e-3], 10.)
        np.testing.assert_allclose(field.data[z > 1.5e-3], 30.)

    def test_unmapped_label_raises(self, meshed_space, label_volume):
        volume, affine = label_volume
        labels = meshed_space.sample_labels(volume, affine=affine)

        # Silently defaulting an unmapped label to a zero property would produce
        # a plausible-looking but wrong solve.
        with pytest.raises(KeyError):
            MeshedField.from_labels(labels, {1: 10.}, name='alpha',
                                    space=meshed_space)

    def test_label_count_must_match_nodes(self, meshed_space, material_properties):
        with pytest.raises(ValueError):
            MeshedField.from_labels(np.ones(5, dtype=np.int64),
                                    value_lut(material_properties),
                                    name='alpha',
                                    space=meshed_space)

    def test_label_past_the_end_of_an_array_lut_raises(self, meshed_space, material_properties):
        labels = np.full(meshed_space.num_nodes, 9, dtype=np.int64)

        # An array lut would raise IndexError here rather than KeyError; normalise it so both
        # lut forms report an unmapped label the same way.
        with pytest.raises(KeyError):
            MeshedField.from_labels(labels, value_lut(material_properties), name='alpha',
                                    space=meshed_space)

    def test_negative_label_raises(self, meshed_space, material_properties):
        labels = np.full(meshed_space.num_nodes, -1, dtype=np.int64)

        # Labelled volumes do use -1 as a sentinel, and a negative index would otherwise
        # silently read from the end of the lookup table.
        with pytest.raises(KeyError):
            MeshedField.from_labels(labels, value_lut(material_properties), name='alpha',
                                    space=meshed_space)


class TestFieldFromCellTags:
    def test_values_are_assigned_per_cell(self, tagged_meshed_space):
        field = MeshedField.from_cell_tags({1: 0.1, 2: 0.5}, name='alpha',
                                           space=tagged_meshed_space)

        # One value per cell, as for a DG-0 medium function.
        assert tuple(field.shape) == (tagged_meshed_space.num_cells,)

        tags = tagged_meshed_space.cell_tags
        np.testing.assert_allclose(field.data[tags == 1], 0.1)
        np.testing.assert_allclose(field.data[tags == 2], 0.5)

    def test_missing_tag_in_mapping_raises(self, tagged_meshed_space):
        # A per-material mapping has to cover every cell tag label present on
        # the mesh, or some cells would be left without a value.
        with pytest.raises(KeyError):
            MeshedField.from_cell_tags({1: 0.1}, name='alpha',
                                       space=tagged_meshed_space)

    def test_requires_cell_tags_on_the_space(self, meshed_space):
        with pytest.raises(ValueError):
            MeshedField.from_cell_tags({1: 0.1}, name='alpha', space=meshed_space)


class TestComplexCombination:
    """
    Two real node fields combined into a single complex one.

    A frequency-domain medium coefficient is typically built this way: one field
    is the real part, another is scaled onto the imaginary part.
    """

    def _fields(self, meshed_space, label_volume, material_properties):
        volume, affine = label_volume
        labels = meshed_space.sample_labels(volume, affine=affine)
        grid = Grid(meshed_space, None, None)

        alpha = MeshedField.from_labels(labels, value_lut(material_properties),
                                        name='alpha', dtype=np.complex128, grid=grid)
        beta = MeshedField.from_labels(labels, value_lut(material_properties, prop='beta'),
                                       name='beta', dtype=np.complex128, grid=grid)
        return alpha, beta

    def test_combination_is_alpha_plus_j_scale_beta(self, meshed_space, label_volume,
                                                    material_properties):
        scale = 1e-4
        alpha, beta = self._fields(meshed_space, label_volume, material_properties)

        combined = alpha + beta * (1j * scale)

        assert combined.data.dtype == np.complex128

        z = meshed_space.nodes[:, 2]
        expected = 1e-1 + 1j * scale * 2e3
        np.testing.assert_allclose(combined.data[z < 1.5e-3], expected, rtol=1e-6)

    def test_real_part_is_the_first_field(self, meshed_space, label_volume,
                                          material_properties):
        scale = 1e-4
        alpha, beta = self._fields(meshed_space, label_volume, material_properties)

        combined = alpha + beta * (1j * scale)

        np.testing.assert_allclose(combined.data.real, alpha.data.real, rtol=1e-6)

    def test_zero_scale_reduces_to_the_first_field(self, meshed_space, label_volume,
                                                   material_properties):
        alpha, beta = self._fields(meshed_space, label_volume, material_properties)

        combined = alpha + beta * (1j * 0.)

        np.testing.assert_allclose(combined.data, alpha.data, rtol=1e-6)


class TestMeshedMedium:
    """Medium is space-agnostic; these pin that it stays that way for meshes."""

    @pytest.fixture
    def project(self, tmp_path):
        return {'path': str(tmp_path), 'project_name': 'meshed_medium'}

    def _medium(self, meshed_space, label_volume, material_properties):
        volume, affine = label_volume
        labels = meshed_space.sample_labels(volume, affine=affine)
        grid = Grid(meshed_space, None, None)

        medium = Medium(grid=grid)
        medium.add(MeshedField.from_labels(labels, value_lut(material_properties),
                                           name='alpha', grid=grid))
        medium.add(MeshedField.from_labels(labels, value_lut(material_properties, prop='beta'),
                                           name='beta', grid=grid))
        return medium

    def test_fields_are_accessible_by_name(self, meshed_space, label_volume,
                                           material_properties):
        medium = self._medium(meshed_space, label_volume, material_properties)

        assert set(medium.fields) == {'alpha', 'beta'}
        assert tuple(medium.alpha.shape) == (meshed_space.num_nodes,)
        assert tuple(medium['beta'].shape) == (meshed_space.num_nodes,)

    def test_medium_round_trip(self, meshed_space, label_volume, material_properties,
                               project):

        medium = self._medium(meshed_space, label_volume, material_properties)
        medium.dump(**project)

        loaded = Medium()
        loaded.add(MeshedField(name='alpha'))
        loaded.add(MeshedField(name='beta'))
        loaded.load(**project)

        assert isinstance(loaded.alpha.space, MeshedSpace)
        assert loaded.alpha.space.num_nodes == meshed_space.num_nodes
        np.testing.assert_allclose(loaded.alpha.data, medium.alpha.data)
        np.testing.assert_allclose(loaded.beta.data, medium.beta.data)
