"""
Tests for loading a medium onto a MeshedSpace.

This is the stride-side port of the medium path in ``ae-modelling``:

- ``ae_modelling.tissue.nifti.NIfTILabelSampler.sample_labels`` samples a voxel
  segmentation at arbitrary coordinates via the NIfTI affine
  -> ``MeshedSpace.sample_labels(volume, affine)``, evaluated at mesh nodes
- ``sample_sigma`` / ``sample_eps`` map labels through a lookup table
  -> ``MeshedField.from_labels(labels, lut, ...)``
- ``ae_modelling.fem.space.DielectricSpace.admittivity`` combines the two into
  ``sigma + 1j * omega * eps_r * EPS0``
  -> ordinary MeshedField arithmetic, so nothing new is needed for it
- ``ae_modelling.fem.interpolate.interpolate_medium`` also supports a
  label -> value map applied per cell tag
  -> ``MeshedField.from_cell_tags(mapping, ...)``

The sampling here is nearest-voxel, matching ``world_to_voxel``'s ``np.rint``
plus clipping. Note that stride must not grow a nibabel dependency: the volume
and its affine are passed in as plain arrays, and reading the ``.nii`` file
stays the caller's job.
"""

import numpy as np
import pytest

from stride.problem.domain import Grid


# ae_modelling.fem.space.EPS0
EPS0 = 8.854e-12


def sigma_lut(tissue_properties):
    """Conductivity indexed by label, as get_tissue_sigma_array does."""
    lut = np.zeros(max(tissue_properties) + 1)
    for label, properties in tissue_properties.items():
        lut[label] = properties['sigma']
    return lut


def eps_lut(tissue_properties):
    """Relative permittivity indexed by label, as get_tissue_eps_array does."""
    lut = np.zeros(max(tissue_properties) + 1)
    for label, properties in tissue_properties.items():
        lut[label] = properties['eps_r']
    return lut


class TestTissueLookupTables:
    """Guards on the fixture itself, so the sampling tests read unambiguously."""

    def test_sigma_lut_is_indexed_by_label(self, tissue_properties):
        lut = sigma_lut(tissue_properties)

        assert lut.shape == (6,)
        assert lut[1] == pytest.approx(1.52e-1)
        assert lut[3] == pytest.approx(2e0)

    def test_eps_lut_is_indexed_by_label(self, tissue_properties):
        lut = eps_lut(tissue_properties)

        assert lut[1] == pytest.approx(2.19e3)
        assert lut[3] == pytest.approx(1.09e2)


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
        from stride.problem.domain import MeshedSpace

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
        from stride.problem.domain import MeshedSpace

        volume, _ = label_volume
        nodes, cells = tetra_mesh

        # Nodes at integer voxel positions, no affine given.
        space = MeshedSpace(nodes=nodes * 1e3, cells=cells)
        labels = space.sample_labels(volume)

        z = space.nodes[:, 2]
        np.testing.assert_array_equal(labels[z < 1.5], 1)
        np.testing.assert_array_equal(labels[z > 1.5], 3)

    def test_out_of_volume_nodes_are_clipped(self, tetra_mesh, label_volume):
        from stride.problem.domain import MeshedSpace

        volume, affine = label_volume
        nodes, cells = tetra_mesh

        # Push the mesh well past the 4x4x4 voxel volume. world_to_voxel clips
        # rather than raising, so edge nodes take the nearest in-range label.
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

    def test_conductivity_from_labels(self, meshed_space, label_volume, tissue_properties):
        from stride.problem.data import MeshedField

        volume, affine = label_volume
        labels = meshed_space.sample_labels(volume, affine=affine)

        sigma = MeshedField.from_labels(labels, sigma_lut(tissue_properties),
                                        name='sigma',
                                        grid=Grid(meshed_space, None, None))

        assert tuple(sigma.shape) == (meshed_space.num_nodes,)

        z = meshed_space.nodes[:, 2]
        np.testing.assert_allclose(sigma.data[z < 1.5e-3], 1.52e-1, rtol=1e-6)
        np.testing.assert_allclose(sigma.data[z > 1.5e-3], 2e0, rtol=1e-6)

    def test_permittivity_from_labels(self, meshed_space, label_volume, tissue_properties):
        from stride.problem.data import MeshedField

        volume, affine = label_volume
        labels = meshed_space.sample_labels(volume, affine=affine)

        eps = MeshedField.from_labels(labels, eps_lut(tissue_properties),
                                      name='eps', grid=Grid(meshed_space, None, None))

        z = meshed_space.nodes[:, 2]
        np.testing.assert_allclose(eps.data[z < 1.5e-3], 2.19e3, rtol=1e-6)
        np.testing.assert_allclose(eps.data[z > 1.5e-3], 1.09e2, rtol=1e-6)

    def test_lut_may_be_a_mapping(self, meshed_space, label_volume):
        from stride.problem.data import MeshedField

        volume, affine = label_volume
        labels = meshed_space.sample_labels(volume, affine=affine)

        # Sparse or non-contiguous label sets are common in segmentations, so a
        # dict has to work as well as an indexable array.
        field = MeshedField.from_labels(labels, {1: 10., 3: 30.}, name='sigma',
                                        grid=Grid(meshed_space, None, None))

        z = meshed_space.nodes[:, 2]
        np.testing.assert_allclose(field.data[z < 1.5e-3], 10.)
        np.testing.assert_allclose(field.data[z > 1.5e-3], 30.)

    def test_unmapped_label_raises(self, meshed_space, label_volume):
        from stride.problem.data import MeshedField

        volume, affine = label_volume
        labels = meshed_space.sample_labels(volume, affine=affine)

        # Silently defaulting an unmapped tissue to zero conductivity would
        # produce a plausible-looking but wrong solve.
        with pytest.raises(KeyError):
            MeshedField.from_labels(labels, {1: 10.}, name='sigma',
                                    grid=Grid(meshed_space, None, None))

    def test_label_count_must_match_nodes(self, meshed_space, tissue_properties):
        from stride.problem.data import MeshedField

        with pytest.raises(ValueError):
            MeshedField.from_labels(np.ones(5, dtype=np.int64),
                                    sigma_lut(tissue_properties),
                                    name='sigma',
                                    grid=Grid(meshed_space, None, None))

    def test_label_past_the_end_of_an_array_lut_raises(self, meshed_space, tissue_properties):
        from stride.problem.data import MeshedField

        labels = np.full(meshed_space.num_nodes, 9, dtype=np.int64)

        # An array lut would raise IndexError here rather than KeyError; normalise it so both
        # lut forms report an unmapped label the same way.
        with pytest.raises(KeyError):
            MeshedField.from_labels(labels, sigma_lut(tissue_properties), name='sigma',
                                    grid=Grid(meshed_space, None, None))

    def test_negative_label_raises(self, meshed_space, tissue_properties):
        from stride.problem.data import MeshedField

        labels = np.full(meshed_space.num_nodes, -1, dtype=np.int64)

        # Segmentations do use -1 as a sentinel, and a negative index would otherwise
        # silently read from the end of the lookup table.
        with pytest.raises(KeyError):
            MeshedField.from_labels(labels, sigma_lut(tissue_properties), name='sigma',
                                    grid=Grid(meshed_space, None, None))


class TestFieldFromCellTags:
    """The dict branch of ae_modelling.fem.interpolate.interpolate_medium."""

    def test_values_are_assigned_per_cell(self, tagged_meshed_space):
        from stride.problem.data import MeshedField

        field = MeshedField.from_cell_tags({1: 0.1, 2: 0.5}, name='sigma',
                                           grid=Grid(tagged_meshed_space, None, None))

        # One value per cell, as for a DG-0 medium function.
        assert tuple(field.shape) == (tagged_meshed_space.num_cells,)

        tags = tagged_meshed_space.cell_tags
        np.testing.assert_allclose(field.data[tags == 1], 0.1)
        np.testing.assert_allclose(field.data[tags == 2], 0.5)

    def test_missing_tag_in_mapping_raises(self, tagged_meshed_space):
        from stride.problem.data import MeshedField

        # Mirrors the assertion in load_mesh that sigma's keys must cover the
        # cell tag labels.
        with pytest.raises(KeyError):
            MeshedField.from_cell_tags({1: 0.1}, name='sigma',
                                       grid=Grid(tagged_meshed_space, None, None))

    def test_requires_cell_tags_on_the_space(self, meshed_space):
        from stride.problem.data import MeshedField

        with pytest.raises(ValueError):
            MeshedField.from_cell_tags({1: 0.1}, name='sigma',
                                       grid=Grid(meshed_space, None, None))


class TestAdmittivity:
    """DielectricSpace.admittivity, expressed with MeshedField arithmetic."""

    def _fields(self, meshed_space, label_volume, tissue_properties):
        from stride.problem.data import MeshedField

        volume, affine = label_volume
        labels = meshed_space.sample_labels(volume, affine=affine)
        grid = Grid(meshed_space, None, None)

        sigma = MeshedField.from_labels(labels, sigma_lut(tissue_properties),
                                        name='sigma', dtype=np.complex128, grid=grid)
        eps = MeshedField.from_labels(labels, eps_lut(tissue_properties),
                                      name='eps', dtype=np.complex128, grid=grid)
        return sigma, eps

    def test_admittivity_is_sigma_plus_j_omega_eps(self, meshed_space, label_volume,
                                                   tissue_properties):
        omega = 2 * np.pi * 5e5
        sigma, eps = self._fields(meshed_space, label_volume, tissue_properties)

        admittivity = sigma + eps * (1j * omega * EPS0)

        assert admittivity.data.dtype == np.complex128

        z = meshed_space.nodes[:, 2]
        expected = 1.52e-1 + 1j * omega * 2.19e3 * EPS0
        np.testing.assert_allclose(admittivity.data[z < 1.5e-3], expected, rtol=1e-6)

    def test_real_part_is_the_conductivity(self, meshed_space, label_volume,
                                          tissue_properties):
        omega = 2 * np.pi * 5e5
        sigma, eps = self._fields(meshed_space, label_volume, tissue_properties)

        admittivity = sigma + eps * (1j * omega * EPS0)

        np.testing.assert_allclose(admittivity.data.real, sigma.data.real, rtol=1e-6)

    def test_zero_frequency_reduces_to_conductivity(self, meshed_space, label_volume,
                                                    tissue_properties):
        sigma, eps = self._fields(meshed_space, label_volume, tissue_properties)

        admittivity = sigma + eps * (1j * 0. * EPS0)

        np.testing.assert_allclose(admittivity.data, sigma.data, rtol=1e-6)


class TestMeshedMedium:
    """Medium is space-agnostic; these pin that it stays that way for meshes."""

    @pytest.fixture
    def project(self, tmp_path):
        return {'path': str(tmp_path), 'project_name': 'meshed_medium'}

    def _medium(self, meshed_space, label_volume, tissue_properties):
        from stride.problem.data import MeshedField
        from stride.problem.medium import Medium

        volume, affine = label_volume
        labels = meshed_space.sample_labels(volume, affine=affine)
        grid = Grid(meshed_space, None, None)

        medium = Medium(grid=grid)
        medium.add(MeshedField.from_labels(labels, sigma_lut(tissue_properties),
                                          name='sigma', grid=grid))
        medium.add(MeshedField.from_labels(labels, eps_lut(tissue_properties),
                                          name='eps', grid=grid))
        return medium

    def test_fields_are_accessible_by_name(self, meshed_space, label_volume,
                                           tissue_properties):
        medium = self._medium(meshed_space, label_volume, tissue_properties)

        assert set(medium.fields) == {'sigma', 'eps'}
        assert tuple(medium.sigma.shape) == (meshed_space.num_nodes,)
        assert tuple(medium['eps'].shape) == (meshed_space.num_nodes,)

    def test_medium_round_trip(self, meshed_space, label_volume, tissue_properties,
                               project):
        from stride.problem.data import MeshedField
        from stride.problem.domain import MeshedSpace
        from stride.problem.medium import Medium

        medium = self._medium(meshed_space, label_volume, tissue_properties)
        medium.dump(**project)

        loaded = Medium()
        loaded.add(MeshedField(name='sigma'))
        loaded.add(MeshedField(name='eps'))
        loaded.load(**project)

        assert isinstance(loaded.sigma.space, MeshedSpace)
        assert loaded.sigma.space.num_nodes == meshed_space.num_nodes
        np.testing.assert_allclose(loaded.sigma.data, medium.sigma.data)
        np.testing.assert_allclose(loaded.eps.data, medium.eps.data)
