"""
Shared fixtures for the FEM/meshed tests.

The reference mesh is a Kuhn (Freudenthal) tetrahedralisation of a structured
box: every hexahedral cell of an ``(nx, ny, nz)`` node grid is split into six
tetrahedra. This gives a genuine unstructured mesh (flat node list, explicit
connectivity) without needing DOLFINx or gmsh to be installed, which mirrors
what ``ae_modelling.fem.mesh.make_mesh`` produces via
``dolfinx.mesh.create_box``.
"""

import numpy as np
import pytest


# Local corner index within a hexahedron is 4*i + 2*j + k, so the six tets of
# the Kuhn decomposition along the 0 -> 7 diagonal are:
KUHN_TETS = (
    (0, 1, 3, 7),
    (0, 1, 5, 7),
    (0, 2, 3, 7),
    (0, 2, 6, 7),
    (0, 4, 5, 7),
    (0, 4, 6, 7),
)


def box_tetra_mesh(shape=(3, 3, 3), spacing=(1e-3, 1e-3, 1e-3), origin=(0., 0., 0.)):
    """
    Build a tetrahedral mesh of an axis-aligned box.

    Parameters
    ----------
    shape : tuple
        Number of nodes per axis.
    spacing : tuple
        Node spacing per axis, in metres.
    origin : tuple
        Lower corner of the box, in metres.

    Returns
    -------
    nodes : ndarray
        ``(num_nodes, 3)`` float64 node coordinates.
    cells : ndarray
        ``(num_cells, 4)`` int32 node indices, one row per tetrahedron.

    """
    axes = [np.arange(n) * d + o for n, d, o in zip(shape, spacing, origin)]
    mesh = np.meshgrid(*axes, indexing='ij')
    nodes = np.stack([each.ravel() for each in mesh], axis=-1).astype(np.float64)

    strides = (shape[1] * shape[2], shape[2], 1)

    cells = []
    for i in range(shape[0] - 1):
        for j in range(shape[1] - 1):
            for k in range(shape[2] - 1):
                corners = [
                    (i + ((local >> 2) & 1)) * strides[0] +
                    (j + ((local >> 1) & 1)) * strides[1] +
                    (k + (local & 1)) * strides[2]
                    for local in range(8)
                ]
                for tet in KUHN_TETS:
                    cells.append([corners[each] for each in tet])

    return nodes, np.asarray(cells, dtype=np.int32)


def triangle_mesh(shape=(3, 3), spacing=(1e-3, 1e-3), origin=(0., 0.)):
    """
    Build a 2D triangular mesh of an axis-aligned rectangle.

    Returns
    -------
    nodes : ndarray
        ``(num_nodes, 2)`` float64 node coordinates.
    cells : ndarray
        ``(num_cells, 3)`` int32 node indices, one row per triangle.

    """
    axes = [np.arange(n) * d + o for n, d, o in zip(shape, spacing, origin)]
    mesh = np.meshgrid(*axes, indexing='ij')
    nodes = np.stack([each.ravel() for each in mesh], axis=-1).astype(np.float64)

    cells = []
    for i in range(shape[0] - 1):
        for j in range(shape[1] - 1):
            bottom_left = i * shape[1] + j
            bottom_right = bottom_left + 1
            top_left = bottom_left + shape[1]
            top_right = top_left + 1
            cells.append([bottom_left, bottom_right, top_right])
            cells.append([bottom_left, top_right, top_left])

    return nodes, np.asarray(cells, dtype=np.int32)


@pytest.fixture
def tetra_mesh():
    """Nodes and cells of a 3x3x3-node, 1 mm box (27 nodes, 48 tets)."""
    return box_tetra_mesh(shape=(3, 3, 3), spacing=(1e-3, 1e-3, 1e-3))


@pytest.fixture
def tri_mesh():
    """Nodes and cells of a 3x3-node, 1 mm rectangle (9 nodes, 8 triangles)."""
    return triangle_mesh(shape=(3, 3), spacing=(1e-3, 1e-3))


@pytest.fixture
def meshed_space(tetra_mesh):
    """A 3D MeshedSpace over the reference tetrahedral mesh."""
    from stride.problem.domain import MeshedSpace

    nodes, cells = tetra_mesh
    return MeshedSpace(nodes=nodes, cells=cells)


@pytest.fixture
def meshed_space_2d(tri_mesh):
    """A 2D MeshedSpace over the reference triangular mesh."""
    from stride.problem.domain import MeshedSpace

    nodes, cells = tri_mesh
    return MeshedSpace(nodes=nodes, cells=cells)


@pytest.fixture
def tagged_meshed_space(tetra_mesh):
    """
    A MeshedSpace whose cells carry two material tags.

    Cells in the lower half of the box (in z) are tagged ``1``, the rest ``2``,
    mirroring the ``cell_tags`` that ``ae_modelling`` reads out of a gmsh file.
    """
    from stride.problem.domain import MeshedSpace

    nodes, cells = tetra_mesh
    centroids = nodes[cells].mean(axis=1)
    cell_tags = np.where(centroids[:, 2] < 1e-3, 1, 2).astype(np.int32)

    return MeshedSpace(nodes=nodes, cells=cells, cell_tags=cell_tags)


@pytest.fixture
def structured_space():
    """A conventional structured Space, used for the serialisation regressions."""
    from stride.problem.domain import Space

    return Space(shape=(6, 8), spacing=(1e-3, 1e-3), extra=(2, 2), absorbing=(1, 1))


@pytest.fixture
def tissue_properties():
    """
    Label -> (conductivity, relative permittivity) map.

    Values are the defaults from ``ae_modelling.tissue.properties``.

    """
    return {
        0: {'name': 'background', 'sigma': 1e-6, 'eps_r': 1e0},
        1: {'name': 'grey_matter', 'sigma': 1.52e-1, 'eps_r': 2.19e3},
        2: {'name': 'white_matter', 'sigma': 9.47e-2, 'eps_r': 7.12e2},
        3: {'name': 'csf', 'sigma': 2e0, 'eps_r': 1.09e2},
        4: {'name': 'skull', 'sigma': 2.22e-2, 'eps_r': 1.75e2},
        5: {'name': 'skin', 'sigma': 4.36e-3, 'eps_r': 1.06e3},
    }


@pytest.fixture
def label_volume():
    """
    A small voxel label volume plus its world affine.

    Voxels are 1 mm isotropic with the volume origin at the world origin, so
    world coordinate ``x`` maps to voxel index ``round(x / 1e-3)``. Labels vary
    along z only: the lower half is ``1``, the upper half is ``3``.
    """
    volume = np.zeros((4, 4, 4), dtype=np.int64)
    volume[:, :, :2] = 1
    volume[:, :, 2:] = 3

    affine = np.diag([1e-3, 1e-3, 1e-3, 1.])

    return volume, affine
