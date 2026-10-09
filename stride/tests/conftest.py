"""
Shared fixtures for the FEM/meshed tests.

The reference mesh is a Kuhn (Freudenthal) tetrahedralisation of a structured
box: every hexahedral cell of an ``(nx, ny, nz)`` node grid is split into six
tetrahedra. This gives a genuine unstructured mesh (flat node list, explicit
connectivity) without needing DOLFINx or gmsh to be installed, mirroring what
``dolfinx.mesh.create_box`` produces.
"""

import numpy as np
import pytest
from stride.problem.domain import MeshedSpace, Space


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
    nodes, cells = tetra_mesh
    return MeshedSpace(nodes=nodes, cells=cells)


@pytest.fixture
def meshed_space_2d(tri_mesh):
    """A 2D MeshedSpace over the reference triangular mesh."""
    nodes, cells = tri_mesh
    return MeshedSpace(nodes=nodes, cells=cells)


@pytest.fixture
def tagged_meshed_space(tetra_mesh):
    """
    A MeshedSpace whose cells carry two material tags.

    Cells in the lower half of the box (in z) are tagged ``1``, the rest ``2``,
    mirroring the ``cell_tags`` that come out of a gmsh file.
    """
    nodes, cells = tetra_mesh
    centroids = nodes[cells].mean(axis=1)
    cell_tags = np.where(centroids[:, 2] < 1e-3, 1, 2).astype(np.int32)

    return MeshedSpace(nodes=nodes, cells=cells, cell_tags=cell_tags)


@pytest.fixture
def structured_space():
    """A conventional structured Space, used for the serialisation regressions."""
    return Space(shape=(6, 8), spacing=(1e-3, 1e-3), extra=(2, 2), absorbing=(1, 1))


@pytest.fixture
def material_properties():
    """
    Label -> material property map.

    Each label carries two independent scalar properties, ``alpha`` and
    ``beta``, so that tests can build more than one field from the same set of
    labels. The names and values are arbitrary: what matters is that a label
    indexes a set of physical values.

    """
    return {
        0: {'name': 'material_0', 'alpha': 1e-6, 'beta': 1e0},
        1: {'name': 'material_1', 'alpha': 1e-1, 'beta': 2e3},
        2: {'name': 'material_2', 'alpha': 2e-1, 'beta': 7e2},
        3: {'name': 'material_3', 'alpha': 5e-1, 'beta': 1e2},
        4: {'name': 'material_4', 'alpha': 8e-1, 'beta': 3e2},
        5: {'name': 'material_5', 'alpha': 1e0, 'beta': 5e2},
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
