from enum import Enum
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from meshlib.mrmeshnumpy import getNumpyFaces, getNumpyVerts
from meshlib.mrmeshpy import (
    AffineXf3f,
    Mesh,
    MeshPart,
    Vector2f,
    Vector3f,
    findCollidingTriangles,
    findSignedDistance,
)

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from shapely.geometry import Polygon
from shapely.ops import unary_union


class Dim(Enum):
    TWO_D = 2
    THREE_D = 3


class Bounds:
    min: Vector3f
    max: Vector3f

    def __init__(self, min: Vector2f | Vector3f, max: Vector2f | Vector3f):
        assert type(min) is type(max)

        if isinstance(min, Vector2f):
            min = Vector3f(min.x, min.y, 0)
            max = Vector3f(max.x, max.y, 0)

        assert isinstance(min, Vector3f)
        assert isinstance(max, Vector3f)
        self.min = min
        self.max = max

    def random_point(self) -> Vector3f:
        random_x = np.random.random() * (self.max.x - self.min.x) + self.min.x
        random_y = np.random.random() * (self.max.y - self.min.y) + self.min.y
        random_z = np.random.random() * (self.max.z - self.min.z) + self.min.z
        return Vector3f(random_x, random_y, random_z)


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_3dmesh(
    mesh: Mesh,
    ax: Axes3D,
    alpha=0.5,
    face_color="lightblue",
    edge_color="black",
    xf: Optional[AffineXf3f] = None,
):
    verts_t = np.transpose(getNumpyVerts(mesh))
    faces_t = np.transpose(getNumpyFaces(mesh.topology))
    if xf is not None:
        A = np.array([[xf.A[r][c] for c in range(3)] for r in range(3)], dtype=float)
        b = np.array([[xf.b.x], [xf.b.y], [xf.b.z]], dtype=float)
        verts_t = A @ verts_t + b
    verts = list(zip(verts_t[0], verts_t[1], verts_t[2]))
    # if xf is not None:
    #     A = np.array([[xf.A[r][c] for c in range(3)] for r in range(3)], dtype=float)
    #     b = np.array([[xf.b.x], [xf.b.y], [xf.b.z]], dtype=float)
    #     verts_t = A @ verts_t + b

    # Each face is a triangle (i, j, k)
    tri_verts = [
        [verts[i], verts[j], verts[k]]
        for i, j, k in zip(faces_t[0], faces_t[1], faces_t[2])
    ]

    mesh_plt = Poly3DCollection(
        tri_verts, alpha=alpha, facecolor=face_color, edgecolor=edge_color
    )
    ax.add_collection3d(mesh_plt)


def plot_3dmesh_to_2d(
    mesh: Mesh,
    face_color="lightblue",
    edge_color="black",
    alpha=0.5,
    ax: Optional[Axes] = None,
    xf: Optional[AffineXf3f] = None,
):
    verts_t = np.transpose(getNumpyVerts(mesh))
    if xf is not None:
        A = np.array([[xf.A[r][c] for c in range(3)] for r in range(3)], dtype=float)
        b = np.array([[xf.b.x], [xf.b.y], [xf.b.z]], dtype=float)
        verts_t = A @ verts_t + b
    faces_t = np.transpose(getNumpyFaces(mesh.topology))
    x, y = verts_t[:2]
    i, j, k = faces_t[:3]
    if ax == None:
        fig, ax = plt.subplots()

    triangles = []
    for a, b, c in zip(i, j, k):
        triangle = Polygon([(x[a], y[a]), (x[b], y[b]), (x[c], y[c])])
        if triangle.is_valid:
            triangles.append(triangle)

    merged = unary_union(triangles)

    if merged.geom_type == "Polygon":
        xs, ys = merged.exterior.xy
        ax.fill(xs, ys, facecolor=face_color, edgecolor=edge_color, alpha=alpha)
    elif merged.geom_type == "MultiPolygon":
        for geom in merged.geoms:
            xs, ys = geom.exterior.xy
            ax.fill(xs, ys, facecolor=face_color, edgecolor=edge_color, alpha=alpha)


def _np_affine_from_mr(xf: AffineXf3f):
    A = np.array([[xf.A[r][c] for c in range(3)] for r in range(3)], dtype=float)
    b = np.array([xf.b.x, xf.b.y, xf.b.z], dtype=float)
    return A, b


def _mr_from_np(A: np.ndarray, b: np.ndarray) -> AffineXf3f:
    xf = AffineXf3f()
    for r in range(3):
        for c in range(3):
            xf.A[r][c] = float(A[r, c])
    xf.b = Vector3f(float(b[0]), float(b[1]), float(b[2]))
    return xf


def _compose(A1, b1, A2, b2):
    A = A1 @ A2
    b = A1 @ b2 + b1
    return A, b


def _invert_rigid(A: np.ndarray, b: np.ndarray):
    A_inv = A.T
    b_inv = -A_inv @ b
    return A_inv, b_inv


def _signed_distance_pt_mesh(ptA: Vector3f, mp: MeshPart):
    sd = findSignedDistance(ptA, mp)  # SignedDistanceToMeshResult

    if hasattr(sd, "signedDist"):
        return float(sd.signedDist)

    # TODO check what is correct
    dist = None
    if hasattr(sd, "dist"):  # some versions expose 'dist'
        dist = float(sd.dist)
    elif hasattr(sd, "distance"):  # or 'distance'
        dist = float(sd.distance)
    elif hasattr(sd, "distanceSq"):  # last resort: sqrt(distanceSq)
        dist = float(sd.distanceSq) ** 0.5
    else:
        dist = float("inf")

    sign = 1
    if hasattr(sd, "sign"):
        sign = -1 if sd.sign < 0 else 1

    return sign * dist


def _point_inside_or_on(pt: np.ndarray, mp: MeshPart, tol: float) -> bool:
    sdist = _signed_distance_pt_mesh(Vector3f(*pt), mp)
    return sdist <= tol

def relative_transform(xf_a: AffineXf3f, xf_b: AffineXf3f) -> AffineXf3f:
    A_a, b_a = _np_affine_from_mr(xf_a)
    A_b, b_b = _np_affine_from_mr(xf_b)
    A_a_inv, b_a_inv = _invert_rigid(A_a, b_a)
    A_rel, b_rel = _compose(A_a_inv, b_a_inv, A_b, b_b)
    return _mr_from_np(A_rel, b_rel)


def meshes_collide(
    mesh_a: Mesh, mesh_b: Mesh, xf: AffineXf3f, tol: float = 1e-6
) -> bool:
    mp_a = MeshPart(mesh_a)
    mp_b = MeshPart(mesh_b)

    # 1) Fast: face–face intersections (supports rigidB2A)
    hits = findCollidingTriangles(mp_a, mp_b, xf, True)
    if len(hits) > 0:
        return True

    # Prepare transforms
    if xf is not None:
        A, b = _np_affine_from_mr(xf)
        A_inv, b_inv = _invert_rigid(A, b)
    else:
        A = A_inv = np.eye(3, dtype=float)
        b = b_inv = np.zeros(3, dtype=float)

    # 2) Any vertex of A inside/on transformed B?
    verts_a = getNumpyVerts(mesh_a)  # shape (Na, 3)
    # Map A-vertices into B's local frame using inverse transform
    verts_a_in_B = (verts_a @ A_inv.T) + b_inv  # (Na,3)
    for pB in verts_a_in_B:
        if _point_inside_or_on(pB, mp_b, tol):
            return True

    # 3) Any vertex of transformed B inside/on A?
    verts_b = getNumpyVerts(mesh_b)  # (Nb,3)
    verts_b_in_A = (verts_b @ A.T) + b  # (Nb,3)
    for pA in verts_b_in_A:
        if _point_inside_or_on(pA, mp_a, tol):
            return True

    return False
