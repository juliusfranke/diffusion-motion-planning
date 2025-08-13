from enum import Enum

import numpy as np
from meshlib.mrmeshpy import Vector2f, Vector3f
import matplotlib.pyplot as plt

# from matplotlib.patches import Polygon
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D

# from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import Polygon
from shapely.ops import unary_union
from typing import Optional
from meshlib.mrmeshpy import Mesh
from meshlib.mrmeshnumpy import getNumpyVerts, getNumpyFaces
import numpy as np


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
    mesh: Mesh, ax: Axes3D, alpha=0.5, face_color="lightblue", edge_color="black"
):
    verts_t = np.transpose(getNumpyVerts(mesh))
    faces_t = np.transpose(getNumpyFaces(mesh.topology))
    verts = list(zip(verts_t[0], verts_t[1], verts_t[2]))

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
):
    verts_t = np.transpose(getNumpyVerts(mesh))
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
