from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from meshlib.mrmeshpy import (
    AffineXf3f,
    BooleanOperation,
    Matrix3f,
    Mesh,
    MeshPart,
    Vector3f,
    boolean,
    findSignedDistance,
    makeCube,
)
from shapely.geometry import Polygon

import diffmp.utils as du
import diffmp.torch as to

from .etc import (
    Bounds,
    Dim,
    plot_3dmesh,
    plot_3dmesh_to_2d,
    set_axes_equal,
    meshes_collide,
    relative_transform,
)
from .obstacle import BoxObstacle, Obstacle

if TYPE_CHECKING:
    from diffmp.dynamics.base import DynamicsBase
    from diffmp.problems.robots import Robot


class Environment:
    dim: Dim
    area_mesh: Mesh
    blocked_mesh: Mesh

    def __init__(
        self,
        obstacles: list[Obstacle],
        env_min: Vector3f,
        env_max: Vector3f,
    ):
        assert env_min.x < env_max.x and env_min.y < env_max.y
        assert env_min.z <= env_max.z

        self.obstacles = obstacles
        size = env_max - env_min

        max_size = max(size)
        boundary_border_size = (size - Vector3f(1, 1, 1) * max_size) / 2 - Vector3f(
            1, 1, 1
        )
        boundary_size = size - 3 * boundary_border_size
        if size.z == 0:
            self.dim = Dim.TWO_D
            env_min.z = -0.5
            env_max.z = 0.5
            boundary_size.z = 0.98
            boundary_border_size.z = 0
            xf_area = AffineXf3f.xfAround(
                Matrix3f.scale(size.x, size.y, 1), stable=Vector3f(env_min)
            )
            xf_boundary = AffineXf3f.xfAround(
                Matrix3f.scale(boundary_size),
                stable=Vector3f(env_min + boundary_border_size),
            )

        else:
            self.dim = Dim.THREE_D
            xf_area = AffineXf3f.xfAround(
                Matrix3f.scale(size), stable=Vector3f(env_min)
            )
            xf_boundary = AffineXf3f.xfAround(
                Matrix3f.scale(boundary_size),
                stable=Vector3f(env_min + boundary_border_size),
            )
        self.area_mesh = makeCube(base=env_min)
        self.area_mesh.transform(xf_area)
        self.area_aabb = self.area_mesh.getBoundingBox()

        b_mesh = makeCube(base=env_min + boundary_border_size)
        b_mesh.transform(xf_boundary)
        self.boundary_mesh = boolean(
            b_mesh, self.area_mesh, BooleanOperation.DifferenceAB
        ).mesh

        self.area = size.x * size.y
        self.volume = self.area * size.z
        self.min = tuple(env_min)
        self.max = tuple(env_max)
        self.size = tuple(size)
        self.env_width = self.size[0]
        self.env_height = self.size[1]

        self.update()

    def update(self) -> None:
        self.blocked_mesh = Mesh()
        for obstacle in self.obstacles:
            self.blocked_mesh = boolean(
                obstacle.mesh, self.blocked_mesh, BooleanOperation.Union
            ).mesh
        self.blocked_mesh = boolean(
            self.blocked_mesh, self.area_mesh, BooleanOperation.Intersection
        ).mesh

        self.area_blocked = self.blocked_mesh.projArea(Vector3f(0, 0, 1))
        self.area_free = self.area - self.area_blocked
        self.volume_blocked = self.blocked_mesh.volume()
        self.n_obstacles = len(self.obstacles)
        if self.dim == Dim.TWO_D:
            self.p_obstacles = self.area_blocked / self.area
        elif self.dim == Dim.THREE_D:
            self.p_obstacles = self.volume_blocked / self.volume

    def add_obstacle(self, obstacle: Obstacle) -> None:
        self.obstacles.append(obstacle)
        self.update()

    def plot2d(self, ax: Axes):
        environment = Polygon(
            (
                (self.min[0], self.min[1]),
                (self.max[0], self.min[1]),
                (self.max[0], self.max[1]),
                (self.min[0], self.max[1]),
            )
        )
        xs, ys = environment.exterior.xy
        ax.fill(xs, ys, facecolor="white", edgecolor="black", alpha=1)
        plot_3dmesh_to_2d(self.blocked_mesh, ax=ax)

    def plot3d(self, ax: Axes):
        x0, y0, z0 = self.min
        x1, y1, z1 = self.max
        corners = np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ]
        )
        edges = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
        for start, end in edges:
            xs, ys, zs = zip(corners[start], corners[end])
            ax.plot(xs, ys, zs, color="black")
        plot_3dmesh(self.blocked_mesh, ax)

    def plot(self, ax: Optional[Axes] = None):
        match self.dim:
            case Dim.TWO_D:
                if ax is None:
                    fig, ax = plt.subplots(1)
                self.plot2d(ax)
                ax.autoscale()
                ax.set_aspect("equal")
            case Dim.THREE_D:
                if ax is None:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection="3d")
                self.plot3d(ax)
                ax.set_box_aspect([1.0, 1.0, 1.0])
                set_axes_equal(ax)

    def to_dict(self) -> dict:
        min, max = list(self.min), list(self.max)
        if self.dim == Dim.TWO_D:
            min = min[:2]
            max = max[:2]
        return {
            "min": min,
            "max": max,
            "obstacles": [o.to_dict() for o in self.obstacles],
        }

    def discretize_sd(self, n: int) -> npt.NDArray[np.floating]:
        Nx, Ny, Nz = n, n, n
        s_max = max(self.size)
        d = (s_max / n) / 2
        xs = np.linspace(0+d, s_max-d, n)
        ys = np.linspace(0+d, s_max-d, n)
        zs = np.linspace(0+d, s_max-d, n)
        # breakpoint()
        if self.dim == Dim.TWO_D:
            zs = np.array([0])
            Nz = 1

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

        blocked = boolean(
            self.blocked_mesh, self.boundary_mesh, BooleanOperation.Union
        ).mesh
        mp = MeshPart(blocked)
        sdf = np.empty((Nx, Ny, Nz), dtype=np.float32)

        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    pt = Vector3f(
                        float(X[i, j, k]), float(Y[i, j, k]), float(Z[i, j, k])
                    )
                    sd = findSignedDistance(pt, mp)
                    val = None
                    if hasattr(sd, "signedDist"):
                        val = float(sd.signedDist)  # pyright:ignore
                    elif hasattr(sd, "dist"):
                        val = float(sd.dist) * (-1 if getattr(sd, "sign", 1) < 0 else 1)
                    else:
                        raise RuntimeError(
                            "Unexpected SignedDistanceToMeshResult fields"
                        )
                    sdf[i, j, k] = val

        return sdf

    def discretize_percent(self, n: int) -> npt.NDArray[np.floating]:
        return np.array([])

    def discretize(
        self, discretize_config: to.DiscretizeConfig
    ) -> npt.NDArray[np.floating]:
        match discretize_config.method:
            case "sd":
                return self.discretize_sd(discretize_config.resolution)
            case "percent":
                return self.discretize_percent(discretize_config.resolution)

    def random_free(
        self,
        robot_dynamics: DynamicsBase,
        other_robots: list[Robot],
        max_tries: int = 1000,
    ) -> Optional[list[float]]:
        for _ in range(max_tries):
            px, py, pz = np.random.random(size=3)
            pos = du.mult_el_wise(
                Vector3f(px, py, pz), Vector3f(*self.max) - Vector3f(*self.min)
            ) + Vector3f(*self.min)

            start = robot_dynamics.random_state(x=pos.x, y=pos.y, z=pos.z)
            xf = robot_dynamics.tf_from_state(np.array(start))
            robot_aabb = robot_dynamics.mesh.computeBoundingBox(xf)
            # Check if robot mesh is completely inside environment bounds
            in_env = self.area_aabb.contains(robot_aabb)
            if not in_env:
                continue
            # Check if robot mesh is colliding with obstacles
            if meshes_collide(self.blocked_mesh, robot_dynamics.mesh, xf):
                continue
            for other_robot in other_robots:
                xf_start = other_robot.dynamics.tf_from_state(
                    np.array(other_robot.start)
                )
                xf_goal = other_robot.dynamics.tf_from_state(np.array(other_robot.goal))
                xf_start_rel = relative_transform(xf_start, xf)
                xf_goal_rel = relative_transform(xf_goal, xf)
                if meshes_collide(
                    robot_dynamics.mesh, other_robot.dynamics.mesh, xf_start_rel
                ):
                    break
                if meshes_collide(
                    robot_dynamics.mesh, other_robot.dynamics.mesh, xf_goal_rel
                ):
                    break
            else:
                return start
        return None

    @classmethod
    def from_dict(
        cls, data: dict[str, list[float] | list[int | dict[str, Any]]]
    ) -> Environment:
        assert isinstance(data["min"], list)
        assert isinstance(data["max"], list)
        assert len(data["min"]) == len(data["max"])
        if len(data["min"]) == 2:
            min = Vector3f(*data["min"], 0)
            max = Vector3f(*data["max"], 0)
        else:
            min = Vector3f(*data["min"])
            max = Vector3f(*data["max"])

        assert isinstance(data["obstacles"], list)

        obstacles = [
            BoxObstacle.from_dict(obstacle)
            for obstacle in data["obstacles"]
            if isinstance(obstacle, dict)
        ]

        return cls(obstacles=obstacles, env_min=min, env_max=max)

    @classmethod
    def random(
        cls, min_size: int, max_size: int, p_obstacles: float, dim: Dim = Dim.TWO_D
    ) -> Environment:
        assert 0 <= p_obstacles <= 1
        env_min = Vector3f(0, 0, 0)
        env_max = Vector3f(0, 0, 0)
        env_max.x = int(np.random.random() * (max_size - min_size) + min_size)
        env_max.y = int(np.random.random() * (max_size - min_size) + min_size)
        if dim == Dim.THREE_D:
            env_max.z = int(np.random.random() * (max_size - min_size) + min_size)

        env_bounds = Bounds(env_min, env_max)
        # print(f"{env_bounds.min=}")
        env = cls(obstacles=[], env_min=Vector3f(env_min), env_max=Vector3f(env_max))
        ob_small_side = 0.2
        ob_min = env_max / 10
        ob_med = env_max / 5
        ob_max = env_max / 2
        if dim == Dim.TWO_D:
            orient_x = Bounds(
                min=Vector3f(ob_min.x, ob_small_side, 0),
                max=Vector3f(ob_max.x, ob_small_side, 0),
            )
            orient_y = Bounds(
                min=Vector3f(ob_small_side, ob_min.y, 0),
                max=Vector3f(ob_small_side, ob_max.y, 0),
            )
            no_orient = Bounds(
                min=Vector3f(ob_min.x, ob_min.y, 0),
                max=Vector3f(ob_med.x, ob_med.y, 0),
            )
            ob_bounds = [orient_x, orient_y, no_orient]
        elif dim == Dim.THREE_D:
            ob_small_side = 1
            ob_med = env_max / 7
            orient_xy = Bounds(
                min=Vector3f(ob_min.x, ob_small_side, ob_min.z),
                max=Vector3f(ob_max.x, ob_small_side, ob_max.z),
            )
            orient_yz = Bounds(
                min=Vector3f(ob_small_side, ob_min.y, ob_min.z),
                max=Vector3f(ob_small_side, ob_max.y, ob_max.z),
            )
            orient_xz = Bounds(
                min=Vector3f(ob_min.x, ob_small_side, ob_min.z),
                max=Vector3f(ob_max.x, ob_small_side, ob_max.z),
            )
            no_orient = Bounds(
                min=ob_min,
                max=ob_med,
            )
            ob_bounds = [orient_xy, orient_yz, orient_xz, no_orient]

        while env.p_obstacles < p_obstacles:
            size_bounds = random.choice(ob_bounds)
            obstacle = BoxObstacle.random(
                center_bounds=env_bounds, size_bounds=size_bounds
            )
            env.add_obstacle(obstacle)
        return env
