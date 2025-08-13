from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, cast

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from shapely.geometry import Polygon
from matplotlib.collections import PatchCollection
from meshlib.mrmeshpy import (
    AffineXf3f,
    Matrix3f,
    Mesh,
    Vector3f,
    boolean,
    BooleanOperation,
    makeCube,
)

# from shapely import Polygon, difference, intersection, union_all
import random
from shapely.geometry.base import BaseGeometry

from .etc import Bounds, Dim, plot_3dmesh_to_2d, set_axes_equal, plot_3dmesh
from .obstacle import BoxObstacle, Obstacle


class Environment:
    dim: Dim
    boundary_mesh: Mesh
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
        self.min = env_min
        self.max = env_max
        self.size = env_max - env_min
        self.boundary_mesh = makeCube(base=self.min)
        if self.size.z == 0:
            self.dim = Dim.TWO_D
            transform = AffineXf3f.xfAround(
                Matrix3f.scale(self.size.x, self.size.y, 1), stable=Vector3f(self.min)
            )

        else:
            self.dim = Dim.THREE_D
            transform = AffineXf3f.xfAround(
                Matrix3f.scale(self.size), stable=Vector3f(self.min)
            )
        self.boundary_mesh.transform(transform)

        self.area = self.size.x * self.size.y
        self.volume = self.area * self.size.z

        self.update()

    def update(self) -> None:
        self.blocked_mesh = Mesh()
        for obstacle in self.obstacles:
            self.blocked_mesh = boolean(
                obstacle.mesh, self.blocked_mesh, BooleanOperation.Union
            ).mesh
        self.blocked_mesh = boolean(
            self.blocked_mesh, self.boundary_mesh, BooleanOperation.Intersection
        ).mesh
        # self.blocked_mesh = self.

        self.area_blocked = self.blocked_mesh.projArea(Vector3f(0, 0, 1))
        self.volume_blocked = self.blocked_mesh.volume()
        self.n_obstacles = len(self.obstacles)
        if self.dim == Dim.TWO_D:
            self.p_obstacles = self.area_blocked / self.area
        elif self.dim == Dim.THREE_D:
            self.p_obstacles = self.volume_blocked / self.volume
        print(f"{self.p_obstacles=}")

    def add_obstacle(self, obstacle: Obstacle) -> None:
        self.obstacles.append(obstacle)
        self.update()

    def __plot2d(self, ax: Axes):
        environment = Polygon(
            (
                (self.min.x, self.min.y),
                (self.max.x, self.min.y),
                (self.max.x, self.max.y),
                (self.min.x, self.max.y),
            )
        )
        xs, ys = environment.exterior.xy
        ax.fill(xs, ys, facecolor="white", edgecolor="black", alpha=1)
        plot_3dmesh_to_2d(self.blocked_mesh, ax=ax)

    def __plot3d(self, ax: Axes):

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
        if ax is None:
            if self.dim == Dim.TWO_D:
                fig, ax = plt.subplots(1)
            elif self.dim == Dim.THREE_D:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
        match self.dim:
            case Dim.TWO_D:
                if ax is None:
                    fig, ax = plt.subplots(1)
                self.__plot2d(ax)
                ax.autoscale()
                ax.set_aspect("equal")
            case Dim.THREE_D:
                if ax is None:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection="3d")
                self.__plot3d(ax)
                ax.set_box_aspect([1.0, 1.0, 1.0])
                set_axes_equal(ax)

        plt.show()

    def to_dict(self) -> Dict:
        return {
            "min": list(self.min),
            "max": list(self.max),
            "obstacles": [o.to_dict() for o in self.obstacles],
        }

    def random_free(self, clearance: float = 0.0) -> Optional[Tuple[float, float]]:
        max_tries = 1000
        for _ in range(max_tries):
            x = np.random.random() * (self.max[0] - self.min[0]) + self.min[0]
            y = np.random.random() * (self.max[1] - self.min[1]) + self.min[1]
            is_free = True
            for obstacle in self.obstacles:
                if obstacle.is_inside(x=x, y=y, clearance=clearance):
                    is_free = False
                    break
            if not is_free:
                continue
            break
        else:
            return None

        return (x, y)

    @classmethod
    def from_dict(
        cls, data: Dict[str, List[float] | List[int | Dict[str, Any]]]
    ) -> Environment:
        assert isinstance(data["min"], list) and len(data["min"]) == 2
        assert isinstance(data["max"], list) and len(data["max"]) == 2
        min = cast(Tuple[float, float], data["min"])
        max = cast(Tuple[float, float], data["max"])

        assert isinstance(data["obstacles"], list)

        obstacles = [
            obstacle_from_dict(obstacle)
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
        env = cls(obstacles=[], env_min=env_min, env_max=env_max)
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
