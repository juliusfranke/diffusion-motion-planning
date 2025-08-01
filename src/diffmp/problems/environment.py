from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, cast

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from meshlib.mrmeshpy import (
    AffineXf3f,
    Matrix3f,
    Mesh,
    Vector3f,
    boolean,
    BooleanOperation,
    makeCube,
)
from shapely import Polygon, difference, intersection, union_all
from shapely.geometry.base import BaseGeometry

from .etc import Bounds, Dim
from .obstacle import BoxObstacle, Obstacle, obstacle_from_dict


def blocked_geometry(
    env_min: Tuple[float, float],
    env_max: Tuple[float, float],
    obstacles: List[Obstacle],
) -> BaseGeometry:
    geom_env = Polygon(
        (
            (env_min[0], env_min[1]),
            (env_max[0], env_min[1]),
            (env_max[0], env_max[1]),
            (env_min[0], env_max[1]),
        )
    )
    geom_obstacles = union_all([o.geometry() for o in obstacles])
    geom_obstacles_in_env = intersection(geom_obstacles, geom_env)
    return geom_obstacles_in_env


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

        self.n_obstacles = len(self.obstacles)
        self.p_obstacles = self.area_blocked / self.area

    def update(self) -> None:
        for obstacle in self.obstacles:
            self.blocked_mesh = boolean(
                obstacle.mesh, self.blocked_mesh, BooleanOperation.Union
            ).mesh
        self.blocked_mesh = boolean(
            self.blocked_mesh, self.boundary_mesh, BooleanOperation.DifferenceAB
        ).mesh

        self.area_blocked = self.blocked_mesh.projArea(Vector3f(0, 0, 1)) / 2
        self.volume_blocked = self.blocked_mesh.volume()

    def plot(self, ax: Optional[Axes] = None):
        environment = Polygon(
            (
                (self.min[0], self.min[1]),
                (self.max[0], self.min[1]),
                (self.max[0], self.max[1]),
                (self.min[0], self.max[1]),
            )
        )
        obstacles = gpd.GeoSeries(blocked_geometry(self.min, self.max, self.obstacles))
        if ax is None:
            fig, ax = plt.subplots(1)
        assert isinstance(ax, Axes)
        # environment.plot(ax=ax)
        x, y = environment.exterior.xy
        ax.plot(x, y, color="black")
        obstacles.plot(ax=ax, color="black")

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
        cls,
        min_size: int,
        max_size: int,
        n_obstacles_min: int,
        p_obstacles: float,
    ) -> Environment:
        assert 0 <= p_obstacles <= 1
        x_max = int(np.random.random() * (max_size - min_size) + min_size)
        y_max = int(np.random.random() * (max_size - min_size) + min_size)
        bounds_environment = Bounds2D(0, x_max, 0, y_max)
        env_area = x_max * y_max
        obstacle_max_area = env_area / n_obstacles_min
        blocked_goal = env_area * p_obstacles
        obstacles = []
        count = 0
        max_tries = n_obstacles_min * 10
        while True:
            if count >= max_tries:
                return cls.random(min_size, max_size, n_obstacles_min, p_obstacles)
            count += 1
            obstacle = BoxObstacle.random(
                bounds_environment=bounds_environment, max_area=obstacle_max_area
            )
            obstacles.append(obstacle)
            area_b = blocked_geometry((0, 0), (x_max, y_max), obstacles).area
            if area_b > blocked_goal or np.isclose(area_b, blocked_goal, atol=0.1):
                break
            obstacle_max_area = min(blocked_goal - area_b, obstacle_max_area)
        return cls(obstacles=obstacles, env_min=(0, 0), env_max=(x_max, y_max))
