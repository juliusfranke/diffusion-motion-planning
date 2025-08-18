from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, NamedTuple, Tuple, cast

import meshlib.mrmeshpy as mm
import numpy as np
from shapely import Point, Polygon


class Bounds2D(NamedTuple):
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def random_point(self) -> Tuple[float, float]:
        random_x = np.random.random() * (self.x_max - self.x_min) + self.x_min
        random_y = np.random.random() * (self.y_max - self.y_min) + self.y_min
        return (random_x, random_y)


@dataclass
class Bounds3D:
    min: mm.Vector3f
    max: mm.Vector3f

    def random_point(self) -> mm.Vector3f:
        random_x = np.random.random() * (self.max.x - self.min.x) + self.min.x
        random_y = np.random.random() * (self.max.y - self.min.y) + self.min.y
        random_z = np.random.random() * (self.max.z - self.min.z) + self.min.z
        return mm.Vector3f(random_x, random_y, random_z)


@dataclass
class Obstacle3(ABC):
    center: Vec3

    def area(self) -> float:
        raise NotImplementedError

    def volume(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def is_inside(self, pos: Vec3, clearance: float = 0.0) -> bool: ...

    @abstractmethod
    def to_dict(self) -> Dict: ...

    # TODO try pymesh? Shapely no 3D
    # @abstractmethod
    # def geometry(self) -> Polygon: ...

    @classmethod
    @abstractmethod
    def random(cls, bounds_environment: Bounds3D, max_size: float) -> Obstacle3: ...


class BoxObstacle2(Obstacle3):
    size: Vec2

    def __init__(self, center: Vec2, size: Vec2):
        self.size = size
        super().__init__(Vec3.from_vec2(center))

    def area(self) -> float:
        return self.size.x * self.size.y

    def to_dict(self) -> Dict:
        return {"type": "box", "center": self.center, "size": self.size}

    def is_inside(self, pos: Vec3, clearance: float = 0.0) -> bool:
        x_is_in = abs(self.center[0] - pos.x) - clearance <= self.size[0] / 2
        y_is_in = abs(self.center[1] - pos.y) - clearance <= self.size[1] / 2
        return x_is_in and y_is_in

    @classmethod
    def random(cls, bounds_environment: Bounds3D, max_size: float) -> Obstacle3:
        center = bounds_environment.random_point()
        min_side_xy = max_size / 5
        max_side_x = bounds_environment.x_max / 3
        size_x = np.random.random() * (max_side_x - min_side_xy) + min_side_xy
        max_side_y = min(max_size / size_x, bounds_environment.y_max / 3)
        size_y = np.random.random() * (max_side_y - min_side_xy) + min_side_xy
        size = [size_x, size_y]
        random.shuffle(size)
        return cls(center=list(center), size=list(size))


@dataclass
class Obstacle(ABC):
    center: Tuple[float, float]

    @abstractmethod
    def area(self) -> float: ...

    @abstractmethod
    def is_inside(self, x: float, y: float, clearance: float = 0.0) -> bool: ...

    @abstractmethod
    def to_dict(self) -> Dict: ...

    @abstractmethod
    def geometry(self) -> Polygon: ...

    @classmethod
    @abstractmethod
    def random(
        cls, bounds_environment: Bounds2D, bounds_size: Bounds2D
    ) -> Obstacle: ...


@dataclass
class BoxObstacle(Obstacle):
    size: Tuple[float, float]

    def area(self) -> float:
        return self.size[0] * self.size[1]

    def to_dict(self) -> Dict:
        return {"type": "box", "center": self.center, "size": self.size}

    def is_inside(self, x: float, y: float, clearance: float = 0.0) -> bool:
        x_is_in = abs(self.center[0] - x) - clearance <= self.size[0] / 2
        y_is_in = abs(self.center[1] - y) - clearance <= self.size[1] / 2
        return x_is_in and y_is_in

    def geometry(self) -> Polygon:
        geometry = Polygon(
            (
                (self.center[0] - self.size[0] / 2, self.center[1] - self.size[1] / 2),
                (self.center[0] + self.size[0] / 2, self.center[1] - self.size[1] / 2),
                (self.center[0] + self.size[0] / 2, self.center[1] + self.size[1] / 2),
                (self.center[0] - self.size[0] / 2, self.center[1] + self.size[1] / 2),
            )
        )
        return geometry

    @classmethod
    def random(cls, bounds_environment: Bounds2D, max_area: float) -> Obstacle:
        center = bounds_environment.random_point()
        min_side_xy = max_area / 5
        max_side_x = bounds_environment.x_max / 3
        size_x = np.random.random() * (max_side_x - min_side_xy) + min_side_xy
        max_side_y = min(max_area / size_x, bounds_environment.y_max / 3)
        size_y = np.random.random() * (max_side_y - min_side_xy) + min_side_xy
        size = [size_x, size_y]
        random.shuffle(size)
        return cls(center=list(center), size=list(size))


@dataclass
class CylinderObstacle(Obstacle):
    radius: float

    def area(self) -> float:
        return math.pi * self.radius**2

    def is_inside(self, x: float, y: float, clearance: float = 0.0) -> bool:
        d_x = abs(self.center[0] - x)
        d_y = abs(self.center[1] - y)
        d = np.linalg.norm([d_x, d_y])
        return bool(d - clearance <= self.radius)

    def to_dict(self) -> Dict:
        return {"type": "box", "center": list(self.center), "radius": self.radius}

    def geometry(self) -> Polygon:
        geometry = Point(self.center).buffer(self.radius)
        return geometry

    @classmethod
    def random(cls, bounds_environment: Bounds2D, max_area: float) -> Obstacle:
        center = bounds_environment.random_point()
        radius = np.random.random() * (max_area - max_area / 5) + max_area / 5
        return cls(center=center, radius=radius)


class ObstacleType(Enum):
    box = BoxObstacle
    cylinder = CylinderObstacle


def obstacle_from_dict(
    data: Dict[str, float | int | str | Tuple[int, int]],
) -> Obstacle:
    assert isinstance(data["type"], str)
    assert isinstance(data["center"], list) and len(data["center"]) == 2
    center = cast(Tuple[float, float], data["center"])

    obstacle_type = ObstacleType[data["type"]]
    match obstacle_type:
        case ObstacleType.box:
            assert isinstance(data["size"], list) and len(data["size"]) == 2
            size = cast(Tuple[float, float], data["size"])
            return ObstacleType.box.value(center, size)
        case ObstacleType.cylinder:
            assert isinstance(data["radius"], float | int)
            return ObstacleType.cylinder.value(center, data["radius"])
        case _:
            raise TypeError(f"Obstacle type {obstacle_type} does not exist")
