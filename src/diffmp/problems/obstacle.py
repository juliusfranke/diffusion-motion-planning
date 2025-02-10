from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import math
import numpy as np
from typing import NamedTuple, Tuple, Dict, cast


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
class Obstacle(ABC):
    center: Tuple[float, float]

    @abstractmethod
    def area(self) -> float: ...

    @abstractmethod
    def is_inside(self, x: float, y: float) -> bool: ...

    @abstractmethod
    def to_dict(self) -> Dict: ...

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

    def is_inside(self, x: float, y: float) -> bool:
        x_is_in = abs(self.center[0] - x) <= self.size[0] / 2
        y_is_in = abs(self.center[1] - y) <= self.size[1] / 2
        return x_is_in and y_is_in

    @classmethod
    def random(cls, bounds_environment: Bounds2D, bounds_size: Bounds2D) -> Obstacle:
        center = bounds_environment.random_point()
        size = bounds_size.random_point()
        return cls(center=center, size=size)


@dataclass
class CylinderObstacle(Obstacle):
    radius: float

    def area(self) -> float:
        return math.pi * self.radius**2

    def is_inside(self, x: float, y: float) -> bool:
        d_x = abs(self.center[0] - x)
        d_y = abs(self.center[1] - y)
        d = np.linalg.norm([d_x, d_y])
        return bool(d <= self.radius)

    def to_dict(self) -> Dict:
        return {"type": "box", "center": self.center, "radius": self.radius}

    @classmethod
    def random(cls, bounds_environment: Bounds2D, bounds_size: Bounds2D) -> Obstacle:
        center = bounds_environment.random_point()
        radius = np.min(bounds_size.random_point())
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
