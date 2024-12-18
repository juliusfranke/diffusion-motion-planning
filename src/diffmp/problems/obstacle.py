from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
import math
from typing import Tuple, Dict


@dataclass
class Obstacle:
    center: Tuple[float, float]

    # @property
    # @abstractmethod
    def area(self) -> float: ...


@dataclass
class BoxObstacle(Obstacle):
    size: Tuple[float, float]

    # @property
    def area(self) -> float:
        return self.size[0] * self.size[1]


@dataclass
class CylinderObstacle(Obstacle):
    radius: float

    # @property
    def area(self) -> float:
        return math.pi * self.radius**2


class ObstacleType(Enum):
    box = BoxObstacle
    cylinder = CylinderObstacle


def obstacle_from_dict(data: Dict) -> Obstacle:
    obstacle_type = ObstacleType[data["type"]]
    match obstacle_type:
        case ObstacleType.box:
            return obstacle_type.value(data["center"], data["size"])
        case ObstacleType.cylinder:
            return obstacle_type.value(data["center"], data["radius"])
        case _:
            raise TypeError(f"Obstacle type {obstacle_type} does not exist")
