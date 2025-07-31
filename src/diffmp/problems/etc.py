from enum import Enum

import numpy as np
from meshlib.mrmeshpy import Vector2f, Vector3f


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
