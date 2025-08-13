from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from meshlib.mrmeshpy import (
    AffineXf3f,
    Matrix3f,
    Mesh,
    Vector2f,
    Vector3f,
    makeCube,
    makePlane,
)
from .etc import Bounds, Dim


class Obstacle(ABC):
    mesh: Mesh
    dim: Dim

    def area(self) -> float:
        return (
            self.mesh.area()
            if self.dim == Dim.THREE_D
            else self.mesh.projArea(Vector3f(0, 0, 1) / 2)
        )

    def volume(self) -> float:
        return self.mesh.volume()

    @abstractmethod
    def to_dict(self) -> dict: ...

    @classmethod
    @abstractmethod
    def random(cls, center_bounds: Bounds, size_bounds: Bounds) -> Obstacle: ...


class BoxObstacle(Obstacle):
    center: Vector3f
    size: Vector3f

    def __init__(self, center: Vector3f, size: Vector3f) -> None:
        if size.z == 0:
            assert center.z == 0
            size.z = 1
            self.dim = Dim.TWO_D
        else:
            assert center.z > 0
            self.dim = Dim.THREE_D

        self.center = center
        self.size = size

        self.mesh = makeCube()
        scaling = AffineXf3f.xfAround(
            Matrix3f.scale(self.size), stable=Vector3f(0, 0, 0)
        )
        translation = AffineXf3f.translation(self.center)
        self.mesh.transform(scaling)
        self.mesh.transform(translation)

    def to_dict(self):
        center = list(self.center)
        size = list(self.size)
        if self.dim == Dim.TWO_D:
            center = center[:2]
            size = size[:2]

        return {"type": "Box", "center": center, "size": size}

    @classmethod
    def random(cls, center_bounds: Bounds, size_bounds: Bounds) -> BoxObstacle:
        center = center_bounds.random_point()
        size = size_bounds.random_point()
        return cls(center=center, size=size)


def box2d(center: Vector2f, size: Vector2f) -> Mesh:
    plane = makePlane()

    transform = AffineXf3f.xfAround(
        Matrix3f.scale(Vector3f(size.x, size.y, 1)),
        stable=Vector3f(0, 0, 0),
    ).translation(Vector3f(center.x, center.y))

    plane.transform(transform)
    return plane


def box3d(center: Vector3f, size: Vector3f) -> Mesh:
    box = makeCube()

    transform = AffineXf3f.xfAround(
        Matrix3f.scale(size), stable=Vector3f(0, 0, 0)
    ).translation(center)

    box.transform(transform)
    return box


def obstacle_from_yaml(data: dict) -> Mesh:
    ob_type = data["type"]
    center = data["center"]
    if ob_type == "box":
        size = data["size"]
        if len(center) == 2:
            return box2d(Vector2f(center), Vector2f(size))
        elif len(center) == 3:
            return box3d(Vector3f(center), Vector3f(size))
    raise NotImplementedError
