from typing import Optional
import numpy as np
import numpy.typing as npt

from meshlib.mrmeshpy import (
    AffineXf3f,
    Mesh,
    SphereParams,
    Vector3f,
    makeCube,
    makeSphere,
)

import diffmp
from diffmp.utils.config import get_default_parameter_set

from . import DynamicsBase


class Integrator2_3D(DynamicsBase):
    def __init__(
        self,
        timesteps: int,
        name: str,
        shape: str,
        radius: float,
        dt: float = 0.1,
        **kwargs,
    ) -> None:
        parameter_set = get_default_parameter_set()
        q = ["x", "y", "z", "vx", "vy", "vz"]
        u = ["ax", "ay", "az"]
        assert shape == "sphere"
        mesh = makeSphere(SphereParams(radius, 50))
        q_lims = {}
        u_lims = {}
        super().__init__(dt, q, u, parameter_set, timesteps, name, mesh, q_lims, u_lims)

    def _step(
        self, q: npt.NDArray[np.floating], u: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        # TODO Implement
        return q

    def tf_from_state(self, state: npt.NDArray[np.floating]) -> AffineXf3f:
        xf = AffineXf3f()
        xf.b = Vector3f(state[0], state[1], state[2])
        return xf

    def to_mp(self, data: npt.NDArray) -> dict:
        # TODO Implement
        return {}
