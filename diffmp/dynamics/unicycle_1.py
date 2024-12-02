from typing import Dict, List
from .base import DynamicsBase
import numpy as np


class UnicycleFirstOrder(DynamicsBase):
    def __init__(
        self,
        u_lims: None | Dict[str, List[float]] = {
            "lower": -0.5,
            "upper": 0.5,
        },
        dt: float = 0.1,
    ):
        DynamicsBase.__init__(
            self, q=["x", "y", "theta"], u=["s", "phi"], dt=dt, u_lims=u_lims
        )

    def _step(self, q: np.ndarray, u: np.ndarray) -> np.ndarray:
        next = q.copy()
        next[:, 0] += np.cos(q[:, 2]) * u[:, 0] * self.dt
        next[:, 1] += np.sin(q[:, 2]) * u[:, 0] * self.dt
        next[:, 2] += u[:, 1] * self.dt
        return next
