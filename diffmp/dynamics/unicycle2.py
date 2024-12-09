from .base import DynamicsBase
from nptyping import NDArray
import numpy as np


class UnicycleSecondOrder(DynamicsBase):
    def __init__(
        self,
        max_vel: float,
        min_vel: float,
        min_angular_vel: float,
        max_angular_vel: float,
        max_acc_abs: float,
        max_angular_acc: float,
        dt: float,
        **kwargs,
    ) -> None:
        DynamicsBase.__init__(
            self,
            q=["x", "y", "theta", "s", "phi"],
            u=["a", "dphi"],
            dt=dt,
            q_lims={
                "min": {"s": min_vel, "phi": min_angular_vel},
                "max": {"s": max_vel, "phi": max_angular_vel},
            },
            u_lims={
                "min": {"a": -max_acc_abs, "dphi": -max_angular_acc},
                "max": {"a": max_acc_abs, "dphi": max_angular_acc},
            },
        )

    def _step(self, q: NDArray, u: NDArray) -> NDArray:
        next = q.copy()
        next[:, 0] += np.cos(q[:, 2]) * q[:, 3] * self.dt
        next[:, 1] += np.sin(q[:, 2]) * q[:, 3] * self.dt
        next[:, 2] += q[:, 4] * self.dt
        next[:, 3] += u[:, 0] * self.dt
        next[:, 4] += u[:, 1] * self.dt
        return next
