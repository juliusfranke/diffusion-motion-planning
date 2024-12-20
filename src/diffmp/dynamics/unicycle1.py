import numpy as np
import numpy.typing as npt

from diffmp.dynamics.base import DynamicsBase


class UnicycleFirstOrder(DynamicsBase):
    def __init__(
        self,
        max_vel: float,
        min_vel: float,
        min_angular_vel: float,
        max_angular_vel: float,
        dt: float,
        **kwargs,
    ) -> None:
        DynamicsBase.__init__(
            self,
            q=["x", "y", "theta"],
            u=["s", "phi"],
            dt=dt,
            u_lims={
                "min": {"s": min_vel, "phi": min_angular_vel},
                "max": {"s": max_vel, "phi": max_angular_vel},
            },
        )

    def _step(
        self, q: npt.NDArray[np.floating], u: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        next = q.copy()
        next[:, 0] += np.cos(q[:, 2]) * u[:, 0] * self.dt
        next[:, 1] += np.sin(q[:, 2]) * u[:, 0] * self.dt
        next[:, 2] += u[:, 1] * self.dt
        return next
