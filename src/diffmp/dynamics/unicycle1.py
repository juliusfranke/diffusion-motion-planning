import numpy as np
import numpy.typing as npt

from diffmp.dynamics.base import DynamicsBase
from diffmp.utils.config import (
    CalculatedParameter,
    DatasetParameter,
    get_default_parameter_set,
)


class UnicycleFirstOrder(DynamicsBase):
    def __init__(
        self,
        max_vel: float,
        min_vel: float,
        min_angular_vel: float,
        max_angular_vel: float,
        dt: float,
        timesteps: int,
        **kwargs,
    ) -> None:
        parameter_set = get_default_parameter_set()

        parameter_set.add_parameters(
            [
                DatasetParameter(
                    "actions",
                    2 * timesteps,
                    0,
                    [("actions", f"s_{i}") for i in range(timesteps)]
                    + [("actions", f"phi_{i}") for i in range(timesteps)],
                ),
                DatasetParameter(
                    "theta_0",
                    3 * timesteps,
                    0,
                    [("states", "theta_0")],
                ),
                DatasetParameter("theta_s", 1, 0, [("env", "theta_s")]),
                DatasetParameter("theta_q", 1, 0, [("env", "theta_g")]),
            ],
            condition=False,
        )
        parameter_set.add_parameters(
            [
                CalculatedParameter(
                    "states",
                    3 * timesteps,
                    0,
                    [(f"s_{i}", f"phi_{i}") for i in range(timesteps)],
                    ["actions", "theta_0"],
                    lambda x: x,
                ),
                CalculatedParameter(
                    "Theta_0",
                    2,
                    0,
                    [("Theta_0_x", "Theta_0_y")],
                    ["theta_0"],
                    lambda x: x,
                ),
            ],
            condition=True,
        )

        DynamicsBase.__init__(
            self,
            q=["x", "y", "theta"],
            u=["s", "phi"],
            dt=dt,
            u_lims={
                "min": {"s": min_vel, "phi": min_angular_vel},
                "max": {"s": max_vel, "phi": max_angular_vel},
            },
            parameter_set=parameter_set,
        )

    def _step(
        self, q: npt.NDArray[np.floating], u: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        next = q.copy()
        next[:, 0] += np.cos(q[:, 2]) * u[:, 0] * self.dt
        next[:, 1] += np.sin(q[:, 2]) * u[:, 0] * self.dt
        next[:, 2] += u[:, 1] * self.dt
        return next
