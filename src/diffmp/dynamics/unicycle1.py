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
                DatasetParameter(name="actions", size=2 * timesteps, col_1="actions"),
                DatasetParameter(
                    name="theta_0", size=3 * timesteps, col_1="states", col_2="theta_0"
                ),
                DatasetParameter(name="theta_s", size=1, col_1="env", col_2="theta_s"),
                DatasetParameter(name="theta_q", size=1, col_1="env", col_2="theta_g"),
            ],
            condition=False,
        )
        parameter_set.add_parameters(
            [
                CalculatedParameter(
                    name="states",
                    size=3 * timesteps,
                    requires=["actions", "theta_0"],
                    to=self.step,
                ),
                CalculatedParameter(
                    name="Theta_0",
                    size=2,
                    requires=["theta_0"],
                    to=self.step,
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
