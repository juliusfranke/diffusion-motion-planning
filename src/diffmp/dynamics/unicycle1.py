from functools import partial
from typing import List
import numpy as np
import numpy.typing as npt

from diffmp.dynamics.base import DynamicsBase
from diffmp.utils.config import (
    CalculatedParameter,
    DatasetParameter,
    get_default_parameter_set,
)
from diffmp.utils import Theta_to_theta, theta_to_Theta
import pandas as pd


class UnicycleFirstOrder(DynamicsBase):
    def __init__(
        self,
        max_vel: float,
        min_vel: float,
        min_angular_vel: float,
        max_angular_vel: float,
        dt: float,
        timesteps: int,
        name: str,
        **kwargs,
    ) -> None:
        parameter_set = get_default_parameter_set()
        actions_set = []
        for i in range(timesteps):
            actions_set.append(("actions", f"s_{i}"))
            actions_set.append(("actions", f"phi_{i}"))

        parameter_set.add_parameters(
            [
                DatasetParameter("actions", 2 * timesteps, 0, actions_set),
                DatasetParameter(
                    "theta_0",
                    1,
                    0,
                    [("states", "theta_0")],
                ),
                DatasetParameter("theta_s", 1, 0, [("env", "theta_s")]),
                DatasetParameter("theta_g", 1, 0, [("env", "theta_g")]),
            ],
            condition=False,
        )
        parameter_set.add_parameters(
            [
                CalculatedParameter(
                    "Theta_0",
                    2,
                    0,
                    [("states", "Theta_0_x"), ("states", "Theta_0_y")],
                    ["theta_0"],
                    partial(theta_to_Theta, col1="states"),
                    partial(Theta_to_theta, col1="states"),
                ),
            ],
            condition=False,
        )

        DynamicsBase.__init__(
            self,
            q=["x", "y", "theta"],
            u=["s", "phi"],
            dt=dt,
            q_lims={"min": {"theta": -np.pi}, "max": {"theta": np.pi}},
            u_lims={
                "min": {"s": min_vel, "phi": min_angular_vel},
                "max": {"s": max_vel, "phi": max_angular_vel},
            },
            parameter_set=parameter_set,
            timesteps=timesteps,
            name=name,
        )

    def _step(
        self, q: npt.NDArray[np.floating], u: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        next = q.copy()
        next[:, 0] += np.cos(q[:, 2]) * u[:, 0] * self.dt
        next[:, 1] += np.sin(q[:, 2]) * u[:, 0] * self.dt
        theta = next[:, 2] + u[:, 1] * self.dt
        theta = np.where(theta < -np.pi, theta + 2 * np.pi, theta)
        theta = np.where(theta > np.pi, theta - 2 * np.pi, theta)
        next[:, 2] = theta

        return next

    def to_mp(self, data: npt.NDArray):
        reg_cols, _ = self.parameter_set.get_columns()
        columns = pd.MultiIndex.from_tuples(reg_cols)
        df = pd.DataFrame(data, columns=columns)
        assert df.actions.shape[1] == self.timesteps * self.u_dim
        has_theta_0 = ("states", "theta_0") in columns
        has_Theta_0 = ("states", "Theta_0_x") in columns and (
            "states",
            "Theta_0_y",
        ) in columns
        assert has_theta_0 or has_Theta_0

        if has_Theta_0:
            df = Theta_to_theta(df, col1="states")

        actions = np.swapaxes(
            df.actions.to_numpy().reshape(df.shape[0], self.timesteps, self.u_dim), 0, 1
        )
        actions = np.clip(actions, -0.5, 0.5)
        state = np.zeros((df.shape[0], self.q_dim))
        state[:, 2] = df.states.theta_0
        states: List[npt.NDArray] = [state]
        for i in range(self.timesteps):
            state = self.step(state, actions[i], clip=True)
            states.append(state)

        mp = {"states": np.array(states), "actions": actions}
        # breakpoint()
        return mp
