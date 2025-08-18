from functools import partial
from typing import List
from meshlib.mrmeshpy import AffineXf3f, Matrix3f, Vector3f, makeCube
import numpy as np
import numpy.typing as npt

from diffmp.dynamics.base import DynamicsBase
from diffmp.utils import (
    CalculatedParameter,
    DatasetParameter,
    get_default_parameter_set,
    ParameterSeq,
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
        size: list[float],
        **kwargs,
    ) -> None:
        parameter_set = get_default_parameter_set()
        actions_set = []
        for i in range(timesteps):
            actions_set.append(("actions", f"s_{i}"))
            actions_set.append(("actions", f"phi_{i}"))

        regular_parameters: ParameterSeq = [
            DatasetParameter("actions", 2 * timesteps, 0, actions_set),
            DatasetParameter(
                "theta_0",
                1,
                0,
                [("states", "theta_0")],
            ),
            CalculatedParameter(
                "Theta_0",
                2,
                0,
                [("states", "Theta_0_x"), ("states", "Theta_0_y")],
                ["theta_0"],
                partial(theta_to_Theta, col1="states"),
                partial(Theta_to_theta, col1="states"),
            ),
            DatasetParameter("theta_s", 1, 0, [("env", "theta_s")]),
            DatasetParameter("theta_g", 1, 0, [("env", "theta_g")]),
        ]
        condition_parameters: ParameterSeq = [
            CalculatedParameter(
                "Theta_s",
                2,
                0,
                [("env", "Theta_s_x"), ("env", "Theta_s_y")],
                ["theta_s"],
                partial(theta_to_Theta, col1="env", i="s"),
                partial(Theta_to_theta, col1="env", i="s"),
            ),
            CalculatedParameter(
                "Theta_g",
                2,
                0,
                [("env", "Theta_g_x"), ("env", "Theta_g_y")],
                ["theta_g"],
                partial(theta_to_Theta, col1="env", i="g"),
                partial(Theta_to_theta, col1="env", i="g"),
            ),
        ]
        parameter_set.add_parameters(
            regular_parameters,
            condition=False,
        )
        parameter_set.add_parameters(
            condition_parameters,
            condition=True,
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
            mesh=makeCube(size=Vector3f(size[0], size[1], 1))
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

    def tf_from_state(self, state: npt.NDArray[np.floating]) -> AffineXf3f:
        xf = AffineXf3f()
        xf.b = Vector3f(state[0], state[1], 0)
        xf.A = Matrix3f.rotation(Vector3f(0,0,1), state[2])
        return xf

    def to_mp(self, data: npt.NDArray):
        df = self.prepare_out(data)
        assert df.actions.shape[1] == self.timesteps * self.u_dim
        assert ("states", "theta_0") in df.columns

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
        return mp
