from functools import partial
from typing import List, Sequence
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


class CarWithTrailers(DynamicsBase):
    def __init__(
        self,
        max_vel: float,
        min_vel: float,
        max_angular_vel: float,
        max_steering_abs: float,
        l: float,
        num_trailers: float,
        hitch_lengths: List[float],
        dt: float,
        timesteps: int,
        name: str,
        **kwargs,
    ) -> None:
        self.l = l
        self.num_trailers = num_trailers
        self.hitch_lengths = hitch_lengths
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
            DatasetParameter("theta_2_s", 1, 0, [("env", "theta_2_s")]),
            DatasetParameter("theta_2_g", 1, 0, [("env", "theta_2_g")]),
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
        q = ["x", "y", "theta"]
        if num_trailers == 1:
            regular_parameters.append(
                DatasetParameter(
                    "theta_2_0",
                    1,
                    0,
                    [("states", "theta_2_0")],
                )
            )
            regular_parameters.append(
                CalculatedParameter(
                    "Theta_2_0",
                    2,
                    0,
                    [("states", "Theta_2_0_x"), ("states", "Theta_2_0_y")],
                    ["theta_2_0"],
                    partial(theta_to_Theta, col1="states", i="2_0"),
                    partial(Theta_to_theta, col1="states", i="2_0"),
                ),
            )
            condition_parameters.append(
                CalculatedParameter(
                    "Theta_2_s",
                    2,
                    0,
                    [("env", "Theta_2_s_x"), ("env", "Theta_2_s_y")],
                    ["theta_2_s"],
                    partial(theta_to_Theta, col1="env", i="2_s"),
                    partial(Theta_to_theta, col1="env", i="2_s"),
                )
            )
            condition_parameters.append(
                CalculatedParameter(
                    "Theta_2_g",
                    2,
                    0,
                    [("env", "Theta_2_g_x"), ("env", "Theta_2_g_y")],
                    ["theta_2_g"],
                    partial(theta_to_Theta, col1="env", i="2_g"),
                    partial(Theta_to_theta, col1="env", i="2_g"),
                )
            )
            q.append("theta_2")
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
            q=q,
            u=["s", "phi"],
            dt=dt,
            q_lims={"min": {"theta": -np.pi}, "max": {"theta": np.pi}},
            u_lims={
                "min": {"s": min_vel, "phi": -max_steering_abs},
                "max": {"s": max_vel, "phi": max_steering_abs},
            },
            parameter_set=parameter_set,
            timesteps=timesteps,
            name=name,
        )

    def random_state(self, **kwargs) -> List[float]:
        state = super().random_state(**kwargs)
        if self.num_trailers == 1:
            state[3] = state[2]
        return state

    def _step(
        self, q: npt.NDArray[np.floating], u: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        V = np.zeros(q.shape)
        v = u[:, 0]
        phi = u[:, 1]
        yaw = q[:, 2]

        c = np.cos(yaw)
        s = np.sin(yaw)
        V[:, 0] = v * c
        V[:, 1] = v * s
        V[:, 2] = v / self.l * np.tan(phi)
        if self.num_trailers:
            theta_dot = v / self.hitch_lengths[0]
            theta_dot *= np.sin(q[:, 2] - q[:, 3])
            V[:, 3] = theta_dot
        next = q + V * self.dt
        if self.num_trailers:
            next[:, 3] = np.where(
                next[:, 3] < -np.pi, next[:, 3] + 2 * np.pi, next[:, 3]
            )
            next[:, 3] = np.where(
                next[:, 3] > np.pi, next[:, 3] - 2 * np.pi, next[:, 3]
            )
        next[:, 2] = np.where(next[:, 2] < -np.pi, next[:, 2] + 2 * np.pi, next[:, 2])
        next[:, 2] = np.where(next[:, 2] > np.pi, next[:, 2] - 2 * np.pi, next[:, 2])

        return next

    def to_mp(self, data: npt.NDArray):
        df = self.prepare_out(data)
        assert df.actions.shape[1] == self.timesteps * self.u_dim
        assert ("states", "theta_0") in df.columns
        assert ("states", "theta_2_0") in df.columns
        actions = np.swapaxes(
            df.actions.to_numpy().reshape(df.shape[0], self.timesteps, self.u_dim), 0, 1
        )
        # actions = np.clip(actions, , 0.25)
        for i in range(self.u_dim):
            actions[:, i] = np.clip(
                actions[:, i], self._u_lims["min"][i], self._u_lims["max"][i]
            )
        state = np.zeros((df.shape[0], self.q_dim))
        state[:, 2] = df.states.theta_0
        if self.num_trailers == 1:
            state[:, 3] = df.states.theta_2_0
        states: List[npt.NDArray] = [state]
        for i in range(self.timesteps):
            state = self.step(state, actions[i], clip=True)
            states.append(state)

        mp = {"states": np.array(states), "actions": actions}
        return mp
