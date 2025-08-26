from functools import partial

import numpy as np
import numpy.typing as npt

import diffmp.utils as du

from .base import DynamicsBase


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
        timesteps: int,
        name: str,
        **kwargs,
    ) -> None:
        parameter_set = du.get_default_parameter_set()
        actions_columns = []
        for i in range(timesteps):
            actions_columns.append(("actions", f"a_{i}"))
            actions_columns.append(("actions", f"dphi_{i}"))

        parameter_set.add_parameters(
            [
                du.DatasetParameter("actions", 2 * timesteps, 0, actions_columns),
                du.DatasetParameter(
                    "theta_0",
                    1,
                    0,
                    [("states", "theta_0")],
                ),
                du.DatasetParameter(
                    "s_0",
                    1,
                    0,
                    [("states", "s_0")],
                ),
                du.DatasetParameter(
                    "phi_0",
                    1,
                    0,
                    [("states", "phi_0")],
                ),
                du.DatasetParameter("theta_s", 1, 0, [("env", "theta_s")]),
                du.DatasetParameter("theta_g", 1, 0, [("env", "theta_g")]),
                du.DatasetParameter("s_s", 1, 0, [("env", "s_s")]),
                du.DatasetParameter("s_g", 1, 0, [("env", "s_g")]),
                du.DatasetParameter("phi_s", 1, 0, [("env", "phi_s")]),
                du.DatasetParameter("phi_g", 1, 0, [("env", "phi_g")]),
            ],
            condition=False,
        )
        states_columns = []
        for i in range(timesteps + 1):
            states_columns.append(("actions", f"x_{i}"))
            states_columns.append(("actions", f"y_{i}"))
            states_columns.append(("actions", f"s_{i}"))
            states_columns.append(("actions", f"phi_{i}"))
        parameter_set.add_parameters(
            [
                du.CalculatedParameter(
                    "Theta_0",
                    2,
                    0,
                    [("states", "Theta_0_x"), ("states", "Theta_0_y")],
                    ["theta_0"],
                    partial(du.theta_to_Theta, col1="states"),
                    partial(du.Theta_to_theta, col1="states"),
                ),
                du.CalculatedParameter(
                    "Theta_s",
                    2,
                    0,
                    [("env", "Theta_s_x"), ("env", "Theta_s_y")],
                    ["theta_s"],
                    partial(du.theta_to_Theta, col1="env", i="s"),
                    partial(du.Theta_to_theta, col1="env", i="s"),
                ),
                du.CalculatedParameter(
                    "Theta_g",
                    2,
                    0,
                    [("env", "Theta_g_x"), ("env", "Theta_g_y")],
                    ["theta_g"],
                    partial(du.theta_to_Theta, col1="env", i="g"),
                    partial(du.Theta_to_theta, col1="env", i="g"),
                ),
            ],
            condition=False,
        )
        DynamicsBase.__init__(
            self,
            q=["x", "y", "theta", "s", "phi"],
            u=["a", "dphi"],
            dt=dt,
            q_lims={
                "min": {"theta": -np.pi, "s": min_vel, "phi": min_angular_vel},
                "max": {"theta": np.pi, "s": max_vel, "phi": max_angular_vel},
            },
            u_lims={
                "min": {"a": -max_acc_abs, "dphi": -max_angular_acc},
                "max": {"a": max_acc_abs, "dphi": max_angular_acc},
            },
            parameter_set=parameter_set,
            timesteps=timesteps,
            name=name,
        )

    def random_state(self, **kwargs) -> list[float]:
        state = super().random_state(**kwargs)
        state[3] = 0
        state[4] = 0
        return state

    def _step(
        self, q: npt.NDArray[np.floating], u: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        # V = np.zeros(q.shape)

        yaw = q[:, 2]
        vv = q[:, 3]
        w = q[:, 4]

        c = np.cos(yaw)
        s = np.sin(yaw)

        a = u[:, 0]
        w_dot = u[:, 1]

        V = np.zeros(q.shape)
        V[:, 0] = vv * c
        V[:, 1] = vv * s
        V[:, 2] = w
        V[:, 3] = a
        V[:, 4] = w_dot

        next = q + V * self.dt
        next[:, 2] = np.where(next[:, 2] < -np.pi, next[:, 2] + 2 * np.pi, next[:, 2])
        next[:, 2] = np.where(next[:, 2] > np.pi, next[:, 2] - 2 * np.pi, next[:, 2])
        next[:, 3] = np.clip(next[:, 3], self._q_lims["min"][3], self._q_lims["max"][3])
        next[:, 4] = np.clip(next[:, 4], self._q_lims["min"][4], self._q_lims["max"][4])

        return next

    def to_mp(self, data: npt.NDArray):
        # reg_cols, _ = self.parameter_set.get_columns()
        # columns = pd.MultiIndex.from_tuples(reg_cols)
        # df = pd.DataFrame(data, columns=columns)
        df = self.prepare_out(data)
        assert df.actions.shape[1] == self.timesteps * self.u_dim
        assert ("states", "phi_0") in df.columns
        assert ("states", "s_0") in df.columns
        assert ("states", "theta_0") in df.columns
        actions = np.swapaxes(
            df.actions.to_numpy().reshape(df.shape[0], self.timesteps, self.u_dim), 0, 1
        )
        actions = np.clip(actions, -0.25, 0.25)
        state = np.zeros((df.shape[0], self.q_dim))
        state[:, 2] = df.states.theta_0
        state[:, 3] = df.states.s_0
        state[:, 4] = df.states.phi_0
        states: list[npt.NDArray] = [state]
        for i in range(self.timesteps):
            state = self.step(state, actions[i], clip=True)
            states.append(state)

        mp = {"states": np.array(states), "actions": actions}
        return mp
