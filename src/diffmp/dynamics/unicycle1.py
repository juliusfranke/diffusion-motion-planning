from functools import partial

import numpy as np
import numpy.typing as npt
from meshlib.mrmeshpy import AffineXf3f, Matrix3f, Vector3f, makeCube

import diffmp.utils as du
from .base import DynamicsBase


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
        n_robots: int = 1,
        **kwargs,
    ) -> None:
        parameter_set = du.get_default_parameter_set()
        actions_set = []
        for i in range(timesteps):
            actions_set.append(("actions", f"s_{i}"))
            actions_set.append(("actions", f"phi_{i}"))
        r_cols = [f"robot_{i:03}" for i in range(n_robots)]
        regular_parameters: du.ParameterSeq = [
            du.DatasetParameter("actions", 0, actions_set),
            du.DatasetParameter(
                "theta_0",
                0,
                [("states", "theta_0")],
            ),
            du.CalculatedParameter(
                "Theta_0",
                0,
                [("states", "Theta_0_x"), ("states", "Theta_0_y")],
                ["theta_0"],
                partial(du.theta_to_Theta, col1="states"),
                partial(du.Theta_to_theta, col1="states"),
            ),
            du.DatasetParameter("theta_s", 0, [(r_col, "theta_s") for r_col in r_cols]),
            du.DatasetParameter("theta_g", 0, [(r_col, "theta_g") for r_col in r_cols]),
            du.DatasetParameter("x_s", 0, [(r_col, "x_s") for r_col in r_cols]),
            du.DatasetParameter("x_g", 0, [(r_col, "x_g") for r_col in r_cols]),
            du.DatasetParameter("y_s", 0, [(r_col, "y_s") for r_col in r_cols]),
            du.DatasetParameter("y_g", 0, [(r_col, "y_g") for r_col in r_cols]),
        ]
        condition_parameters: du.ParameterSeq = [
            du.CalculatedParameter(
                "start_T",
                0,
                [(r_col, "Theta_s_x") for r_col in r_cols]
                + [(r_col, "Theta_s_y") for r_col in r_cols],
                ["theta_s"],
                partial(du.theta_to_Theta, col1=r_cols, i="s"),
                partial(du.Theta_to_theta, col1=r_cols, i="s"),
            ),
            du.CalculatedParameter(
                "Theta_s",
                0,
                [(r_col, "Theta_s_x") for r_col in r_cols]
                + [(r_col, "Theta_s_y") for r_col in r_cols],
                ["theta_s"],
                partial(du.theta_to_Theta, col1=r_cols, i="s"),
                partial(du.Theta_to_theta, col1=r_cols, i="s"),
            ),
            du.CalculatedParameter(
                "Theta_g",
                0,
                [(r_col, "Theta_g_x") for r_col in r_cols]
                + [(r_col, "Theta_g_y") for r_col in r_cols],
                ["theta_g"],
                partial(du.theta_to_Theta, col1=r_cols, i="g"),
                partial(du.Theta_to_theta, col1=r_cols, i="g"),
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
            mesh=makeCube(
                size=Vector3f(size[0], size[1], 1),
                base=Vector3f(-size[0] / 2, -size[1] / 2, -0.5),
            ),
            n_robots=n_robots,
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
        xf.A = Matrix3f.rotation(Vector3f(0, 0, 1), state[2])
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
        states: list[npt.NDArray] = [state]
        for i in range(self.timesteps):
            state = self.step(state, actions[i], clip=True)
            states.append(state)

        mp = {"states": np.array(states), "actions": actions}
        return mp
