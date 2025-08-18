from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from meshlib.mrmeshpy import AffineXf3f, Matrix3f, Vector3f

import diffmp
from diffmp.problems.etc import plot_3dmesh, plot_3dmesh_to_2d

if TYPE_CHECKING:
    from diffmp.dynamics import DynamicsBase
    from diffmp.problems.environment import Environment
    from diffmp.problems.etc import Dim


@dataclass
class Robot:
    start: list[float]
    goal: list[float]
    dynamics: DynamicsBase
    dim: Dim

    def plot2d(self, ax: Axes):
        xf_start = self.dynamics.tf_from_state(np.array(self.start))
        xf_goal = self.dynamics.tf_from_state(np.array(self.goal))

        plot_3dmesh_to_2d(self.dynamics.mesh, xf=xf_start, ax=ax, face_color="red")
        plot_3dmesh_to_2d(self.dynamics.mesh, xf=xf_goal, ax=ax, face_color="green")

    def plot3d(self, ax: Axes):
        xf_start = self.dynamics.tf_from_state(np.array(self.start))
        xf_goal = self.dynamics.tf_from_state(np.array(self.goal))

        plot_3dmesh(self.dynamics.mesh, xf=xf_start, ax=ax, face_color="red")
        plot_3dmesh(self.dynamics.mesh, xf=xf_goal, ax=ax, face_color="green")

    def plot(self, ax: Optional[Axes] = None) -> None:
        if ax is None:
            fig, ax = plt.subplots(1)
        assert isinstance(ax, Axes)
        ar_len = 0.5
        dx_s = ar_len * np.cos(self.start[2])
        dy_s = ar_len * np.sin(self.start[2])
        ax.arrow(
            self.start[0], self.start[1], dx_s, dy_s, fc="red", ec="red", head_width=0.2
        )
        dx_g = ar_len * np.cos(self.goal[2])
        dy_g = ar_len * np.sin(self.goal[2])
        ax.arrow(
            self.goal[0],
            self.goal[1],
            dx_g,
            dy_g,
            fc="green",
            ec="green",
            head_width=0.2,
        )

    def to_dict(self) -> dict:
        return {"start": self.start, "goal": self.goal, "type": self.dynamics.name}

    @classmethod
    def from_dict(cls, data: dict[str, list[float] | str]) -> Robot:
        assert isinstance(data["start"], list)
        assert isinstance(data["goal"], list)
        assert isinstance(data["type"], str)
        start = data["start"]
        goal = data["goal"]
        dim = Dim.TWO_D if len(data["start"]) == 2 else Dim.THREE_D
        dynamics = diffmp.dynamics.get_dynamics(data["type"], 1)
        return cls(start=start, goal=goal, dynamics=dynamics, dim=dim)

    @classmethod
    def random(cls, env: Environment, dynamics_type: str, dim: Dim) -> Optional[Robot]:
        dynamics = diffmp.dynamics.get_dynamics(dynamics_type, 1)

        start_free = env.random_free(dynamics)
        if start_free is None:
            return None

        goal_free = env.random_free(dynamics)
        if goal_free is None:
            return None

        print(start_free)
        print(goal_free)

        return cls(start=start_free, goal=goal_free, dynamics=dynamics, dim=dim)
