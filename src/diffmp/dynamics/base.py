from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from meshlib.mrmeshpy import (
    AffineXf3f,
    Mesh,
)

import diffmp.utils as du
import diffmp.problems as pb


class DynamicsBase(ABC):
    def __init__(
        self,
        dt: float,
        q: list[str],
        u: list[str],
        parameter_set: du.ParameterSet,
        timesteps: int,
        name: str,
        mesh: Mesh,
        n_robots: int = 0,
        q_lims: Optional[dict[str, dict[str, float]]] = None,
        u_lims: Optional[dict[str, dict[str, float]]] = None,
    ) -> None:
        self.dt = dt
        self.q = q
        self.q_dim: int = len(q)
        self.u = u
        self.u_dim: int = len(u)
        self.q_lims = q_lims
        self._q_lims = self._lims_to_vec(q_lims, self.q)
        self.u_lims = u_lims
        self._u_lims = self._lims_to_vec(u_lims, self.u)
        self.parameter_set = parameter_set
        self.timesteps = timesteps
        self.name = name
        self.mesh = mesh
        self.n_robots = n_robots
        self.dim: pb.Dim = pb.Dim.TWO_D

    def random_state(self, **kwargs) -> list[float]:
        state = np.zeros(self.q_dim)
        for i, name in enumerate(self.q):
            if value := kwargs.get(name):
                state[i] = value
                continue
            min = self._q_lims["min"][i]
            max = self._q_lims["max"][i]
            if np.isinf(min) or np.isinf(max):
                value = 0
            else:
                value = np.random.random() * (max - min) + min
            state[i] = value
        return state.tolist()

    def step(
        self,
        q: npt.NDArray[np.floating],
        u: npt.NDArray[np.floating],
        clip: bool = False,
    ) -> npt.NDArray[np.floating]:
        """Applies the action u to the starting state q.

        Args:
            q: Starting state.
            u: Action to perform.
            clip: Clips u and q to bounds

        Returns:
            The next state when performing the input action on the input state.
        """
        eps = 1e-3
        q = np.atleast_2d(q).astype(float)
        u = np.atleast_2d(u).astype(float)
        assert q.shape[1] == self.q_dim
        assert u.shape[1] == self.u_dim
        if clip:
            for i in range(self.u_dim):
                u[:, i] = np.clip(
                    u[:, i], self._u_lims["min"][i], self._u_lims["max"][i]
                )
        else:
            assert (self._u_lims["min"] - eps <= u).all() and (
                (u <= self._u_lims["max"] + eps).all()
            ).all(), "u is not within bounds"

        q_new = self._step(q, u)

        if clip:
            for i in range(self.q_dim):
                q_new[:, i] = np.clip(
                    q_new[:, i],
                    self._q_lims["min"][i] + eps,
                    self._q_lims["max"][i] - eps,
                )
        else:
            assert (self._q_lims["min"] <= q_new).all() and (
                (q_new <= self._q_lims["max"]).all()
            ).all(), (
                f"q is not within bounds {q_new[:, 2], np.max(q_new[:, 2]), np.min(q_new[:, 2]), q_new.shape, self._q_lims}"
            )
        return q_new

    def prepare_out(self, data: npt.NDArray) -> pd.DataFrame:
        reg_cols, _ = self.parameter_set.get_columns()
        columns = pd.MultiIndex.from_tuples(reg_cols)
        df = pd.DataFrame(data, columns=columns)
        for param in self.parameter_set.iter_regular():
            if isinstance(param, du.DatasetParameter):
                continue
            if param.fr is None:
                continue
            df = param.fr(df)
        return df

    @abstractmethod
    def tf_from_state(self, state: npt.NDArray[np.floating]) -> AffineXf3f: ...

    @abstractmethod
    def to_mp(self, data: npt.NDArray) -> dict: ...

    @abstractmethod
    def _step(
        self, q: npt.NDArray[np.floating], u: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Abstractmethod to be implemented.

        Args:
            q: Starting state.
            u: Action to perform.

        Returns:
            The next state when performing the input action on the input state.
        """
        ...

    @staticmethod
    def _lims_to_vec(
        lims: None | dict[str, dict[str, float]], names: list[str]
    ) -> dict[str, npt.NDArray[np.floating]]:
        if not lims:
            return {
                "min": np.array([-np.inf] * len(names)),
                "max": np.array([np.inf] * len(names)),
            }
        lower = np.array(
            [
                (
                    lims["min"][name]
                    if isinstance(lims["min"], dict) and name in lims["min"].keys()
                    else -np.inf
                )
                for name in names
            ]
        )
        upper = np.array(
            [
                (
                    lims["max"][name]
                    if isinstance(lims["max"], dict) and name in lims["max"].keys()
                    else np.inf
                )
                for name in names
            ]
        )
        return {"min": lower, "max": upper}

    def __str__(self):
        return f"{self.__class__.__name__}: u{self.u} -> q{self.q}"
