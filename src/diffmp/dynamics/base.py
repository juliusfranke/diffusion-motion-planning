from abc import ABC, abstractmethod
import pandas as pd
import diffmp
from typing import Dict, List

import numpy as np
import numpy.typing as npt


class DynamicsBase(ABC):
    def __init__(
        self,
        dt: float,
        q: List[str],
        u: List[str],
        parameter_set: diffmp.utils.ParameterSet,
        timesteps: int,
        q_lims: None | Dict[str, Dict[str, float]] = None,
        u_lims: None | Dict[str, Dict[str, float]] = None,
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

    def random_state(self, **kwargs) -> List[float]:
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
        q = np.atleast_2d(q).astype(float)
        u = np.atleast_2d(u).astype(float)
        assert q.shape[1] == self.q_dim
        assert u.shape[1] == self.u_dim
        if clip:
            for i in range(self.u_dim):
                u[:, i] = np.clip(
                    u[:, i], self._u_lims["min"][i], self._u_lims["max"][i]
                )
        assert (self._u_lims["min"] <= u).all() and (
            (u <= self._u_lims["max"]).all()
        ).all(), "u is not within bounds"

        q_new = self._step(q, u)

        if clip:
            for i in range(self.q_dim):
                q[:, i] = np.clip(
                    q[:, i], self._q_lims["min"][i], self._q_lims["max"][i]
                )
        assert (self._q_lims["min"] <= q_new).all() and (
            (q_new <= self._q_lims["max"]).all()
        ).all(), "q is not within bounds"
        return q_new

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

    @abstractmethod
    def to_mp(self, data: npt.NDArray) -> Dict: ...

    @staticmethod
    def _lims_to_vec(
        lims: None | Dict[str, Dict[str, float]], names: List[str]
    ) -> Dict[str, npt.NDArray[np.floating]]:
        if not lims:
            return {
                "min": np.array([-np.inf] * len(names)),
                "max": np.array([np.inf] * len(names)),
            }
        lower = np.array(
            [
                lims["min"][name]
                if isinstance(lims["min"], dict) and name in lims["min"].keys()
                else -np.inf
                for name in names
            ]
        )
        upper = np.array(
            [
                lims["max"][name]
                if isinstance(lims["max"], dict) and name in lims["max"].keys()
                else np.inf
                for name in names
            ]
        )
        return {"min": lower, "max": upper}

    def __str__(self):
        return f"{self.__class__.__name__}: u{self.u} -> q{self.q}"
