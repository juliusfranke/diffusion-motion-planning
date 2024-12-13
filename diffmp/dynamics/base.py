from typing import Dict, List

import numpy as np
from numpy.typing import NDArray


class DynamicsBase:
    def __init__(
        self,
        dt: float,
        q: List[str],
        u: List[str],
        q_lims: None | Dict[str, Dict[str, float]] = None,
        u_lims: None | Dict[str, Dict[str, float]] = None,
    ):
        self.dt = dt
        self.q = q
        self.q_dim: int = len(q)
        self.u = u
        self.u_dim: int = len(u)
        self.q_lims = q_lims
        self._q_lims = self._lims_to_vec(q_lims, self.q)
        self.u_lims = u_lims
        self._u_lims = self._lims_to_vec(u_lims, self.u)

    def step(self, q: NDArray, u: NDArray):
        q = np.atleast_2d(q).astype(float)
        u = np.atleast_2d(u).astype(float)
        assert q.shape[1] == self.q_dim
        assert u.shape[1] == self.u_dim
        assert (self._u_lims["min"] <= u).all() and (
            (u <= self._u_lims["max"]).all()
        ).all(), f"u:{u} is not within bounds"
        q_new = self._step(q, u)
        if not (self._q_lims["min"] <= q_new).all():
            breakpoint()
        if not (self._q_lims["max"] >= q_new).all():
            breakpoint()
        assert (self._q_lims["min"] <= q_new).all() and (
            (q_new <= self._q_lims["max"]).all()
        ).all(), f"q:{q} is not within bounds"
        return q_new

    def _step(self, q: NDArray, u: NDArray) -> NDArray: ...

    @staticmethod
    def _lims_to_vec(
        lims: None | Dict[str, Dict[str, float]], names: List[str]
    ) -> Dict[str, NDArray]:
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
