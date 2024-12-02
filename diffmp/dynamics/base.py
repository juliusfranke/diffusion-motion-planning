from typing import Dict, List
import numpy as np


class DynamicsBase:
    def __init__(
        self,
        dt: float,
        q: List[str],
        u: List[str],
        q_lims: None | Dict[str, float | List[float]] = None,
        u_lims: None | Dict[str, float | List[float]] = None,
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

    def step(self, q: np.ndarray, u: np.ndarray):
        q = np.atleast_2d(q).astype(float)
        u = np.atleast_2d(u).astype(float)
        assert q.shape[1] == self.q_dim
        assert u.shape[1] == self.u_dim
        assert (self._q_lims["lower"] <= q).all() and (
            q <= self._q_lims["upper"]
        ).all(), f"q:{q} is not within bounds"
        assert (self._u_lims["lower"] <= u).all() and (
            u <= self._u_lims["upper"]
        ).all(), f"u:{u} is not within bounds"
        return self._step(q, u)

    def _step(self, q: np.ndarray, u: np.ndarray) -> np.ndarray: ...

    def _lims_to_vec(self, lims, names):
        if not lims:
            return {
                "lower": np.array([-np.inf] * len(names)),
                "upper": np.array([np.inf] * len(names)),
            }
        lower = np.array(
            [
                lims["lower"][_u] if isinstance(lims["lower"], dict) else lims["lower"]
                for _u in self.u
            ]
        )
        upper = np.array(
            [
                lims["upper"][_u] if isinstance(lims["upper"], dict) else lims["upper"]
                for _u in self.u
            ]
        )
        return {"lower": lower, "upper": upper}

    def __str__(self):
        return f"{self.__class__.__name__}: u{self.u} -> q{self.q}"
