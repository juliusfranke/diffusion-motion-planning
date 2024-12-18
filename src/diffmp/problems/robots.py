from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import diffmp


@dataclass
class Robot:
    start: Tuple[float, float, float]
    goal: Tuple[float, float, float]
    _type: diffmp.dynamics.DynamicsBase

    @classmethod
    def from_dict(cls, data: Dict) -> Robot:
        start = data["start"]
        goal = data["goal"]
        _type = diffmp.dynamics.get_dynamics(data["type"])

        return cls(start=start, goal=goal, _type=_type)
