from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import diffmp


@dataclass
class Robot:
    start: List[float]
    goal: List[float]
    # _type: diffmp.dynamics.DynamicsBase

    @classmethod
    def from_dict(cls, data: Dict[str, List[float] | str]) -> Robot:
        assert isinstance(data["start"], list)
        assert isinstance(data["goal"], list)
        assert isinstance(data["type"], str)
        start = data["start"]
        goal = data["goal"]
        # _type = diffmp.dynamics.get_dynamics(data["type"], 5)

        # return cls(start=start, goal=goal, _type=_type)
        return cls(start=start, goal=goal)
