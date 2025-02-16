from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import diffmp
from diffmp.problems.environment import Environment


@dataclass
class Robot:
    start: List[float]
    goal: List[float]
    dynamics: str

    def to_dict(self) -> Dict:
        return {"start": self.start, "goal": self.goal, "type": self.dynamics}

    @classmethod
    def from_dict(cls, data: Dict[str, List[float] | str]) -> Robot:
        assert isinstance(data["start"], list)
        assert isinstance(data["goal"], list)
        assert isinstance(data["type"], str)
        start = data["start"]
        goal = data["goal"]
        dynamics = data["type"]
        return cls(start=start, goal=goal, dynamics=dynamics)

    @classmethod
    def random(cls, env: Environment, dynamics_type: str) -> Optional[Robot]:
        dynamics = diffmp.dynamics.get_dynamics(dynamics_type, 1)
        start_free = env.random_free(clearance=0.26)
        if start_free is None:
            return None
        x_start, y_start = start_free
        start = dynamics.random_state(x=x_start, y=y_start)
        goal_free = env.random_free(clearance=0.26)
        if goal_free is None:
            return None
        x_goal, y_goal = goal_free
        goal = dynamics.random_state(x=x_goal, y=y_goal)

        return cls(start=start, goal=goal, dynamics=dynamics_type)
