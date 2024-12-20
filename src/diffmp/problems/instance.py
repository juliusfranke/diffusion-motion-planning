from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .environment import Environment
from .robots import Robot


@dataclass
class Instance:
    environment: Environment
    robots: List[Robot]

    @classmethod
    def from_dict(cls, data: Dict[Any, Any]) -> Instance:
        env = Environment.from_dict(data["environment"])
        robots = [Robot.from_dict(robot_data) for robot_data in data["robots"]]
        return cls(env, robots)
