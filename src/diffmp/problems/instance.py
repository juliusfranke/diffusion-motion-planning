from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import diffmp

from .environment import Environment
from .robots import Robot

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class Instance:
    environment: Environment
    robots: List[Robot]
    data: Optional[Dict] = None
    name: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[Any, Any], name: Optional[str] = None) -> Instance:
        env = Environment.from_dict(data["environment"])
        robots = [Robot.from_dict(robot_data) for robot_data in data["robots"]]
        return cls(env, robots, data, name)

    @classmethod
    def from_yaml(cls, path: Path) -> Instance:
        data = diffmp.utils.load_yaml(path)
        name = path.stem
        return cls.from_dict(data, name = name)
