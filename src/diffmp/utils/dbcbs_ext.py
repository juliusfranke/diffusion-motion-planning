from __future__ import annotations
import pandas as pd
import diffmp
import yaml
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import List, Dict
import tempfile
import dbcbs_py


DEFAULT_CONFIG = {
    "delta_0": 0.5,
    "delta_rate": 0.9,
    "num_primitives_0": 200,
    "num_primitives_rate": 1.5,
    "alpha": 0.5,
    "filter_duplicates": True,
    "heuristic1": "reverse-search",
    "heuristic1_delta": 1.0,
}


@dataclass
class Trajectory:
    actions: npt.NDArray
    states: npt.NDArray

    @classmethod
    def from_db_cbs(cls, db_cbs_traj: dbcbs_py.Trajectory) -> Trajectory:
        actions = np.array(db_cbs_traj.actions)
        states = np.array(db_cbs_traj.states)
        return cls(actions=actions, states=states)


@dataclass
class Solution:
    discrete: Trajectory
    optimized: Trajectory
    runtime: float
    delta: float
    cost: float

    @classmethod
    def from_db_cbs(cls, db_cbs_solution: dbcbs_py.Result) -> Solution:
        discrete = Trajectory.from_db_cbs(db_cbs_solution.discrete.trajectories[0])
        optimized = Trajectory.from_db_cbs(db_cbs_solution.optimized.trajectories[0])
        runtime = db_cbs_solution.runtime
        delta = db_cbs_solution.delta
        cost = discrete.actions.shape[0] / 10
        return cls(
            discrete=discrete,
            optimized=optimized,
            runtime=runtime,
            delta=delta,
            cost=cost,
        )


@dataclass
class Task:
    instance: diffmp.problems.Instance
    config: Dict
    timelimit_db_astar: float
    timelimit_db_cbs: float
    solutions: List[Solution]


def execute_task(task: Task):
    tmp1 = tempfile.NamedTemporaryFile()
    tmp2 = tempfile.NamedTemporaryFile()
    if isinstance(task.instance.data, Dict):
        instance_data = task.instance.data
    else:
        instance_data = task.instance.to_dict()
    results = dbcbs_py.db_cbs(
        instance_data,
        tmp1.name,
        tmp2.name,
        task.config,
        task.timelimit_db_astar,
        task.timelimit_db_cbs,
    )
    task.solutions = [Solution.from_db_cbs(sol) for sol in results]

    tmp1.close()
    tmp2.close()
    return task


def out_to_mps(actions: npt.NDArray, states: npt.NDArray, out_path: Path):
    length = actions.shape[1]
    mps = []
    # breakpoint()
    for i in range(length):
        mp = {
            # "time_stamp": 0,
            # "cost": 0.5,
            # "feasible": 1,
            # "traj_feas": 1,
            # "goal_feas": 1,
            # "start_feas": 1,
            # "col_feas": 1,
            # "x_bounds_feas": 1,
            # "u_bounds_feas": 1,
            "start": states[0, i, :].tolist(),
            "goal": states[-1, i, :].tolist(),
            # "max_jump": 0,
            # "max_collision": 0,
            # "start_distance": 0,
            # "goal_distance": 0,
            # "x_bound_distance": 0,
            # "u_bound_distance": 0,
            # "num_states": 6,
            # "num_actions": 5,
            # "info": "i:480-s:0-l:5",
            "actions": actions[:, i, :].tolist(),
            "states": states[:, i, :].tolist(),
        }
        mps.append(mp)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as file:
        yaml.safe_dump(mps, file, default_flow_style=None)


def export(
    model: diffmp.torch.Model,
    instance: diffmp.problems.Instance,
    out_path: Path,
    n_mp: int = 100,
) -> None:
    out = diffmp.torch.sample(model, n_mp, instance).detach().cpu().numpy()

    mps = model.dynamics.to_mp(out)
    out_to_mps(mps["actions"], mps["states"], out_path)
