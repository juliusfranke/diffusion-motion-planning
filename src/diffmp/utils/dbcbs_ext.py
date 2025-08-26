from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import dbcbs_py
import msgpack
import numpy as np
import numpy.typing as npt
import yaml

import diffmp

DEFAULT_CONFIG = {
    "delta_0": 0.5,
    "delta_rate": 0.9,
    "num_primitives_0": 100,
    "num_primitives_rate": 1.5,
    "alpha": 0.5,
    "filter_duplicates": True,
    "heuristic1": "reverse-search",
    "heuristic1_delta": 1.0,
    "execute_joint_optimization": True,
    "execute_greedy_optimization": False,
    "heuristic1_num_primitives_0": 100,
    "always_add_node": False,
    "rewire": True,
    "residual_force": False,
    "suboptimality_factor": 1.3,
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
    discrete: list[Trajectory]
    optimized: list[Trajectory]
    runtime: float
    delta: float
    cost: float

    @classmethod
    def from_db_cbs(cls, db_cbs_solution: dbcbs_py.Result) -> Solution:
        discrete = [
            Trajectory.from_db_cbs(traj)
            for traj in db_cbs_solution.discrete.trajectories
        ]
        optimized = [
            Trajectory.from_db_cbs(traj)
            for traj in db_cbs_solution.optimized.trajectories
        ]
        runtime = db_cbs_solution.runtime
        delta = db_cbs_solution.delta

        cost = sum([d.actions.shape[0] / 10 for d in optimized])
        return cls(
            discrete=discrete,
            optimized=optimized,
            runtime=runtime,
            delta=delta,
            cost=cost,
        )


@dataclass
class Task:
    instance: diffmp.problems.Instance | dict
    config: dict
    timelimit_db_astar: float
    timelimit_db_cbs: float
    solutions: list[Solution]


def execute_task(task: Task) -> Task:
    tmp1 = tempfile.NamedTemporaryFile()
    tmp2 = tempfile.NamedTemporaryFile()
    if isinstance(task.instance, dict):
        instance_data = task.instance
    elif isinstance(task.instance.data, dict):
        instance_data = task.instance.data
    else:
        instance_data = task.instance.to_dict()
    try:
        results = dbcbs_py.db_ecbs(
            instance_data,
            tmp1.name,
            tmp2.name,
            task.config,
            task.timelimit_db_astar,
            task.timelimit_db_cbs,
        )
    except IndexError:
        results = []
    except RuntimeError:
        results = []
    task.solutions = [
        Solution.from_db_cbs(sol)
        for sol in results
        if len(sol.optimized.trajectories) > 0
    ]

    tmp1.close()
    tmp2.close()
    return task


def out_to_mps(
    actions: npt.NDArray, states: npt.NDArray, fmt: Literal["yaml", "msgpack"]
):
    length = actions.shape[1]
    if fmt == "yaml":
        mps = [
            {
                "start": states[0, i, :].tolist(),
                "goal": states[-1, i, :].tolist(),
                "actions": actions[:, i, :].tolist(),
                "states": states[:, i, :].tolist(),
            }
            for i in range(length)
        ]
    elif fmt == "msgpack":
        mps = {
            "data": [
                {
                    "actions": actions[:, i, :].tolist(),
                    "states": states[:, i, :].tolist(),
                }
                for i in range(length)
            ]
        }
    # mps = []
    # for i in range(length):
    #     mp = {
    #         "start": states[0, i, :].tolist(),
    #         "goal": states[-1, i, :].tolist(),
    #         "actions": actions[:, i, :].tolist(),
    #         "states": states[:, i, :].tolist(),
    #     }
    #     mps.append(mp)
    return mps


def export(
    model: diffmp.torch.Model,
    instance: diffmp.problems.Instance,
    out_path: Path,
    n_mp: int = 100,
    robot_idx: int = 0,
) -> None:
    fmt = "yaml" if out_path.suffix == ".yaml" else "msgpack"
    # time1 = time.perf_counter()
    out = diffmp.torch.sample(model, n_mp, instance, robot_idx).detach().cpu().numpy()
    if np.isnan(out).any():
        print("WARNING: Output contains NaN")
        out = np.nan_to_num(out)

    mps = model.dynamics.to_mp(out)
    dbcbs_mps = out_to_mps(mps["actions"], mps["states"], fmt)

    # time2 = time.perf_counter()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "yaml":
        with open(out_path, "w") as file:
            yaml.safe_dump(dbcbs_mps, file, default_flow_style=None)
    else:
        with open(out_path, "wb") as file:
            packed = msgpack.packb(dbcbs_mps)
            file.write(packed)  # type:ignore

    # time3 = time.perf_counter()

    # print(f"Gen: {time2 - time1} - Save: {time3 - time2}")


def distribute_motion_primitives(total_count, lengths):
    lengths = np.array(lengths, dtype=float)

    inv_lengths = 1 / lengths
    proportions = inv_lengths / inv_lengths.sum()

    counts = np.round(proportions * total_count).astype(int)

    diff = total_count - counts.sum()
    if diff != 0:
        counts[np.argmax(proportions)] += diff

    return dict(zip(lengths, counts))


def export_composite(
    composite_config: diffmp.torch.CompositeConfig,
    instance: diffmp.problems.Instance,
    out_path: Path,
    n_mp: int = 100,
) -> None:
    primitives = []
    distribution = composite_config.allocation
    if not isinstance(distribution, dict):
        distribution = distribute_motion_primitives(
            n_mp, [m.dynamics.timesteps for m in composite_config.models]
        )
    for model in composite_config.models:
        n_mp_model = distribution[model.dynamics.timesteps]
        out = diffmp.torch.sample(model, n_mp_model, instance).detach().cpu().numpy()
        mps = model.dynamics.to_mp(out)
        dbcbs_mps = out_to_mps(mps["actions"], mps["states"])
        primitives.extend(dbcbs_mps)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as file:
        yaml.safe_dump(primitives, file, default_flow_style=None)
