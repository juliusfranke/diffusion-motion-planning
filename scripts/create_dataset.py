from collections import defaultdict
import multiprocessing as mp
import os
import random
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dbcbs_py
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

import diffmp


@dataclass
class Solution:
    actions: npt.NDArray
    states: npt.NDArray


@dataclass
class Task:
    instance: diffmp.problems.Instance
    config: Dict
    timelimit_db_astar: float
    timelimit_db_cbs: float
    runtime: Optional[float] = None
    discrete_solution: Optional[Solution] = None
    optimized_solution: Optional[Solution] = None


def execute_task(task: Task):
    tmp1 = tempfile.NamedTemporaryFile()
    tmp2 = tempfile.NamedTemporaryFile()
    assert isinstance(task.instance.data, Dict)

    result = dbcbs_py.db_cbs(
        task.instance.data,
        tmp1.name,
        tmp2.name,
        task.config,
        task.timelimit_db_astar,
        task.timelimit_db_astar,
    )
    task.runtime = result.runtime

    if len(result.discrete.trajectories) > 0:
        d_actions = np.array(result.discrete.trajectories[0].actions)
        d_states = np.array(result.discrete.trajectories[0].states)
        task.discrete_solution = Solution(d_actions, d_states)

    if len(result.optimized.trajectories) > 0:
        o_actions = np.array(result.discrete.trajectories[0].actions)
        o_states = np.array(result.discrete.trajectories[0].states)
        task.optimized_solution = Solution(o_actions, o_states)

    tmp1.close()
    tmp2.close()
    return task


def split_solution(task: Task, lengths: List[int], decimals: int = 2):
    assert isinstance(task.optimized_solution, Solution)
    primitives = {
        length: {"actions": [], "states": [], "cost": []} for length in lengths
    }
    i = 0
    min_len = min(lengths)
    len_solution = len(task.optimized_solution.actions)
    cost = len_solution / 10
    len_per = len_solution / len(lengths)
    mp_lens = random.choices(
        lengths,
        weights=[len_per / length for length in lengths],
        k=len_solution // min_len,
    )
    dataframes = {}

    for mp_len in mp_lens:
        if len_solution - i < mp_len:
            if mp_len == min_len:
                break
            continue
        primitives[mp_len]["actions"].append(
            task.optimized_solution.actions[i : i + mp_len].flatten()
        )
        primitives[mp_len]["states"].append(
            task.optimized_solution.states[i : i + mp_len].flatten()
        )
        primitives[mp_len]["cost"].append(cost)
        i += mp_len

    for length in lengths:
        actions = np.round(np.array(primitives[length]["actions"]), decimals=decimals)
        states = np.round(np.array(primitives[length]["states"]), decimals=decimals)
        # cost = np.round(np.array(primitives[length]["cost"]), decimals=decimals)
        cost = np.ones((actions.shape[0], 1)) * cost

        theta_0 = np.atleast_2d(np.round(states[:, 2], decimals=decimals))
        array = np.concatenate([actions, theta_0.T, cost], axis=1)
        # array = np.concatenate([array, cost], axis=1)
        df = pd.DataFrame(array)
        df["instance"] = task.instance.name
        dataframes[length] = df
        # cols.extend(dynamics.parameter_set[param.name].cols)

    return dataframes


def tasks_to_mp(tasks: List[Task], lengths: List[int]):
    dataframes = defaultdict(list)
    primitives = {}
    for task in tasks:
        task_dataframes = split_solution(task, lengths)
        for length in lengths:
            dataframes[length].append(task_dataframes[length])
    for length in lengths:
        columns = []
        for i in range(length):
            columns.append(("actions", f"s_{i}"))
            columns.append(("actions", f"phi_{i}"))
        columns.append(("states", "theta_0"))
        columns.append(("misc", "cost"))
        columns.append(("env", "name"))
        columns.append(("misc", "count"))
        df = pd.concat(dataframes[length])
        multiindex = pd.MultiIndex.from_tuples(columns)
        df = df.groupby(df.columns.tolist(), as_index=False).size()
        breakpoint()
        df.columns = multiindex
        
        primitives[length] = df


    return primitives

def instances_to_df(instances: List[diffmp.problems.Instance]):
    data = defaultdict(list)
    for instance in instances:
        data[("env", "name")].append(instance.name)
        data[("env", "area")].append(instance.environment.area)
        data[("env", "area_blocked")].append(instance.environment.area_blocked)
        data[("env", "area_free")].append(instance.environment.area_free)
        data[("env", "env_width")].append(instance.environment.env_width)
        data[("env", "env_height")].append(instance.environment.env_height)
        data[("env", "n_obstacles")].append(instance.environment.n_obstacles)
        data[("env", "p_obstacles")].append(instance.environment.p_obstacles)
        data[("env", "theta_s")].append(instance.robots[0].start[2])
        data[("env", "theta_g")].append(instance.robots[0].goal[2])

    df = pd.DataFrame(data)
    multiindex = pd.MultiIndex.from_tuples(data.keys())
    df.columns = multiindex
    return df

def main():
    trials = 100
    decimals = 2
    timelimit_db_astar = 1000
    timelimit_db_cbs = 3000
    lengths = [5]
    results = {length: {"actions": [], "states": [], "cost": []} for length in lengths}
    arrays = {}
    data = {
        "delta_0": 0.5,
        "delta_rate": 0.9,
        "num_primitives_0": 100,
        "num_primitives_rate": 1.5,
        "alpha": 0.5,
        "filter_duplicates": True,
        "heuristic1": "reverse-search",
        "heuristic1_delta": 1.0,
        # "mp_path": "../new_format_motions/unicycle1_v0/motions.yaml",
        "mp_path": "../new_format_motions/unicycle1_v0/unicycle1_v0.msgpack",
    }
    bugtrap = diffmp.problems.Instance.from_yaml(Path("../example/bugtrap.yaml"))
    instances = [bugtrap]
    configurations = [data]
    # exec_task = partial(execute_task, env_dict=environments)
    tasks = []
    for instance in instances:
        for configuration in configurations:
            tasks += [
                Task(
                    instance,
                    configuration,
                    timelimit_db_astar,
                    timelimit_db_cbs,
                )
                for _ in range(trials)
            ]
    pbar = tqdm(total=trials * len(instances) * len(configurations))
    solved_tasks = []
    with mp.Pool(6) as p:
        for result in p.imap_unordered(execute_task, tasks):
            if isinstance(result.optimized_solution, Solution):
                solved_tasks.append(result)
            pbar.update()

    pbar.close()
    primitives = tasks_to_mp(solved_tasks, lengths)
    instances_df = instances_to_df(instances)

    breakpoint()
    for length, df in primitives.items():
        dataset = df.merge(instances_df).drop(columns=("env", "name"))
        df.to_parquet(f"data/training_datasets/{length}.parquet")


if __name__ == "__main__":
    main()
