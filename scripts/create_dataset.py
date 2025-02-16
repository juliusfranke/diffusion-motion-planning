from __future__ import annotations
from collections import defaultdict
import multiprocessing as mp
import sys
import random
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, List

import dbcbs_py
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

import diffmp


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


def split_solution(task: Task, lengths: List[int]):
    primitives = {
        length: {"actions": [], "states": [], "rel_l": [], "cost": [], "delta": []}
        for length in lengths
    }
    min_len = min(lengths)
    dataframes = {}
    for solution in task.solutions:
        i = 0
        rel_l_temp = defaultdict(list)
        len_solution = len(solution.optimized.actions)
        delta = solution.delta
        cost = len_solution / 10
        len_per = len_solution / len(lengths)
        mp_lens = random.choices(
            lengths,
            weights=[len_per / length for length in lengths],
            k=len_solution // min_len,
        )

        for mp_len in mp_lens:
            if len_solution - i < mp_len:
                if mp_len == min_len:
                    primitives[mp_len]["actions"].append(
                        solution.optimized.actions[-mp_len:].flatten()
                    )
                    primitives[mp_len]["states"].append(
                        solution.optimized.states[-mp_len:].flatten()
                    )
                    primitives[mp_len]["cost"].append(cost)
                    primitives[mp_len]["rel_l"].append(i / len_solution)
                    primitives[mp_len]["delta"].append(delta)
                    break
                continue
            primitives[mp_len]["actions"].append(
                solution.optimized.actions[i : i + mp_len].flatten()
            )
            primitives[mp_len]["states"].append(
                solution.optimized.states[i : i + mp_len].flatten()
            )
            rel_l_temp[mp_len].append(i / len_solution)
            # primitives[mp_len]["rel_l"].append(i / len_solution)
            primitives[mp_len]["cost"].append(cost)
            primitives[mp_len]["delta"].append(delta)
            i += mp_len
        for mp_len in rel_l_temp.keys():
            temp = np.array(rel_l_temp[mp_len])
            temp = temp / np.max(temp)
            primitives[mp_len]["rel_l"].extend(temp.tolist())
    for length in lengths:
        # actions = np.round(np.array(primitives[length]["actions"]), decimals=decimals)
        # states = np.round(np.array(primitives[length]["states"]), decimals=decimals)
        actions = np.array(primitives[length]["actions"])
        states = np.array(primitives[length]["states"])
        # cost = np.round(np.array(primitives[length]["cost"]), decimals=decimals)
        cost = np.atleast_2d(np.array(primitives[length]["cost"]))
        delta = np.atleast_2d(np.array(primitives[length]["delta"]))

        # rel_l = np.atleast_2d(np.array(primitives[length]["rel_l"]))
        # rel_l = rel_l / np.max(rel_l)
        rel_l = np.atleast_2d(np.array(primitives[length]["rel_l"]))

        theta_0 = np.atleast_2d(states[:, 2])
        if states.shape[1] == length * 5:
            s_0 = np.atleast_2d(states[:, 3])
            phi_0 = np.atleast_2d(states[:, 4])
            array = np.concatenate(
                [actions, theta_0.T, s_0.T, phi_0.T, cost.T, rel_l.T, delta.T], axis=1
            )
        else:
            array = np.concatenate(
                [actions, theta_0.T, cost.T, rel_l.T, delta.T], axis=1
            )
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
        df = pd.concat(dataframes[length])
        columns = []
        for i in range(length):
            if df.shape[1] == 17:
                columns.append(("actions", f"a_{i}"))
                columns.append(("actions", f"dphi_{i}"))
            else:
                columns.append(("actions", f"s_{i}"))
                columns.append(("actions", f"phi_{i}"))
        columns.append(("states", "theta_0"))
        if df.shape[1] == 17:
            columns.append(("states", "s_0"))
            columns.append(("states", "phi_0"))
        columns.append(("misc", "cost"))
        columns.append(("misc", "rel_l"))
        columns.append(("misc", "delta_0"))
        columns.append(("env", "name"))
        columns.append(("misc", "count"))
        multiindex = pd.MultiIndex.from_tuples(columns)
        df = df.groupby(df.columns.tolist(), as_index=False).size()
        df.columns = multiindex
        # min_cost = pd.DataFrame(df[("misc", "cost")]).groupby(df[("env", "name")], as_index=False).transform("min")
        min_cost = df.groupby([("env", "name")])[[("misc", "cost")]].min()
        df[("misc", "min_cost")] = df[("env", "name")].apply(lambda x: min_cost.loc[x])  # type: ignore
        df[("misc", "rel_c")] = df[("misc", "min_cost")] / df[("misc", "cost")]
        df.drop(columns=[("misc", "min_cost"), ("misc", "cost")], inplace=True)
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
        if len(instance.robots[0].goal) == 5:
            data[("env", "s_s")].append(instance.robots[0].start[3])
            data[("env", "s_g")].append(instance.robots[0].goal[3])
            data[("env", "phi_s")].append(instance.robots[0].start[4])
            data[("env", "phi_g")].append(instance.robots[0].goal[4])

    df = pd.DataFrame(data)
    multiindex = pd.MultiIndex.from_tuples(data.keys())
    df.columns = multiindex
    return df


def main():
    trials = int(sys.argv[1])
    # trials = 1000
    timelimit_db_astar = 3000
    timelimit_db_cbs = 3000
    lengths = [5]
    dynamics = "unicycle1_v0"
    dynamics = "unicycle2_v0"
    # results = {length: {"actions": [], "states": [], "cost": []} for length in lengths}
    # arrays = {}
    data = {
        "delta_0": 0.5,
        "delta_rate": 0.9,
        "num_primitives_0": 100,
        "num_primitives_rate": 1.5,
        "alpha": 0.5,
        "filter_duplicates": True,
        "heuristic1": "reverse-search",
        "heuristic1_delta": 1.0,
        "mp_path": f"../new_format_motions/{dynamics}/{dynamics}.msgpack",
    }
    # random_instances = [
    #     diffmp.problems.Instance.random(
    #         6, 8, random.randint(4, 6), random.random() * (0.6 - 0.2) + 0.2, [dynamics]
    #     )
    #     for _ in range(50)
    # ]
    n_random = 100
    random_instances = []
    pbar_0 = tqdm(total=n_random)
    for _ in range(n_random):
        instance = diffmp.problems.Instance.random(
            6, 8, random.randint(4, 6), random.random() * (0.4 - 0.1) + 0.1, [dynamics]
        )
        random_instances.append(instance)
        pbar_0.update()
    pbar_0.close()
    instances = random_instances
    configurations = [data]
    # exec_task = partial(execute_task, env_dict=environments)

    # input = instances[0].data
    # assert isinstance(input, dict)
    # result = dbcbs_py.db_cbs(
    #     instances[0].to_dict(),
    #     "/tmp/1.txt",
    #     "/tmp/2.txt",
    #     configurations[0],
    #     timelimit_db_astar,
    #     timelimit_db_cbs,
    # )
    # breakpoint()
    tasks = []
    for instance in instances:
        for configuration in configurations:
            tasks += [
                Task(
                    instance,
                    configuration,
                    timelimit_db_astar,
                    timelimit_db_cbs,
                    [],
                )
                for _ in range(trials)
            ]
    total = len(tasks)
    solved = 0
    failure = 0
    pbar = tqdm(total=total)
    pbar.set_postfix({"s": solved, "f": failure})
    solved_tasks = []
    random.shuffle(tasks)
    with mp.Pool(6, maxtasksperchild=10) as p:
        try:
            for result in p.imap_unordered(execute_task, tasks):
                if len(result.solutions) > 0:
                    solved_tasks.append(result)
                    solved += 1
                else:
                    failure += 1

                pbar.set_postfix({"s": solved, "f": failure})
                pbar.update()
        except KeyboardInterrupt:
            pass

    pbar.close()
    primitives = tasks_to_mp(solved_tasks, lengths)

    instances_df = instances_to_df(instances)

    for length, df in primitives.items():
        dataset = df.merge(instances_df).drop(columns=("env", "name"))
        print(dataset.env.describe())
        print(dataset.misc.describe())
        breakpoint()
        dataset.to_parquet(f"data/training_datasets/new_{dynamics}.parquet")


if __name__ == "__main__":
    main()
