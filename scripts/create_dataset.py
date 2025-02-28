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


def split_solution(task: diffmp.utils.Task, lengths: List[int]):
    dynamics = task.instance.robots[0].dynamics
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
        max_rel_l = np.max(
            [np.max(np.array(rel_l_temp[length])) for length in rel_l_temp.keys()]
        )
        for mp_len in rel_l_temp.keys():
            temp = np.array(rel_l_temp[mp_len])
            temp = temp / max_rel_l
            primitives[mp_len]["rel_l"].extend(temp.tolist())
    for length in lengths:
        actions = np.array(primitives[length]["actions"])
        states = np.array(primitives[length]["states"])
        if len(states.shape) != 2:
            continue
        data = [actions]
        columns = []
        match dynamics:
            case "unicycle1_v0":
                for i in range(length):
                    columns.append(("actions", f"s_{i}"))
                    columns.append(("actions", f"phi_{i}"))
                theta_0 = np.atleast_2d(states[:, 2])
                columns.append(("states", "theta_0"))
                data.append(theta_0.T)
            case "unicycle2_v0":
                for i in range(length):
                    columns.append(("actions", f"a_{i}"))
                    columns.append(("actions", f"dphi_{i}"))
                theta_0 = np.atleast_2d(states[:, 2])
                s_0 = np.atleast_2d(states[:, 3])
                phi_0 = np.atleast_2d(states[:, 4])
                columns.append(("states", "theta_0"))
                columns.append(("states", "s_0"))
                columns.append(("states", "phi_0"))
                data.extend([theta_0.T, s_0.T, phi_0.T])
            case "car1_v0":
                for i in range(length):
                    columns.append(("actions", f"s_{i}"))
                    columns.append(("actions", f"phi_{i}"))
                theta_0 = np.atleast_2d(states[:, 2])
                theta_2_0 = np.atleast_2d(states[:, 3])
                columns.extend([("states", "theta_0"), ("states", "theta_2_0")])
                data.extend([theta_0.T, theta_2_0.T])

        cost = np.atleast_2d(np.array(primitives[length]["cost"]))
        delta = np.atleast_2d(np.array(primitives[length]["delta"]))
        rel_l = np.atleast_2d(np.array(primitives[length]["rel_l"]))
        columns.extend([("misc", "cost"), ("misc", "rel_l"), ("misc", "delta_0")])
        data.extend([cost.T, rel_l.T, delta.T])
        array = np.concatenate(data, axis=1)
        df = pd.DataFrame(array)
        df["instance"] = task.instance.name
        columns.append(("env", "name"))
        df.columns = columns
        dataframes[length] = df
        # cols.extend(dynamics.parameter_set[param.name].cols)

    return dataframes


def tasks_to_mp(tasks: List[diffmp.utils.Task], lengths: List[int]):
    dataframes = defaultdict(list)
    primitives = {}
    for task in tasks:
        task_dataframes = split_solution(task, lengths)
        for length in lengths:
            if length not in task_dataframes.keys():
                continue
            dataframes[length].append(task_dataframes[length])
    for length in lengths:
        df = pd.concat(dataframes[length])
        columns = list(df.columns)
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
        match instance.robots[0].dynamics:
            case "unicycle1_v0":
                data[("env", "theta_s")].append(instance.robots[0].start[2])
                data[("env", "theta_g")].append(instance.robots[0].goal[2])
            case "unicycle2_v0":
                data[("env", "theta_s")].append(instance.robots[0].start[2])
                data[("env", "theta_g")].append(instance.robots[0].goal[2])
                data[("env", "s_s")].append(instance.robots[0].start[3])
                data[("env", "s_g")].append(instance.robots[0].goal[3])
                data[("env", "phi_s")].append(instance.robots[0].start[4])
                data[("env", "phi_g")].append(instance.robots[0].goal[4])
            case "car1_v0":
                data[("env", "theta_s")].append(instance.robots[0].start[2])
                data[("env", "theta_g")].append(instance.robots[0].goal[2])
                data[("env", "theta_2_s")].append(instance.robots[0].start[3])
                data[("env", "theta_2_g")].append(instance.robots[0].goal[3])

    df = pd.DataFrame(data)
    multiindex = pd.MultiIndex.from_tuples(data.keys())
    df.columns = multiindex
    return df


def main():
    trials = int(sys.argv[1])
    # trials = 1000
    timelimit_db_astar = 3000
    timelimit_db_cbs = 3000
    lengths = [5, 10, 15, 20]
    dynamics = "unicycle1_v0"
    # dynamics = "unicycle2_v0"
    dynamics = "car1_v0"
    # results = {length: {"actions": [], "states": [], "cost": []} for length in lengths}
    # arrays = {}
    data = {
        "delta_0": 0.5,
        "delta_rate": 0.9,
        "num_primitives_0": 200,
        "num_primitives_rate": 1.5,
        "alpha": 0.5,
        "filter_duplicates": True,
        "heuristic1": "reverse-search",
        "heuristic1_delta": 1.0,
        "mp_path": f"../new_format_motions/{dynamics}/{dynamics}.msgpack",
    }
    if dynamics == "car1_v0":
        data["delta_0"] = 0.9
    # random_instances = [
    #     diffmp.problems.Instance.random(
    #         6, 8, random.randint(4, 6), random.random() * (0.6 - 0.2) + 0.2, [dynamics]
    #     )
    #     for _ in range(50)
    # ]
    # n_random = 500
    n_random = 500
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
    tasks = []
    for instance in instances:
        for configuration in configurations:
            tasks += [
                diffmp.utils.Task(
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
            for result in p.imap_unordered(diffmp.utils.execute_task, tasks):
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
        # print(dataset.env.describe())
        # print(dataset.misc.describe())
        print(length, dataset.shape)
        dataset.to_parquet(f"data/training_datasets/{dynamics}_l{length}.parquet")


if __name__ == "__main__":
    main()
