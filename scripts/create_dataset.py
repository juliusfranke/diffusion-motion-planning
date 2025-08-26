from __future__ import annotations
from collections import defaultdict
import multiprocessing as mp
from pathlib import Path
import sys
import random

import dbcbs_py
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm
import yaml

import diffmp


def split_solution(task: diffmp.utils.Task, lengths: list[int], robot_idx: int):
    assert isinstance(task.instance, diffmp.problems.Instance)
    dynamics = task.instance.robots[robot_idx].dynamics
    primitives = {
        length: {"actions": [], "states": [], "rel_l": [], "cost": [], "delta": []}
        for length in lengths
    }
    min_len = min(lengths)
    dataframes = {}
    for solution in task.solutions:
        optimized = solution.optimized[robot_idx]
        i = 0
        rel_l_temp = defaultdict(list)
        len_solution = len(optimized.actions)
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
                        optimized.actions[-mp_len:].flatten()
                    )
                    primitives[mp_len]["states"].append(
                        optimized.states[-mp_len:].flatten()
                    )
                    primitives[mp_len]["cost"].append(cost)
                    primitives[mp_len]["rel_l"].append(i / len_solution)
                    primitives[mp_len]["delta"].append(delta)
                    break
                continue
            primitives[mp_len]["actions"].append(
                optimized.actions[i : i + mp_len].flatten()
            )
            primitives[mp_len]["states"].append(
                optimized.states[i : i + mp_len].flatten()
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
        match dynamics.name:
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


def tasks_to_mp(tasks: list[diffmp.utils.Task], lengths: list[int], robot_idx: int):
    dataframes = defaultdict(list)
    primitives = {}
    for task in tasks:
        task_dataframes = split_solution(task, lengths, robot_idx)
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


# def instances_to_df(instances: list[diffmp.problems.Instance]):
#     data = defaultdict(list)
#     for instance in instances:
#         data[("env", "name")].append(instance.name)
#         data[("env", "area")].append(instance.environment.area)
#         data[("env", "area_blocked")].append(instance.environment.area_blocked)
#         data[("env", "area_free")].append(instance.environment.area_free)
#         data[("env", "env_width")].append(instance.environment.env_width)
#         data[("env", "env_height")].append(instance.environment.env_height)
#         data[("env", "n_obstacles")].append(instance.environment.n_obstacles)
#         data[("env", "p_obstacles")].append(instance.environment.p_obstacles)
#         match instance.robots[0].dynamics:
#             case "unicycle1_v0":
#                 data[("env", "theta_s")].append(instance.robots[0].start[2])
#                 data[("env", "theta_g")].append(instance.robots[0].goal[2])
#             case "unicycle2_v0":
#                 data[("env", "theta_s")].append(instance.robots[0].start[2])
#                 data[("env", "theta_g")].append(instance.robots[0].goal[2])
#                 data[("env", "s_s")].append(instance.robots[0].start[3])
#                 data[("env", "s_g")].append(instance.robots[0].goal[3])
#                 data[("env", "phi_s")].append(instance.robots[0].start[4])
#                 data[("env", "phi_g")].append(instance.robots[0].goal[4])
#             case "car1_v0":
#                 data[("env", "theta_s")].append(instance.robots[0].start[2])
#                 data[("env", "theta_g")].append(instance.robots[0].goal[2])
#                 data[("env", "theta_2_s")].append(instance.robots[0].start[3])
#                 data[("env", "theta_2_g")].append(instance.robots[0].goal[3])

#     df = pd.DataFrame(data)
#     multiindex = pd.MultiIndex.from_tuples(data.keys())
#     df.columns = multiindex
#     return df


def main():
    trials = int(sys.argv[1])
    # trials = 1000
    timelimit_db_astar = 5000
    timelimit_db_cbs = 5000
    lengths = [5, 10, 15, 20]
    lengths = [10]
    dynamics = "unicycle1_v0"
    n_robots = 2
    # dynamics = ["unicycle1_v0", "unicycle1_v0"]
    # dynamics = "unicycle2_v0"
    # dynamics = "car1_v0"
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
        "mp_path": [
            f"../new_format_motions/{dynamics}/{dynamics}.msgpack"
            for _ in range(n_robots)
        ],
        "execute_joint_optimization": True,
        "execute_greedy_optimization": False,
        "heuristic1_num_primitives_0": 100,
        "always_add_node": False,
        "rewire": True,
        "residual_force": False,  # NN, augmented state or Ellipsoid shape
        "suboptimality_factor": 1.3,  # 3.3, 2 - if Ellipsoid shape
    }
    if dynamics == "car1_v0":
        data["delta_0"] = 0.9
    n_random = 20
    random_instances = []
    pbar_0 = tqdm(total=n_random)
    for _ in range(n_random):
        instance = diffmp.problems.Instance.random(
            5,
            5,
            random.random() * (0.4 - 0.1) + 0.1,
            # 0.2,
            [dynamics]*n_robots,
            dim=diffmp.problems.Dim.TWO_D,
        )
        random_instances.append(instance)
        pbar_0.update()
    pbar_0.close()
    instances = random_instances
    configurations = [data]
    tasks = []
    instance_dict = {}

    for instance in instances:
        instance_data = instance.to_dict()
        instance_dict[instance.name] = instance
        for configuration in configurations:
            tasks += [
                diffmp.utils.Task(
                    instance_data,
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
    # swap = diffmp.problems.Instance.from_yaml(Path("../example/swap2_unicycle.yaml"))
    # tasks[0].instance = swap.to_dict()
    # task = diffmp.utils.execute_task(tasks[0])
    # result = task.solutions
    # if len(result) > 0:
    #     print(f"{len(result)=}")
    #     traj_1 = np.array(result[-1].optimized[0].actions)
    #     traj_2 = np.array(result[-1].optimized[1].actions)
    #     # best_traj = result[-1].optimized[0].states
    #     # print(np.array(best_traj[0].actions))
    # else:
    #     print("Failure")
    # sys.exit()
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
    solved_instances = set()
    for task in solved_tasks:
        task.instance = instance_dict[task.instance["name"]]
        solved_instances.add(task.instance)
    primitives = {length: [] for length in lengths}
    for i in range(n_robots):
        robot_primitives = tasks_to_mp(solved_tasks, lengths, i)
        for length in lengths:
            primitives[length].append(robot_primitives[length])

    instances_yaml = [yaml.dump(instance.to_dict()) for instance in solved_instances]
    env_uuid = [instance.name for instance in solved_instances]
    uuid_to_idx = {uuid: idx for idx, uuid in enumerate(env_uuid)}

    filename = f"data/training_datasets/{dynamics}_x{n_robots}_.h5"
    breakpoint()
    with h5py.File(filename, "w") as f:
        f.create_dataset("env_uuid", data=np.array(env_uuid, dtype="S36"))
        f.create_dataset(
            "environments",
            data=np.array(instances_yaml, dtype=h5py.string_dtype("utf-8")),
        )
        for length, df_list in primitives.items():
            group = f.create_group(f"length_{length:03}")

            for robot_idx, df in enumerate(df_list):
                robot_group = group.create_group(f"robot_{robot_idx:03}")
                df[("env", "idx")] = df[("env", "name")].apply(lambda x: uuid_to_idx[x])
                df.drop(columns=("env", "name"), inplace=True)
                robot_group.create_dataset(
                    "scalars",
                    data=df.to_numpy(),
                    chunks=True,
                    compression="gzip",
                )
                columns = np.array([list(t) for t in df.columns.to_list()], dtype="S")
                robot_group.create_dataset("columns", data=columns)
            # group.create_dataset("env_ids", data=np.array(env_ids, dtype="S36"))


if __name__ == "__main__":
    main()
