from collections import defaultdict
import pandas as pd
from typing import List


# import dbcbs_py
import diffmp
import random
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
from create_dataset import Task, Solution, execute_task
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    trials = 20
    timelimit_db_astar = 3000
    timelimit_db_cbs = 3000
    lengths = [5]
    results = {length: {"actions": [], "states": [], "cost": []} for length in lengths}
    dynamics = "unicycle1_v0"
    # dynamics = "unicycle2_v0"
    mp_path_base = [f"../new_format_motions/{dynamics}/{dynamics}.msgpack"]
    mp_paths = [f"data/output/{dynamics}/{i}.yaml" for i in range(10)]
    data = {
        "delta_0": 0.5,
        "delta_rate": 0.9,
        "num_primitives_0": 100,
        "num_primitives_rate": 1.5,
        "alpha": 0.5,
        "filter_duplicates": True,
        "heuristic1": "reverse-search",
        "heuristic1_delta": 1.0,
    }
    if dynamics == "unicycle1_v0":
        bugtrap = diffmp.problems.Instance.from_yaml(Path("../example/bugtrap.yaml"))
    elif dynamics == "unicycle2_v0":
        bugtrap = diffmp.problems.Instance.from_yaml(Path("../example/bugtrap_2.yaml"))
    else:
        raise NotImplementedError
    obstacle_bounds = diffmp.problems.obstacle.Bounds2D(1,2,1,2)
    random_instance = diffmp.problems.Instance.random(5, 10, 5, obstacle_bounds,[dynamics])
    # instances = [bugtrap]
    instances = [random_instance]

    breakpoint()
    configurations_model = []
    for mp_path in mp_paths:
        temp = {"mp_path": mp_path}
        configurations_model.append(data | temp)

    configurations_base = []
    for mp_path in mp_path_base:
        temp = {"mp_path": mp_path}
        configurations_base.append(data | temp)
    # exec_task = partial(execute_task, env_dict=environments)
    tasks = []
    for instance in instances:
        # for configuration in configurations_model:
        #     tasks += [
        #         Task(
        #             instance,
        #             configuration,
        #             timelimit_db_astar,
        #             timelimit_db_cbs,
        #         )
        #         for _ in range(trials // len(mp_paths))
            # ]
        for configuration in configurations_base:
            tasks += [
                Task(
                    instance,
                    configuration,
                    timelimit_db_astar,
                    timelimit_db_cbs,
                )
                for _ in range(trials)
            ]
    random.shuffle(tasks)
    pbar = tqdm(total=trials * len(instances) * 2)
    executed_tasks: List[Task] = []
    with mp.Pool(2) as p:
        for result in p.imap_unordered(execute_task, tasks):
            executed_tasks.append(result)
            # solutions += 1
            # runtimes.append(result.runtime)
            # costs.append(len(result.optimized_solution.actions)/10)
            pbar.update()
    pbar.close()
    results = defaultdict(list)
    for task in executed_tasks:
        path = task.config["mp_path"]
        if "new_format_motions" in path:
            results["name"].append("original")
        else:
            results["name"].append("model")
        # results["name"].append(task.config["mp_path"])
        success = isinstance(task.optimized_solution, Solution)
        results["success"].append(success)
        if not success:
            results["duration"].append(None)
            results["cost"].append(None)
            continue
        results["duration"].append(task.runtime)
        assert isinstance(task.optimized_solution, Solution)
        results["cost"].append(len(task.optimized_solution.actions) / 10)
    result_df = pd.DataFrame(results)
    sns.boxplot(
        result_df,
        x="name",
        y="success",
        medianprops={"color": "r", "linewidth": 2},
        showmeans=True,
    )
    plt.show()
    sns.boxplot(result_df, x="name", y="cost")
    plt.show()
    sns.boxplot(result_df, x="name", y="duration")
    plt.show()


if __name__ == "__main__":
    main()
