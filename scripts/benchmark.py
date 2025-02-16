from collections import defaultdict
import datetime
import sys
import tempfile
import pandas as pd
from typing import List


# import dbcbs_py
import diffmp
import random
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
from create_dataset import Task, Solution, execute_task

# from script_train import  load
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    model_name = sys.argv[1]
    model_path = Path(f"data/models/{model_name}")
    model = diffmp.torch.Model.load(model_path)

    trials = 10
    timelimit_db_astar = 1000
    timelimit_db_cbs = 1000
    lengths = [5]
    results = {length: {"actions": [], "states": [], "cost": []} for length in lengths}
    # dynamics = "unicycle1_v0"
    # dynamics = "unicycle2_v0"
    dynamics = model.dynamics.name
    # mp_path_base = [f"../new_format_motions/{dynamics}/{dynamics}.msgpack"]
    mp_path_base = [f"../new_format_motions/{dynamics}/{dynamics}.bin"]
    n_mp = 200
    data = {
        "delta_0": 0.5,
        "delta_rate": 0.9,
        "num_primitives_0": n_mp,
        "num_primitives_rate": 1.0,
        "alpha": 0.5,
        "filter_duplicates": True,
        "heuristic1": "reverse-search",
        "heuristic1_delta": 1.0,
    }
    n_random = 20
    random_instances = []
    pbar_0 = tqdm(total=n_random)
    for _ in range(n_random):
        instance = diffmp.problems.Instance.random(
            6, 8, random.randint(4, 6), random.random() * (0.4 - 0.1) + 0.1, [dynamics]
        )
        random_instances.append(instance)
        pbar_0.update()
    pbar_0.close()
    # mp_paths = [f"data/output/{dynamics}/{i}.yaml" for i in range(1)]
    instances = random_instances
    # instances = [diffmp.problems.Instance.from_yaml(Path("../example/bugtrap_2.yaml"))]

    configurations_model = []
    # for mp_path in mp_paths:
    #     temp = {"mp_path": mp_path}
    #     configurations_model.append(data | temp)

    configurations_base = []
    for mp_path in mp_path_base:
        temp = {"mp_path": mp_path}
        configurations_base.append(data | temp)
    # exec_task = partial(execute_task, env_dict=environments)
    tasks = []
    for instance in instances:
        for trial in range(trials):

            tmp_path = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
            # tmp_path.name
            # mp_path = Path(f"data/output/{instance.name}_{trial}.yaml")
            diffmp.utils.export(model, instance, Path(tmp_path.name), n_mp=n_mp)
            temp = data | {"mp_path": str(tmp_path.name)}
            tasks += [
                Task(
                    instance,
                    temp,
                    timelimit_db_astar,
                    timelimit_db_cbs,
                    [],
                )
            ]
        for configuration in configurations_base:
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
    random.shuffle(tasks)
    pbar = tqdm(total=len(tasks))
    executed_tasks: List[Task] = []
    with mp.Pool(4, maxtasksperchild=10) as p:
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
        results["instance"].append(task.instance.name)
        if "new_format_motions" in path:
            results["name"].append("original")
        else:
            results["name"].append("model")
        # results["name"].append(task.config["mp_path"])
        success = len(task.solutions) > 0
        results["success"].append(success)
        if not success:
            results["duration"].append(None)
            results["cost"].append(None)
            continue
        results["duration"].append(task.solutions[0].runtime)
        # assert isinstance(task.optimized_solution, Solution)
        results["cost"].append(min([s.cost for s in task.solutions]))
    result_df = pd.DataFrame(results)
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    save_path = f"data/results/{dynamics}/{date}.parquet"
    result_df.to_parquet(save_path)
    print(result_df.groupby(["instance", "name"]).median(numeric_only=True))
    print(result_df.groupby(["instance", "name"]).mean(numeric_only=True))
    breakpoint()
    sns.barplot(result_df, x="instance", y="success", hue="name").set(xticklabels=[])
    plt.show()
    sns.boxplot(result_df, x="instance", y="cost", hue="name").set(xticklabels=[])
    plt.show()
    sns.boxplot(result_df, x="instance", y="duration", hue="name").set(xticklabels=[])
    plt.show()


if __name__ == "__main__":
    main()
