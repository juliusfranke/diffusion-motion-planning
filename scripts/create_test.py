from collections import defaultdict
import tempfile
import pandas as pd
from typing import List


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
    timelimit_db_astar = 1000
    timelimit_db_cbs = 1500
    n_random = 10
    # lengths = [5]
    # results = {length: {"actions": [], "states": [], "cost": []} for length in lengths}
    dynamics = "unicycle1_v0"
    dynamics = "unicycle2_v0"
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
    random_instances: List[diffmp.problems.Instance] = []
    pbar_0 = tqdm(total=n_random)
    while len(random_instances) < n_random:
        pbar_1 = tqdm(total=trials, leave=False)
        tasks = []
        instance = diffmp.problems.Instance.random(
            6, 8, random.randint(4, 6), random.random() * (0.4 - 0.1) + 0.1, [dynamics]
        )
        tasks += [
            diffmp.utils.Task(
                instance,
                data,
                timelimit_db_astar,
                timelimit_db_cbs,
                [],
            )
            for _ in range(trials)
        ]
        successes = 0
        failures = 0
        duration_sum = 0
        cost_sum = 0
        with mp.Pool(4, maxtasksperchild=10) as p:
            for result in p.imap_unordered(diffmp.utils.execute_task, tasks):
                if len(result.solutions):
                    pbar_1.update()
                    result.solutions[0].cost
                    duration_sum += result.solutions[0].runtime
                    cost_sum += min([s.cost for s in result.solutions])
                    successes += 1
                    pbar_1.set_postfix({"s": successes, "f": failures})
                else:
                    failures += 1
                    pbar_1.set_postfix({"s": successes, "f": failures})

        pbar_1.close()
        if successes > 0:
            success = successes / trials
            duration = duration_sum / successes
            cost = cost_sum / successes
            baseline = diffmp.problems.Baseline(success, duration, cost)
            instance.baseline = baseline
            random_instances.append(instance)
            pbar_0.update()
    pbar_0.close()
    breakpoint()
    print([i.baseline.success for i in random_instances])
    for i, random_instance in enumerate(random_instances):
        random_instance.to_yaml(Path(f"data/test_instances/{dynamics}/{i}.yaml"))


if __name__ == "__main__":
    main()
