import multiprocessing as mp
import random
from pathlib import Path

from tqdm import tqdm

import diffmp
import diffmp.problems as pb
import diffmp.utils as du


def main():
    trials = 20
    trials = 20
    timelimit_db_astar = 1000
    timelimit_db_cbs = 1500
    n_random = 40
    n_random = 20
    # lengths = [5]
    # results = {length: {"actions": [], "states": [], "cost": []} for length in lengths}
    dynamics = "unicycle1_v0"
    n_robots = 2
    # dynamics = "unicycle2_v0"
    # dynamics = "car1_v0"
    data = {
        "delta_0": 0.5,
        "delta_rate": 0.9,
        "num_primitives_0": 100,
        "num_primitives_rate": 1.5,
        "alpha": 0.5,
        "filter_duplicates": True,
        "heuristic1": "reverse-search",
        "heuristic1_delta": 1.0,
        "mp_path": [f"../new_format_motions/{dynamics}/{dynamics}.msgpack" for _ in range(n_robots)],
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
    # dynamics = 2 * [dynamics]
    random_instances: list[pb.Instance] = []
    pbar_0 = tqdm(total=n_random)
    while len(random_instances) < n_random:
        pbar_1 = tqdm(total=trials, leave=False)
        tasks = []
        instance = pb.Instance.random(
            5,
            5,
            random.random() * (0.4 - 0.1) + 0.1,
            # 0.2,
            n_robots * [dynamics],
            pb.Dim.TWO_D,
        )
        tasks += [
            du.Task(
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
        for task in tasks:
            task.instance = task.instance.to_dict()
        with mp.Pool(4, maxtasksperchild=10) as p:
            for result in p.imap_unordered(du.execute_task, tasks):
                pbar_1.update()
                if len(result.solutions):
                    result.solutions[0].cost
                    duration_sum += result.solutions[0].runtime
                    cost_sum += min([s.cost for s in result.solutions])
                    successes += 1
                    pbar_1.set_postfix({"s": successes, "f": failures})
                else:
                    failures += 1
                    pbar_1.set_postfix({"s": successes, "f": failures})

        pbar_1.close()
        if successes >= trials * 0.8:
            success = successes / trials
            duration = duration_sum / successes
            cost = cost_sum / successes
            baseline = diffmp.problems.Baseline(success, duration, cost)
            instance.baseline = baseline
            random_instances.append(instance)
            pbar_0.update()
    pbar_0.close()
    print([i.baseline.success for i in random_instances])  # pyright: ignore
    breakpoint()
    path = Path(f"data/test_instances/{dynamics}_x{n_robots}")
    # breakpoint()
    path.mkdir(parents=True, exist_ok=True)
    for i, random_instance in enumerate(random_instances):
        random_instance.to_yaml(path / Path(f"{i}.yaml"))


if __name__ == "__main__":
    main()
