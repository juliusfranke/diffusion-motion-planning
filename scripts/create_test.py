import multiprocessing as mp
import random
from pathlib import Path

from tqdm import tqdm

import diffmp
import diffmp.problems as pb
import diffmp.utils as du
import time


def worker_func(func, args, result_queue, idx):
    """Run a single task and put result or exception in result_queue."""
    try:
        result = func(*args)
        result_queue.put((idx, "done", result))
    except Exception as e:
        result_queue.put((idx, "error", str(e)))


def run_tasks(tasks, func, timeout, max_workers, pbar: tqdm):
    """
    tasks: list of tuples with arguments for func
    func: function to run (e.g., your nanobind C++ binding)
    timeout: max time in seconds per task
    max_workers: number of parallel workers
    """
    result_queue = mp.Queue()
    active = []  # [(Process, idx)]
    results = [None] * len(tasks)
    statuses = ["pending"] * len(tasks)
    next_task = 0

    successes = 0
    failures = 0
    duration_sum = 0
    cost_sum = 0

    def start_task(idx):
        p = mp.Process(target=worker_func, args=(func, [tasks[idx]], result_queue, idx))
        p.start()
        return p

    while next_task < len(tasks) or active:
        # Start new tasks if we have slots
        while len(active) < max_workers and next_task < len(tasks):
            p = start_task(next_task)
            active.append((p, time.time(), next_task))
            next_task += 1

        # Check active tasks
        new_active = []
        for p, start_time, idx in active:
            p.join(timeout=0)  # Non-blocking check
            if p.exitcode is not None:
                # Process finished
                continue
            elif time.time() - start_time > timeout:
                # Timeout: kill
                p.terminate()
                p.join()
                statuses[idx] = "timeout"
            else:
                # Still running
                new_active.append((p, start_time, idx))
        active = new_active

        # Collect results
        while not result_queue.empty():
            idx, status, result = result_queue.get()
            statuses[idx] = status
            results[idx] = result
            pbar.update()

            if len(result.solutions):
                costs = [result.solutions[i].cost for i in range(len(result.solutions))]
                pbar.write(str(costs))
                duration_sum += result.solutions[0].runtime
                cost_sum += min([s.cost for s in result.solutions])
                successes += 1
                pbar.set_postfix({"s": successes, "f": failures})
            else:
                failures += 1
                pbar.set_postfix({"s": successes, "f": failures})

        time.sleep(0.05)  # Avoid busy-looping

    return successes, failures, duration_sum, cost_sum


def main():
    trials = 20
    trials = 20
    timelimit_db_astar = 5000
    timelimit_db_cbs = 5000
    n_random = 40
    n_random = 20
    # lengths = [5]
    # results = {length: {"actions": [], "states": [], "cost": []} for length in lengths}
    dynamics = "unicycle1_v0"
    n_robots = 1
    p_min = 0.4
    p_max = 0.6
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
    # dynamics = 2 * [dynamics]
    random_instances: list[pb.Instance] = []
    pbar_0 = tqdm(total=n_random)
    while len(random_instances) < n_random:
        pbar_1 = tqdm(total=trials, leave=False)
        tasks = []
        instance = pb.Instance.random(
            10,
            10,
            # 0.6,
            # random.random() * (p_max - p_min) + p_min,
            0.7,
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
        for task in tasks:
            task.instance = task.instance.to_dict()
        successes, failures, duration_sum, cost_sum = run_tasks(
            tasks, du.execute_task, timeout=10, max_workers=4, pbar=pbar_1
        )
        # with mp.Pool(4, maxtasksperchild=10) as p:
        #     for result in p.imap_unordered(du.execute_task, tasks):
        #         pbar_1.update()
        #         if len(result.solutions):
        #             result.solutions[0].cost
        #             duration_sum += result.solutions[0].runtime
        #             cost_sum += min([s.cost for s in result.solutions])
        #             successes += 1
        #             pbar_1.set_postfix({"s": successes, "f": failures})
        #         else:
        #             failures += 1
        #             pbar_1.set_postfix({"s": successes, "f": failures})

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
