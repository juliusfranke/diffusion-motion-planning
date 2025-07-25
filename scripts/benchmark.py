from collections import defaultdict
import datetime
from functools import partial
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

# from script_train import  load
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


TRIALS = 10
# TRIALS = 50
TIMELIMIT_DB_ASTAR = 2000
TIMELIMIT_DB_CBS = 5000
N_MPs = [100, 150, 200, 250]
# N_MPs = [100]
DELTA_0s = [0.5]
# DELTA_0s = [0.9]
# DELTA_0s = [0.3, 0.5, 0.7]
# DELTA_0s = [0.7, 0.9, 1.1]
# N_RANDOM = 50
N_RANDOM = 10
DATA = {
    # "delta_0": 0.5,
    "delta_rate": 0.9,
    # "num_primitives_0": N_MP[0],
    "num_primitives_rate": 1.0,
    "alpha": 0.5,
    "filter_duplicates": True,
    "heuristic1": "reverse-search",
    "heuristic1_delta": 1.0,
}


def log_floor(x, precision=0.0) -> float:
    return np.true_divide(np.floor(x * 10**precision), 10**precision)


def apply_df(model_value, instance, baseline):
    baseline_value = baseline.loc[instance]
    return (model_value - baseline_value) / baseline_value


def plot_results(benchmark: pd.DataFrame):
    baseline = benchmark[benchmark.name == "original"]
    bl_median = baseline.groupby("instance").median(numeric_only=True)
    bl_mean = baseline.groupby("instance").mean(numeric_only=True)
    # models = pd.DataFrame(benchmark[benchmark.name != "original"])
    apply_success = partial(apply_df, baseline=bl_mean["success"])
    benchmark["rel_s"] = benchmark.apply(
        lambda x: apply_success(x.success, x.instance), axis=1
    )
    apply_bcost = partial(apply_df, baseline=bl_median["best_cost"])
    benchmark["rel_bc"] = benchmark.apply(
        lambda x: apply_bcost(x.best_cost, x.instance), axis=1
    )
    apply_fcost = partial(apply_df, baseline=bl_median["first_cost"])
    benchmark["rel_fc"] = benchmark.apply(
        lambda x: apply_fcost(x.first_cost, x.instance), axis=1
    )
    apply_duration = partial(apply_df, baseline=bl_median["duration"])
    benchmark["rel_d"] = benchmark.apply(
        lambda x: apply_duration(x.duration, x.instance), axis=1
    )
    lintresh = []
    thresh_df = (
        benchmark.groupby("name")[["rel_d", "rel_bc", "rel_fc"]]
        .quantile([0.25, 0.75])  # pyright:ignore
        .abs()
        .dropna()
        .reset_index()
    )
    thresh_df = thresh_df[thresh_df.name == "model"]
    thresh_d = 10 ** log_floor(np.log10(thresh_df["rel_d"].min()))
    thresh_fc = 10 ** log_floor(np.log10(thresh_df["rel_fc"].min()))
    thresh_bc = 10 ** log_floor(np.log10(thresh_df["rel_bc"].min()))
    thresh = [thresh_d, thresh_fc, thresh_bc]
    mean = benchmark.groupby("name").mean(numeric_only=True)
    med = benchmark.groupby("name").median(numeric_only=True)
    print(mean)
    print(med)
    fig, ax = plt.subplots(1, 3)
    # sns.barplot(benchmark, y="success", hue="name", ax=ax[0])
    h = sns.boxplot(benchmark, y="rel_d", hue="name", ax=ax[0])
    h.get_legend().set_visible(False)
    g = sns.boxplot(benchmark, y="rel_fc", hue="name", ax=ax[1])
    g.get_legend().set_visible(False)
    sns.boxplot(benchmark, y="rel_bc", hue="name", ax=ax[2])
    for i in range(3):
        lintresh = thresh[i]
        ax[i].set_yscale("symlog", linthresh=lintresh)
        ax[i].axhline(y=0, color="red", linestyle="--")
        ax[i].set_xticks(ax[i].get_xticks(), ax[i].get_xticklabels())
        ax[i].yaxis.set_major_locator(
            mticker.SymmetricalLogLocator(base=10, linthresh=lintresh)
        )
        ax[i].yaxis.set_minor_locator(
            mticker.SymmetricalLogLocator(
                base=10, linthresh=lintresh, subs=[x / 10 for x in range(1, 10)]
            )
        )
        ax[i].grid(axis="y")
    plt.show()


def get_baseline_config(dynamics: str):
    mp_path_base = [f"../new_format_motions/{dynamics}/{dynamics}.msgpack"]
    configurations_base = []
    for mp_path in mp_path_base:
        temp = {"mp_path": mp_path}
        configurations_base.append(DATA | temp)
    return configurations_base


def gen_instances(n_random: int, dynamics: str):
    random_instances = []
    pbar_0 = tqdm(total=n_random)
    for _ in range(n_random):
        instance = diffmp.problems.Instance.random(
            6, 8, random.randint(4, 6), random.random() * (0.4 - 0.1) + 0.1, [dynamics]
        )
        random_instances.append(instance)
        pbar_0.update()
    pbar_0.close()
    return random_instances


def benchmark_model(
    model: diffmp.torch.Model,
):
    dynamics = model.config.dynamics.name
    if dynamics == "car1_v0":
        DATA["delta_0"] = 0.9
    configurations_base = get_baseline_config(dynamics)
    instances = gen_instances(N_RANDOM, dynamics)
    tasks = []
    for instance in instances:
        for trial in range(TRIALS):

            tmp_path = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
            diffmp.utils.export(model, instance, Path(tmp_path.name), n_mp=N_MP)
            temp = DATA | {"mp_path": str(tmp_path.name)}
            tasks += [
                diffmp.utils.Task(
                    instance,
                    temp,
                    TIMELIMIT_DB_ASTAR,
                    TIMELIMIT_DB_CBS,
                    [],
                )
            ]
        for configuration in configurations_base:
            tasks += [
                diffmp.utils.Task(
                    instance,
                    configuration,
                    TIMELIMIT_DB_ASTAR,
                    TIMELIMIT_DB_CBS,
                    [],
                )
                for _ in range(TRIALS)
            ]
    random.shuffle(tasks)
    pbar = tqdm(total=len(tasks))
    executed_tasks: List[diffmp.utils.Task] = []
    with mp.Pool(4, maxtasksperchild=10) as p:
        for result in p.imap_unordered(diffmp.utils.execute_task, tasks):
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
        results["cost"].append(task.solutions[0].cost)
    result_df = pd.DataFrame(results)
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    save_path = Path(f"data/results/{dynamics}/{date}.parquet")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(save_path)
    print(result_df.groupby(["instance", "name"]).median(numeric_only=True))
    print(result_df.groupby(["instance", "name"]).mean(numeric_only=True))


def benchmark_composite(
    composite_config: diffmp.torch.CompositeConfig,
):
    dynamics = composite_config.models[0].config.dynamics.name
    # if dynamics == "car1_v0":
    #     DATA["delta_0"] = 0.9
    configurations_base = get_baseline_config(dynamics)
    instances = gen_instances(N_RANDOM, dynamics)
    # instances = [
    #     diffmp.problems.Instance.from_yaml(Path("../example/parallelpark_0.yaml"))
    # ]
    tasks = []
    for instance in instances:
        for delta_0 in DELTA_0s:
            for N_MP in N_MPs:
                for _ in range(TRIALS):

                    tmp_path = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
                    diffmp.utils.export_composite(
                        composite_config, instance, Path(tmp_path.name), n_mp=N_MP * 5
                    )
                    temp = DATA | {
                        "mp_path": str(tmp_path.name),
                        "num_primitives_0": N_MP,
                        "delta_0": delta_0,
                    }
                    # temp["num_primitives_0"] = N_MP
                    tasks += [
                        diffmp.utils.Task(
                            instance,
                            temp,
                            TIMELIMIT_DB_ASTAR,
                            TIMELIMIT_DB_CBS,
                            [],
                        )
                    ]
                for configuration in configurations_base:
                    tmp_config = {
                        "num_primitives_0": N_MP,
                        "delta_0": delta_0,
                    } | configuration
                    tasks += [
                        diffmp.utils.Task(
                            instance,
                            tmp_config,
                            TIMELIMIT_DB_ASTAR,
                            TIMELIMIT_DB_CBS,
                            [],
                        )
                        for _ in range(TRIALS)
                    ]
    random.shuffle(tasks)
    pbar = tqdm(total=len(tasks))
    executed_tasks: List[diffmp.utils.Task] = []
    with mp.Pool(4, maxtasksperchild=10) as p:
        for result in p.imap_unordered(diffmp.utils.execute_task, tasks):
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
        results["n_mp"].append(task.config["num_primitives_0"])
        results["delta_0"].append(task.config["delta_0"])
        if "new_format_motions" in path:
            results["name"].append("original")
        else:
            results["name"].append("model")
        # results["name"].append(task.config["mp_path"])
        success = len(task.solutions) > 0
        results["success"].append(success)
        if not success:
            results["duration"].append(None)
            results["best_cost"].append(None)
            results["first_cost"].append(None)
            continue
        results["duration"].append(task.solutions[0].runtime)
        results["first_cost"].append(task.solutions[0].cost)
        best_cost = min([s.cost for s in task.solutions])
        results["best_cost"].append(best_cost)
    result_df = pd.DataFrame(results)

    # breakpoint()
    print(result_df.groupby(["instance", "name"]).median(numeric_only=True))
    print(result_df.groupby(["instance", "name"]).mean(numeric_only=True))
    breakpoint()
    # plot_results(result_df.copy(deep=True))
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    save_path = Path(f"data/results/{dynamics}/{date}.parquet")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_parquet(save_path)


def main():
    model_name = sys.argv[1]
    model_path = Path(f"data/models/{model_name}")
    if model_path.suffixes[0] == ".standard":
        model = diffmp.torch.Model.load(model_path)
        benchmark_model(model)

    elif model_path.suffixes[0] == ".composite":
        composite_config = diffmp.torch.CompositeConfig.from_yaml(model_path)
        benchmark_composite(composite_config)


if __name__ == "__main__":
    main()
