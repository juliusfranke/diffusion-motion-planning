import dbcbs_py
from pathlib import Path
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import diffmp


def main():

    time_limit_db_astar = 1000 * 60 * 5
    time_limit_db_cbs = 1000 * 60 * 5
    dynamics = "unicycle1_v0"
    n_robots = 1
    # dynamics = "unicycle2_v0"
    config = {
        "delta_0": 0.5,
        "delta_rate": 0.9,
        "num_primitives_0": 100,
        "num_primitives_rate": 1.5,
        "alpha": 0.5,
        "filter_duplicates": True,
        "heuristic1": "reverse-search",
        "heuristic1_delta": 1.0,
        # "mp_path": f"../new_format_motions/{dynamics}/{dynamics}.msgpack",
        # # "mp_path": f"data/output/unicycle2_v0/rand_0.yaml",
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
    instance = diffmp.problems.Instance.from_yaml(Path("../example/bugtrap.yaml"))
    instance = diffmp.problems.Instance.random(
        10, 10, 0.5, [dynamics] * n_robots, diffmp.problems.Dim.TWO_D
    )
    tmp1 = tempfile.NamedTemporaryFile()
    tmp2 = tempfile.NamedTemporaryFile()
    results = dbcbs_py.db_ecbs(
        instance.to_dict(),
        tmp1.name,
        tmp2.name,
        config,
        time_limit_db_astar,
        time_limit_db_cbs,
        True,
    )
    tmp1.close()
    tmp2.close()
    # costs = [result[i].optimized.trajectories[0].cost for i in range(len(result))]
    # durations = [result[i].runtime for i in range(len(result))]
    df_dict = {"x": [], "y": [], "name": []}
    costs = {"i": [], "cost": []}
    for i, result in enumerate(results):
        traj = result.optimized.trajectories[0]
        states = np.array(traj.states)
        # cost = np.ones(states.shape[0]) * traj.cost
        costs["i"].append(i)
        costs["cost"].append(traj.cost)
        name = [f"{result.delta} - {traj.cost}"] * states.shape[0]
        df_dict["x"].extend(states[:, 0])
        df_dict["y"].extend(states[:, 1])
        df_dict["name"].extend(name)

    df = pd.DataFrame(df_dict)
    costs_df = pd.DataFrame(costs)

    # print([len(r.optimized.trajectories[0].actions) for r in result])
    # print([len(r.discrete.trajectories[0].actions) for r in result])
    fig, ax = plt.subplots(ncols=2)
    instance.plot(ax=ax[0])

    sns.lineplot(df, x="x", y="y", hue="name", ax=ax[0], sort=False)
    sns.lineplot(costs_df, x="i", y="cost", ax=ax[1])
    plt.show()


if __name__ == "__main__":
    main()
