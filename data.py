from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List
from icecream import ic
import matplotlib.pyplot as plt
from scipy import stats

S_MIN = -1
S_MAX = 2
PHI_MIN = -np.pi / 3
PHI_MAX = np.pi / 3

SUPP_REG = [
    "actions",
    "theta_0",
    "delta_0",
]
SUPP_ENV = [
    "env_theta_start",
    "env_theta_goal",
    "area",
    "area_blocked",
    "area_free",
    "avg_clustering",
    "avg_node_connectivity",
    "avg_shortest_path",
    "avg_shortest_path_norm",
    "env_width",
    "env_height",
    "mean_size",
    "n_obstacles",
    "p_obstacles",
    "cost",
]
SUPP_COMPLETE = SUPP_REG + SUPP_ENV + ["rel_probability"]


class WeightSampler(stats.rv_continuous):
    def __init__(self, xtol=1e-14, seed=None):
        super().__init__(a=0, b=1, xtol=xtol, seed=seed)

    def _cdf(self, x, *args):
        # return 1 / (1 + (((1 - x) * 0.5) / (x * (1 - 0.5))) ** 2)
        # return 1-np.exp(-5*x**2)
        # return x**4
        return x**10


def pruneDataset(
    actions_data: np.ndarray, theta_0_data: np.ndarray, length: int, dt: float = 0.1
):
    def prune(data, limit: float):
        dataset = []
        statesCheck = []
        for actions, theta_0 in data:
            states = calc_unicycle_states(actions, dt=dt, start=[0, 0, float(theta_0)])
            if statesCheck:
                diff = np.linalg.norm(np.array(statesCheck) - states, axis=(1, 2))
                if (diff < limit).any():
                    continue
            statesCheck.append(states)
            dataset.append([actions, states])
        return dataset

    original_length = len(actions_data)
    if original_length == length:
        return prune(zip(actions_data, theta_0_data), limit=0), 0
    dataset = []
    dataset = prune(zip(actions_data, theta_0_data), limit=0.1)
    current_length = len(dataset)
    limit: float = (length - original_length) / (
        (current_length - original_length) / 0.1
    )
    scaling = 2
    last_length = original_length
    while True:
        dataset = prune(zip(actions_data, theta_0_data), limit=limit)
        current_length = len(dataset)

        ic(current_length, limit)
        if current_length == length:
            break
        elif current_length < length:
            limit *= 1 - scaling * np.abs(current_length - length) / original_length
        else:
            limit *= 1 + scaling * np.abs(current_length - length) / original_length
        if np.abs(current_length - length) < 2 and current_length != last_length:
            scaling *= 0.95
        last_length = current_length

    return dataset, limit


def calc_unicycle_states(
    actions: np.ndarray, dt: float = 0.1, start: List[float] = [0, 0, 0]
):
    x, y, theta = start
    states = [start]
    for s, phi in actions:
        dx = dt * s * np.cos(theta)
        dy = dt * s * np.sin(theta)
        dtheta = dt * phi

        x += dx
        y += dy
        theta += dtheta
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        states.append([x, y, theta])
    return np.array(states)


def get_header(dictionary: Dict[str, int]):
    return [
        f"{key}_{i}" if value > 1 else key
        for key, value in dictionary.items()
        for i in range(value)
    ]


def load_dataset(
    path: Path, regular: Dict[str, int], conditioning: Dict[str, int]
) -> np.ndarray:
    complete = regular | conditioning
    assert set(complete.keys()) <= set(
        SUPP_COMPLETE
    ), f"{[key for key in complete.keys() if key not in SUPP_COMPLETE]} are/is not implemented"

    regular_header = get_header(regular)
    conditioning_header = get_header(conditioning)
    columns = regular_header + conditioning_header

    rel_p = "rel_probability" in columns
    if rel_p:
        columns.remove(
            "rel_probability",
        )

    dataset = pd.read_parquet(path, columns=columns + ["count"])
    dataset = dataset.groupby(columns, as_index=False).agg({"count": "sum"})

    if rel_p:
        env_group = [env_key for env_key in SUPP_ENV if env_key in complete.keys()]
        if env_group:
            dataset["group_max"] = dataset.groupby(env_group)["count"].transform("max")
        else:
            dataset["group_max"] = dataset["count"].max()
        dataset["rel_probability"] = dataset["count"] / dataset["group_max"]
        dataset.drop(columns=["group_max", "count"], inplace=True)

    sorted_header = sorted(regular_header) + sorted(conditioning_header)
    dataset = dataset.reindex(sorted_header, axis=1)
    return_array = dataset.to_numpy()
    # ic(dataset.columns)
    # ic(dataset.min())
    # ic(dataset.max())

    return return_array


if __name__ == "__main__":
    ws = WeightSampler()
    data = ws.rvs(size=1000)
    plt.hist(data, density=True)
    pts = np.linspace(0.001, 1)
    plt.plot(pts, ws.pdf(pts), label="pdf")
    plt.plot(pts, ws.cdf(pts), label="cdf")
    plt.legend()
    plt.show()
