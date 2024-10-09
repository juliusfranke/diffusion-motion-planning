from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List
from icecream import ic
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import NearestNeighbors
import torch
from geomloss import SamplesLoss

S_MIN = -1
S_MAX = 2
PHI_MIN = -np.pi / 3
PHI_MAX = np.pi / 3

SUPP_REG = [
    "actions",
    "theta_0",
    "theta_0_x",
    "theta_0_y",
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
    "location",
]
SUPP_COMPLETE = SUPP_REG + SUPP_ENV + ["rel_probability"]

sinkhorn_loss = SamplesLoss("sinkhorn", p=2, blur=0.05)


def sinkhorn(real: torch.Tensor, pred: torch.Tensor):
    pred = pred.to(torch.float64)
    return sinkhorn_loss(real, pred)


def mae(real: torch.Tensor, pred: torch.Tensor):
    error = torch.abs(real - pred)
    return torch.mean(error)


def mse(real: torch.Tensor, pred: torch.Tensor):
    error = real - pred
    return torch.mean(error**2)


def mse_theta(real: torch.Tensor, pred: torch.Tensor):
    action_error = (real[:, :10] - pred[:, :10]) ** 2
    difference = real[:, 10] - pred[:, 10]
    x = torch.cos(difference)
    y = torch.sin(difference)
    theta_error = torch.atan2(y, x) ** 2

    return torch.mean(torch.concat([action_error.flatten(), theta_error]))


def clipped_mse_loss(real, pred, min_val, max_val):
    pred_clipped = torch.clamp(pred, min_val, max_val)
    return torch.mean((real - pred_clipped) ** 2)


def log_cosh_loss(real, pred):
    return torch.mean(torch.log(torch.cosh(pred - real)))


def weighted_mse_loss(real, pred, min_val, max_val):
    weights = torch.abs(real - (min_val + max_val) / 2)
    return torch.mean(weights * (pred - real) ** 2)


def boundary_aware_loss(real, pred, min_val, max_val):
    mse_loss = torch.mean((pred - real) ** 2)
    min_val = min_val[real.shape[1]]
    max_val = max_val[real.shape[1]]

    boundary_penalty = torch.mean(
        torch.relu(pred - max_val) + torch.relu(min_val - pred)
    )

    return mse_loss + boundary_penalty


def wasserstein_distance_pytorch(real_data, generated_data):
    """Compute the Wasserstein distance between real and generated data for each dimension."""
    n_features = real_data.shape[1]
    distances = []
    for i in range(n_features):
        distances.append(
            stats.wasserstein_distance(real_data[:, i], generated_data[:, i])
        )
    return torch.tensor(distances).mean().item()


def rbf_kernel(x, y, sigma=1.0):
    """Compute the RBF kernel between two sets of data points."""
    return torch.exp(-torch.cdist(x, y) ** 2 / (2 * sigma**2))


def mmd_rbf(real_data, generated_data, sigma=1.0):
    """Compute the Maximum Mean Discrepancy (MMD) with RBF kernel."""
    generated_data = generated_data.to(torch.float64)
    xx = rbf_kernel(real_data, real_data, sigma)
    yy = rbf_kernel(generated_data, generated_data, sigma)
    xy = rbf_kernel(real_data, generated_data, sigma)

    mmd = xx.mean() + yy.mean() - 2 * xy.mean()
    return mmd


def precision_recall_coverage(real_data, generated_data, k=5, threshold=0.5):
    """Compute Precision, Recall, and Coverage for generative models."""
    nbrs_real = NearestNeighbors(n_neighbors=k).fit(real_data)
    nbrs_gen = NearestNeighbors(n_neighbors=k).fit(generated_data)

    # Precision: Fraction of generated points within the manifold of real data
    distances_gen, _ = nbrs_real.kneighbors(generated_data)
    precision = (distances_gen.mean(axis=1) < threshold).mean()

    # Recall: Fraction of real points within the manifold of generated data
    distances_real, _ = nbrs_gen.kneighbors(real_data)
    recall = (distances_real.mean(axis=1) < threshold).mean()

    # Coverage: Fraction of real data points covered by generated points
    coverage = (distances_real.min(axis=1) < threshold).mean()

    return precision, recall, coverage


class WeightSampler(stats.rv_continuous):
    def __init__(self, xtol=1e-14, seed=None):
        super().__init__(a=0, b=1, xtol=xtol, seed=seed)

    def _cdf(self, x, *args):
        # return 1 / (1 + (((1 - x) * 0.5) / (x * (1 - 0.5))) ** 2)
        # return 1-np.exp(-5*x**2)
        # return x**4
        return x**10


def get_violations(samples):
    samples = samples[:, :10]
    is_violation = np.abs(samples) > 0.5

    total_violations = np.sum(is_violation)
    total = np.prod(samples.shape)
    violations = total_violations / total
    violation_score = np.sum(is_violation * (np.abs(samples[:, :10]) - 0.5)) / total
    return violations, violation_score


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
    dataset = dataset.round(decimals=2)
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
    # dataset = dataset[dataset["rel_probability"] > 0.6]
    # s = [f"actions_{i}" for i in [0, 2, 4, 6, 8]]
    # for ac in s:
    #     dataset = dataset[dataset[ac] > 0]
    # breakpoint()

    # phi = [f"actions_{i}" for i in [1, 3, 5, 7, 9]]
    # s = [f"actions_{i}" for i in [0, 2, 4, 6, 8]]
    # dataset[s] = (dataset[s] - dataset[s].mean().mean()) / dataset[s].std().mean()
    # dataset[phi] = (dataset[phi] - dataset[phi].mean().mean()) / dataset[
    #     phi
    # ].std().mean()
    # dataset["theta_0"] = (dataset["theta_0"] - dataset["theta_0"].mean()) / dataset[
    #     "theta_0"
    # ].std()

    # dataset = (dataset - dataset.mean()) / dataset.std()
    # return_array = dataset.to_numpy()
    # ic(dataset.columns)
    # ic(dataset.min())
    # ic(dataset.max())

    return dataset


if __name__ == "__main__":
    ws = WeightSampler()
    data = ws.rvs(size=1000)
    plt.hist(data, density=True)
    pts = np.linspace(0.001, 1)
    plt.plot(pts, ws.pdf(pts), label="pdf")
    plt.plot(pts, ws.cdf(pts), label="cdf")
    plt.legend()
    plt.show()
