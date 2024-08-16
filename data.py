from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List
from icecream import ic
from shapely import is_empty
import yaml
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy import stats

S_MIN = -1
S_MAX = 2
PHI_MIN = -np.pi / 3
PHI_MAX = np.pi / 3


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
    # ic(x, y, theta)
    # if actions.shape != (5,2):
    #     actions = actions.reshape(5, 2)
    # ic(x,y,theta)
    states = [start]
    for s, phi in actions:
        # ic(s,phi)
        dx = dt * s * np.cos(theta)
        dy = dt * s * np.sin(theta)
        dtheta = dt * phi

        # breakpoint()
        x += dx
        y += dy
        theta += dtheta
        # breakpoint()
        states.append([x, y, theta])
    return np.array(states)


def calc_car_state(
    action: np.ndarray,
    dt: float = 0.1,
    L: float = 1,
    start: List[float] = [0, 0, 0],
    steps: int = 5,
) -> np.ndarray:
    x, y, theta = start
    s, phi = action

    for _ in range(steps):
        dx = s * np.cos(theta)
        dy = s * np.sin(theta)
        dtheta = s / L * np.tan(phi)

        x += dt * dx
        y += dt * dy
        theta += dt * dtheta

    return np.array([x, y, theta])


def gen_car_state_area(samples=100):
    s_0 = np.linspace(S_MIN, S_MAX, samples // 4)
    s_1 = np.array(samples // 4 * [S_MAX])
    s_2 = np.flip(s_0)
    s_3 = np.array(samples // 4 * [S_MIN])
    phi_0 = np.array(samples // 4 * [PHI_MIN])
    phi_1 = np.linspace(PHI_MIN, PHI_MAX, samples // 4)
    phi_2 = np.array(samples // 4 * [PHI_MAX])
    phi_3 = np.flip(phi_1)

    s = np.array([*s_0, *s_1, *s_2, *s_3])
    phi = np.array([*phi_0, *phi_1, *phi_2, *phi_3])

    data = np.vstack((s, phi)).T
    # states = [state for calc_car_state(
    states = [
        (state[0], state[1]) for state in [calc_car_state(action) for action in data]
    ]
    return states


def gen_car_action() -> np.ndarray:
    def log_piecewise(min, max, zero_bound=0.2):
        r = np.abs(min / (max - min))
        r = 1 / (1 + np.exp(-5 * (r - 0.5)))

        choice = np.random.uniform(0, 1)
        s_low = -2 * min - zero_bound
        s_high = 2 * max - zero_bound
        if choice >= r:
            return s_high / (1 + np.exp(-7 / r * (choice - r))) + max - s_high
        else:
            return s_low / (1 + np.exp(-7 / r * (choice - r))) + min

    s = log_piecewise(S_MIN, S_MAX)
    phi = log_piecewise(PHI_MIN, PHI_MAX, zero_bound=0)

    return np.array([s, phi])


def metric(a: np.ndarray, b: np.ndarray) -> float:
    err_xy = a[:, :2] - b[:, :2]
    err_th = np.array(
        [
            np.minimum(
                np.abs(a[:, 2] - b[:, 2]),
                2 * np.pi - np.abs(a[:, 2] - b[:, 2]),
            )
        ]
    )
    sq_err = np.sum(np.square(np.concatenate([err_xy.T, err_th], axis=0)), axis=0)

    return sq_err


def car_val(pred: np.ndarray, actions: np.ndarray):
    states = np.array([calc_car_state(action) for action in actions])
    error = metric(states, pred)
    mse = np.mean(error)
    return {"mse": mse, "error": error, "states": states}


def data_gen(length: int) -> np.ndarray:
    # data = [
    #     np.concatenate([calc_car_state(action=action), action], axis=-1)
    #     for action in [gen_car_action() for _ in range(length)]
    # ]
    data = []
    while True:
        if len(data) == length:
            break
        action = gen_car_action()
        state = calc_car_state(action=action)
        if data:
            min_dist = np.min(
                np.linalg.norm(
                    np.array(data)[:, :2] - np.array([state])[:, :2], axis=-1
                )
            )
            # ic(min_dist)
            if min_dist < 0.015:
                continue
        state_action = np.concatenate([state, action], axis=-1)
        noise = np.random.normal(0, 5e-3, state_action.shape[0])
        data.append(state_action + noise)

        # ic(len(data))
        if len(data) % 100 == 0:
            ic(len(data))
    return np.array(data)


def circle_SO2(theta_range, n=100):
    theta = np.linspace(-theta_range, theta_range, n).reshape(-1, 1)
    start = np.concatenate([np.zeros((n, 2)), theta], axis=1)
    goal = np.concatenate([0.25 * np.cos(theta), 0.25 * np.sin(theta), theta], axis=1)
    # ic(start, goal, theta)
    diff = np.array(
        [calc_diff_SO2(start_i, goal_i) for start_i, goal_i in zip(start, goal)]
    )
    data = {"start": start, "goal": goal, "diff": diff}
    return data


def calc_diff_SO2(start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    def diff(start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        rot = -start[2]
        phi = goal[2] + rot
        # print(rot, phi)
        M_T = np.array(
            [
                [np.cos(rot), -np.sin(rot)],
                [np.sin(rot), np.cos(rot)],
            ]
        )
        return np.array([*(M_T @ (goal[:2] - start[:2])), phi])

    if len(goal.shape) == 1:
        return diff(start, goal)
    else:
        return np.array([diff(s, g) for s, g in zip(start, goal)])


def prune(data: np.ndarray, delta: float) -> np.ndarray:
    returnList = []
    for i in range(data.shape[0]):
        entry = data[i, :]
        if returnList:
            diff = np.linalg.norm(np.array(returnList) - entry, axis=1)
            if (diff <= delta).any():
                continue
        returnList.append(entry)

    return np.array(returnList)


def read_yaml(path: Path, **kwargs: int) -> np.ndarray:
    supported_kwargs = [
        "start",
        "goal",
        "actions",
        "states",
        "diff",
        "theta_0",
        "rel_probability",
    ]
    assert (
        set(kwargs.keys()) <= set(supported_kwargs)
    ), f"{[key for key in kwargs.keys() if key not in supported_kwargs]} are/is not implemented"

    with open(path, "r") as file:
        data = yaml.safe_load(file)

    return_array = np.array([])
    key_data = np.array([])

    for key, value in kwargs.items():
        if key in supported_kwargs[:4]:
            key_data = np.array([np.array(mp[key]).flatten() for mp in data])
        elif key == "diff":
            start = np.array([np.array(mp["start"]).flatten() for mp in data])
            goal = np.array([np.array(mp["goal"]).flatten() for mp in data])
            # breakpoint()
            key_data = calc_diff_SO2(start, goal)
        elif key == "theta_0":
            key_data = np.array([np.array(mp["start"])[2] for mp in data]).reshape(
                -1, 1
            )
        elif key == "rel_probability":
            key_data = np.array([np.array(mp[key]) for mp in data]).reshape(-1, 1)
        else:
            raise NotImplementedError

        assert key_data.shape[1] == value
        if not return_array.size:
            return_array = key_data
        else:
            return_array = np.concatenate([return_array, key_data], axis=-1)

    return return_array


def spiral_points(n=100, arc=0.25, separation=0.5):
    """generate points on an Archimedes' spiral
    with `arc` giving the length of arc between two points
    and `separation` giving the distance between consecutive
    turnings
    - approximate arc length with circle arc at given distance
    - use a spiral equation r = b * phi
    """

    def p2c(r, phi):
        """polar to cartesian"""
        return [r * np.cos(phi), r * np.sin(phi), (phi + np.pi / 2) % (2 * np.pi)]

    # yield a point at origin
    states = []
    diff = []
    # initialize the next point in the required distance
    r = arc
    b = separation / (2 * np.pi)
    # find the first phi to satisfy distance of `arc` to the second point
    phi = float(r) / b
    for _ in range(n + 1):
        next_state = p2c(r, phi)
        states.append(next_state)
        # advance the variables
        # calculate phi that will give desired arc length at current radius
        # (approximating with circle)
        phi += float(arc) / r
        r = b * phi
    states = np.array(states)
    start = states[:-1]
    goal = states[1:]

    diff = np.array(
        [calc_diff_SO2(start_i, goal_i) for start_i, goal_i in zip(start, goal)]
    )
    ic(len(start))
    data = {"start": start, "goal": goal, "diff": diff}
    return data


if __name__ == "__main__":
    # data = circle_SO2(8)
    # ic(data)
    ws = WeightSampler()
    data = ws.rvs(size=1000)
    plt.hist(data, density=True)
    pts = np.linspace(0.001, 1)
    plt.plot(pts, ws.pdf(pts), label="pdf")
    plt.plot(pts, ws.cdf(pts), label="cdf")
    plt.legend()
    plt.show()
    # data = read_yaml("data/my_motions.bin.im.bin.sp.bin.yaml")
    # # s = []
    # phi_max = 0
    # phi_min = 0
    # for actions in data["actions"]:
    #     actions = actions.reshape(5, 2)
    #     min = np.min(actions[:, 0])
    #     max = np.max(actions[:, 0])
    #     if min < phi_min:
    #         phi_min = min
    #     if max > phi_max:
    #         phi_max = max

    # ic(phi_min, phi_max)
    # # breakpoint()
    # for sample in data["states"]:
    #     sample = sample.reshape(6, 3)
    #     plt.plot(sample[:, 0], sample[:, 1])
    # plt.show()

    # p = spiral_points()
    # points = np.array([next(p) for _ in range(20)])
    # # ic(points)
    # u = np.cos(points[:, 2])
    # v = np.sin(points[:, 2])

    # # ic(u, v)
    # plt.plot(points[:, 0], points[:, 1])
    # plt.quiver(points[:, 0], points[:, 1], u, v)
    # plt.show()
