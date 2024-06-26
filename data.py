import numpy as np
from typing import List
from icecream import ic
import yaml
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

S_MIN = -1
S_MAX = 2
PHI_MIN = -np.pi / 3
PHI_MAX = np.pi / 3


def calc_unicycle_states(
    actions: np.ndarray, dt: float = 0.1, start: List[float] = [0, 0, 0]
):
    x, y, theta = start
    # ic(x, y, theta)
    actions = actions.reshape(5, 2)

    states = [start]
    for s, phi in actions:
        dx = dt * s * np.cos(theta)
        dy = dt * s * np.sin(theta)
        dtheta = dt * phi

        x += dx
        y += dy
        theta += dtheta
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


def read_yaml(path):
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    states = []
    actions = []
    start = []
    goal = []
    # ic(data)
    for sample in data:
        # offset = np.array([*np.random.uniform(-10, 10, 2), 0])
        offset = 0

        start.append(np.array(sample["start"]) + offset)
        goal.append(np.array(sample["goal"]) + offset)
        sample_state = np.array(sample["states"]) + offset
        states.append([st for sts in sample_state for st in sts])
        actions.append([ac for acs in sample["actions"] for ac in acs])
    states = np.array(states)
    actions = np.array(actions)
    # breakpoint()
    start = np.array(start)
    goal = np.array(goal)

    states, actions, start, goal = shuffle(states, actions, start, goal)
    # print(states)
    ic(start.shape, goal.shape)
    data_dict = {
        "states": states,
        "actions": actions,
        "start": start,
        "goal": goal,
        "state_dim": len(states[0]),
        "action_dim": len(actions[0]),
    }
    return data_dict


def spiral_points(arc=0.25, separation=0.5):
    """generate points on an Archimedes' spiral
    with `arc` giving the length of arc between two points
    and `separation` giving the distance between consecutive
    turnings
    - approximate arc length with circle arc at given distance
    - use a spiral equation r = b * phi
    """

    def p2c(r, phi):
        """polar to cartesian"""
        return (r * np.cos(phi), r * np.sin(phi), (phi + np.pi / 2) % (2 * np.pi))

    # yield a point at origin
    yield (0, 0, 0)

    # initialize the next point in the required distance
    r = arc
    b = separation / (2 * np.pi)
    # find the first phi to satisfy distance of `arc` to the second point
    phi = float(r) / b
    while True:
        yield p2c(r, phi)
        # advance the variables
        # calculate phi that will give desired arc length at current radius
        # (approximating with circle)
        phi += float(arc) / r
        r = b * phi


if __name__ == "__main__":
    data = read_yaml("data/my_motions.bin.im.bin.sp.bin.yaml")
    # s = []
    phi_max = 0
    phi_min = 0
    for actions in data["actions"]:
        actions = actions.reshape(5, 2)
        min = np.min(actions[:, 0])
        max = np.max(actions[:, 0])
        if min < phi_min:
            phi_min = min
        if max > phi_max:
            phi_max = max

    ic(phi_min, phi_max)
    # breakpoint()
    for sample in data["states"]:
        sample = sample.reshape(6, 3)
        plt.plot(sample[:, 0], sample[:, 1])
    plt.show()

    # p = spiral_points()
    # points = np.array([next(p) for _ in range(20)])
    # # ic(points)
    # u = np.cos(points[:, 2])
    # v = np.sin(points[:, 2])

    # # ic(u, v)
    # plt.plot(points[:, 0], points[:, 1])
    # plt.quiver(points[:, 0], points[:, 1], u, v)
    # plt.show()
