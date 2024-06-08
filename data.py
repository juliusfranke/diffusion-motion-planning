import numpy as np
from typing import List
from icecream import ic


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


def gen_car_action(
    s_range: List[float] = [-1, 2], phi_range: List[float] = [-np.pi / 3, np.pi / 3]
) -> np.ndarray:
    def log_piecewise(min, max, zero_bound=0.2):
        r = np.abs(min / (max - min))
        # ic(r)
        r = 1 / (1 + np.exp(-15 * (r - 0.5)))
        ic(r)

        choice = np.random.uniform(0, 1)
        s_low = -2 * min - zero_bound
        s_high = 2 * max - zero_bound
        if choice >= r:
            return s_high / (1 + np.exp(-10 * (choice - r))) + max - s_high
        else:
            return s_low / (1 + np.exp(-10 * (choice - r))) + min

    s_min, s_max = s_range
    phi_min, phi_max = phi_range
    s = log_piecewise(s_min, s_max)
    phi = log_piecewise(phi_min, phi_max, zero_bound=0)

    return np.array([s, phi])


def car_val(pred: np.ndarray, actions: np.ndarray):
    def metric(states: np.ndarray, pred: np.ndarray):
        err_xy = states[:, :2] - pred[:, :2]
        err_th = np.array(
            [
                np.minimum(
                    np.abs(states[:, 2] - pred[:, 2]),
                    2 * np.pi - np.abs(states[:, 2] - pred[:, 2]),
                )
            ]
        )
        # ic(sq_err_xy.shape, sq_err_th.shape)
        # sq_err = np.array([sq_err_xy, sq_err_th])
        sq_err = np.square(np.sum(np.concatenate([err_xy.T, err_th], axis=0), axis=0))

        return sq_err

    states = np.array([calc_car_state(action) for action in actions])
    # ic(actions, pred, states)
    error = metric(states, pred)
    mse = np.mean(error)
    return {"mse": mse, "error": error, "states": states}


def data_gen(length: int) -> np.ndarray:
    data = [
        np.concatenate([calc_car_state(action=action), action], axis=-1)
        for action in [gen_car_action() for _ in range(length)]
    ]
    max = np.max(data, axis=0)
    ic(max)

    return np.array(data)
