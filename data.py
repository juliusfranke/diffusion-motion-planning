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
    s_min, s_max = s_range
    phi_min, phi_max = phi_range

    s_choice = np.random.uniform(0, 1)
    phi_choice = np.random.uniform(0, 1)

    s = 4 / (1 + np.exp(-14 * (s_choice - 0.5))) - 2
    phi = (phi_max - phi_min) / (1 + np.exp(-10 * (phi_choice - 0.5))) - phi_max

    return np.array([s, phi])


def car_val(pred: np.ndarray, actions: np.ndarray):
    def metric(states: np.ndarray, pred: np.ndarray):
        sq_err_xy = np.square(states[:, :2] - pred[:, :2])
        sq_err_th = np.array(
            [
                np.square(
                    np.minimum(
                        np.abs(states[:, 2] - pred[:, 2]),
                        2 * np.pi - np.abs(states[:, 2] - pred[:, 2]),
                    )
                )
            ]
        )
        # ic(sq_err_xy.shape, sq_err_th.shape)
        # sq_err = np.array([sq_err_xy, sq_err_th])
        mse = np.mean(np.sum(np.concatenate([sq_err_xy.T, sq_err_th], axis=0), axis=0))
        return mse

    states = np.array([calc_car_state(action) for action in actions])
    # ic(actions, pred, states)

    return {"mse": metric(states, pred), "states": states}


def data_gen(length: int) -> np.ndarray:
    data = [
        np.concatenate([calc_car_state(action=action), action], axis=-1)
        for action in [gen_car_action() for _ in range(length)]
    ]
    max = np.max(data, axis=0)
    ic(max)

    return np.array(data)
