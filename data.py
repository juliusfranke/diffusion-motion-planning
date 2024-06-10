import numpy as np
from typing import List
from icecream import ic

S_MIN = -1
S_MAX = 2
PHI_MIN = -np.pi / 3
PHI_MAX = np.pi / 3


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
            return s_high / (1 + np.exp(-5 / r * (choice - r))) + max - s_high
        else:
            return s_low / (1 + np.exp(-5 / r * (choice - r))) + min

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
    sq_err = np.square(np.sum(np.concatenate([err_xy.T, err_th], axis=0), axis=0))

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
        data.append(np.concatenate([state, action], axis=-1))
        # ic(len(data))
        if len(data) % 100 == 0:
            ic(len(data) + 1)
    return np.array(data)
