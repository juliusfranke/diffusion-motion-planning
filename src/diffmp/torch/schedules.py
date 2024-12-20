from enum import Enum
from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt


def cosine_noise_schedule(
    N: int,
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    timesteps = np.linspace(0, 1, N)
    alpha_bars = np.square(np.cos((timesteps * np.pi / 2), dtype=np.float64))

    alphas = alpha_bars[1:] / alpha_bars[:-1]
    betas = 1 - alphas
    return alpha_bars, betas


def sigmoid_noise_schedule(
    N: int,
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    beta_min = 0.01
    beta_max = 0.2
    k = 15

    timesteps = np.linspace(0, 1, N)
    sigmoid = 1 / (1 + np.exp(-k * (timesteps - 0.5)))
    betas = beta_min + (beta_max - beta_min) * sigmoid

    alpha_bars = np.cumprod(1 - betas)
    return alpha_bars, betas


def linear_noise_schedule(
    N: int,
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    beta_min = 0.001
    beta_max = 0.2
    betas = np.linspace(beta_min, beta_max, N)
    alpha_bars = np.cumprod(1 - betas)
    return alpha_bars, betas


def linear_noise_schedule_scaled(
    N: int,
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    beta_min = 0.1
    beta_max = 20.0
    betas = np.array(
        [beta_min / N + i / (N * (N - 1)) * (beta_max - beta_min) for i in range(N)]
    )

    alpha_bars = np.cumprod(1 - betas, dtype=np.float64)
    return alpha_bars, betas


class NoiseScheduleFunction:
    def __init__(
        self,
        func: Callable[
            [int], Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
        ],
    ):
        self.func = func

    def __call__(
        self, N: int
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        return self.func(N)


class NoiseSchedule(Enum):
    linear = NoiseScheduleFunction(linear_noise_schedule)
    linear_scaled = NoiseScheduleFunction(linear_noise_schedule_scaled)
    cosine = NoiseScheduleFunction(cosine_noise_schedule)
    sigmoid = NoiseScheduleFunction(sigmoid_noise_schedule)
