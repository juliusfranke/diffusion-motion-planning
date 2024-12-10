from typing import Tuple

import numpy as np
from nptyping import NDArray


def cosine_noise_schedule(N: int) -> Tuple[NDArray, NDArray]:
    timesteps = np.linspace(0, 1, N)
    alpha_bars = np.cos((timesteps * np.pi / 2)) ** 2

    alphas = alpha_bars[1:] / alpha_bars[:-1]
    betas = 1 - alphas
    return alpha_bars, betas


def sigmoid_noise_schedule(N: int) -> Tuple[NDArray, NDArray]:
    beta_min = 0.01
    beta_max = 0.2
    k = 15

    timesteps = np.linspace(0, 1, N)
    sigmoid = 1 / (1 + np.exp(-k * (timesteps - 0.5)))
    betas = beta_min + (beta_max - beta_min) * sigmoid

    alpha_bars = np.cumprod(1 - betas)
    return alpha_bars, betas


def linear_noise_schedule(N: int) -> Tuple[NDArray, NDArray]:
    beta_min = 0.001
    beta_max = 0.2
    betas = np.linspace(beta_min, beta_max, N)
    alpha_bars = np.cumprod(1 - betas)
    return alpha_bars, betas


def linear_noise_schedule_scaled(N: int) -> Tuple[NDArray, NDArray]:
    beta_min = 0.1
    beta_max = 20.0
    betas = np.array(
        [beta_min / N + i / (N * (N - 1)) * (beta_max - beta_min) for i in range(N)]
    )

    alpha_bars = np.cumprod(1 - betas)
    return alpha_bars, betas
