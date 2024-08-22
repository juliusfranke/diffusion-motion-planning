import time
from typing import Dict

import numpy as np
import torch.nn as nn
import torch.utils.data
from icecream import ic
from torch.utils.data import DataLoader
from tqdm import tqdm

N_SEQ = 1_000
EPOCHS = 200
N_SAMPLES = 20
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.input: Dict[str, int] = config["regular"]
        self.conditioning: Dict[str, int] = config["conditioning"]
        self.validate: Dict[str, int] = self.input | self.conditioning

        self.output_size = sum(self.input.values())
        self.condition_size = sum(self.conditioning.values())
        self.input_size = self.output_size + self.condition_size + 1

        ic(self.input_size, self.output_size)
        layers = [nn.Linear(self.input_size, config["s_hidden"])]
        for _ in range(config["n_hidden"]):
            layers.append(nn.Linear(config["s_hidden"], config["s_hidden"]))
        layers.append(nn.Linear(config["s_hidden"], self.output_size))
        self.linears = nn.ModuleList(layers)

        for layer in self.linears:
            nn.init.kaiming_uniform_(layer.weight)
        self.loss_fn = {"actions": self._loss_actions, "theta_0": self._loss_theta_0}

    def loss(self, output: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        idx = 0
        loss = torch.zeros(output.shape, device=DEVICE)
        for key, value in self.input.items():
            idx_to = value + idx
            loss[:, idx:idx_to] = self.loss_fn[key](
                output[:, idx:idx_to], noise[:, idx:idx_to]
            )
            idx += value
        return torch.mean(loss)

    def forward(self, x, t):
        x = torch.concat([x, t], dim=-1)

        for layer in self.linears[:-1]:
            x = nn.ReLU()(layer(x))
        return self.linears[-1](x)

    def _loss_actions(self, output: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        assert output.shape[1] == 10
        assert noise.shape[1] == 10

        return (noise - output) ** 2

    def _loss_theta_0(self, output: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        assert output.shape[1] == 1
        assert noise.shape[1] == 1

        difference = noise - output
        x = torch.cos(difference)
        y = torch.sin(difference)

        return (torch.atan2(y, x) / np.pi) ** 2


def get_alpha_betas(N: int):
    """Schedule from the original paper. Commented out is sigmoid schedule from:

    'Score-Based Generative Modeling through Stochastic Differential Equations.'
     Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar,
     Stefano Ermon, Ben Poole (https://arxiv.org/abs/2011.13456)
    """
    beta_min = 0.1
    beta_max = 20.0
    betas = np.array(
        [beta_min / N + i / (N * (N - 1)) * (beta_max - beta_min) for i in range(N)]
    )
    # betas = np.random.uniform(10e-4, .02, N)  # schedule from the 2020 paper
    alpha_bars = np.cumprod(1 - betas)
    return alpha_bars, betas


def train(
    model: Net,
    loader: DataLoader,
    config: Dict,
    nepochs: int = 10,
    denoising_steps: int = 100,
):
    """Alg 1 from the DDPM paper"""
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    alpha_bars, _ = get_alpha_betas(denoising_steps)  # Precompute alphas
    losses = []

    output_size = model.output_size
    pbar = tqdm(range(nepochs))
    mean_loss = 0
    for epoch in pbar:
        for [data] in loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()

            # Fwd pass
            t = torch.randint(
                denoising_steps, size=(data.shape[0],)
            )  # sample timesteps - 1 per datapoint
            alpha_t = (
                torch.index_select(torch.Tensor(alpha_bars), 0, t)
                .unsqueeze(1)
                .to(DEVICE)
            )  # Get the alphas for each timestep

            noise = torch.randn(
                *data.shape, device=DEVICE
            )  # Sample DIFFERENT random noise for each datapoint
            model_in = (
                alpha_t**0.5 * data + noise * (1 - alpha_t) ** 0.5
            )  # Noise corrupt the data (eq14)
            out = model(model_in, t.unsqueeze(1).to(DEVICE))
            loss = torch.mean(
                (noise[:, :output_size] - out) ** 2
            )  # Compute loss on prediction (eq14)
            loss_new = model.loss(out, noise)
            loss = loss_new
            losses.append(loss.detach().cpu().numpy())

            # Bwd pass
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            mean_loss = np.mean(np.array(losses))
            pbar.set_description(f"Mean loss = {mean_loss:.5f}")

    return model


def sample(
    trained_model: Net,
    n_samples: int,
    n_steps: int = 100,
    conditioning: torch.Tensor | None = None,
):
    """Alg 2 from the DDPM paper."""
    x_t = torch.randn((n_samples, trained_model.output_size)).to(DEVICE)

    alpha_bars, betas = get_alpha_betas(n_steps)
    alphas = 1 - betas
    for t in range(len(alphas))[::-1]:
        ts = t * torch.ones((n_samples, 1)).to(DEVICE)
        ab_t = alpha_bars[t] * torch.ones((n_samples, 1)).to(
            DEVICE
        )  # Tile the alpha to the number of samples
        z = (
            torch.randn((n_samples, trained_model.output_size))
            if t > 1
            else torch.zeros((n_samples, trained_model.output_size))
        ).to(DEVICE)

        if trained_model.condition_size != 0 and isinstance(conditioning, torch.Tensor):
            conditioning = conditioning.reshape(-1, 1)
            model_prediction = trained_model(
                torch.concat([x_t, conditioning], dim=-1), ts
            )
            x_t = (
                1
                / alphas[t] ** 0.5
                * (
                    x_t[:, : trained_model.output_size]
                    - betas[t] / (1 - ab_t) ** 0.5 * model_prediction
                )
            )
            x_t += betas[t] ** 0.5 * z
        else:
            model_prediction = trained_model(x_t, ts)
            x_t = (
                1
                / alphas[t] ** 0.5
                * (x_t - betas[t] / (1 - ab_t) ** 0.5 * model_prediction)
            )
            x_t += betas[t] ** 0.5 * z

    return x_t


if __name__ == "__main__":
    pass
