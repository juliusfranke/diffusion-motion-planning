import time
from typing import Dict, List

import numpy as np
import torch.nn as nn
import torch.utils.data
from icecream import ic
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm._utils import _term_move_up
from pathlib import Path
import logging
from data import (
    get_violations,
    precision_recall_coverage,
    mmd_rbf,
    wasserstein_distance_pytorch,
)
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

N_SEQ = 1_000
EPOCHS = 200
N_SAMPLES = 20
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        logger.debug("Creating Net")
        self.regular: Dict[str, int] = config["regular"]
        self.conditioning: Dict[str, int] = config["conditioning"]
        self.validate: Dict[str, int] = self.regular | self.conditioning

        self.output_size = sum(self.regular.values())
        self.condition_size = sum(self.conditioning.values())
        self.input_size = self.output_size + self.condition_size + 1

        self.info = config.copy()
        self.path: None | Path = None

        logger.debug(f"Input size : {self.input_size}")
        logger.debug(f"Output size : {self.output_size}")

        logger.debug("Input")
        for regular in self.regular.keys():
            logger.debug(f"{regular} : {self.regular[regular]}")

        logger.debug("Conditioning")
        for condition in self.conditioning.keys():
            logger.debug(f"{condition} : {self.conditioning[condition]}")

        layers = [nn.Linear(self.input_size, config["s_hidden"], dtype=torch.float64)]
        for _ in range(config["n_hidden"]):
            layers.append(
                nn.Linear(config["s_hidden"], config["s_hidden"], dtype=torch.float64)
            )
        layers.append(
            nn.Linear(config["s_hidden"], self.output_size, dtype=torch.float64)
        )
        self.linears = nn.ModuleList(layers)

        for layer in self.linears:
            nn.init.kaiming_uniform_(layer.weight)
        self.loss_fn = {"actions": self._loss_actions, "theta_0": self._loss_theta_0}
        logger.debug("Created Net")

    def save(self, epoch, pbar: tqdm):
        if isinstance(self.path, Path):
            torch.save(self.state_dict(), self.path)
            pbar.set_description(f"Saved @ epoch {epoch}")
        else:
            logger.error("Model path not set")
            raise Exception

    def loss(self, output: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return torch.mean((noise[:, : self.output_size] - output) ** 2)
        # idx = 0
        # loss = torch.zeros(output.shape, device=DEVICE)
        # for key, value in self.regular.items():
        #     idx_to = value + idx
        #     loss[:, idx:idx_to] = self.loss_fn[key](
        #         output[:, idx:idx_to], noise[:, idx:idx_to]
        #     )
        #     idx += value
        # return torch.mean(loss)

    def forward(self, x):
        # x = torch.concat([x, t], dim=-1)

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
        rel_dist = (torch.atan2(y, x) / np.pi) ** 2
        # penalty = torch.gt(torch.abs(output), np.pi) * (torch.abs(output))**10
        # rel_dist += penalty
        return rel_dist


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


def train_epoch(
    training_loader: DataLoader,
    model: Net,
    optimizer,
    alpha_bars,
    denoising_steps: int,
):

    output_size = model.output_size
    running_loss = 0.0
    for i, [data] in enumerate(training_loader):
        data = data.to(DEVICE)
        optimizer.zero_grad()

        # Fwd pass
        t = torch.randint(
            denoising_steps, size=(data.shape[0],)
        )  # sample timesteps - 1 per datapoint
        alpha_t = (
            torch.index_select(torch.Tensor(alpha_bars), 0, t).unsqueeze(1).to(DEVICE)
        )  # Get the alphas for each timestep

        noise = torch.randn(
            *data.shape, device=DEVICE
        )  # Sample DIFFERENT random noise for each datapoint
        model_in = (
            alpha_t**0.5 * data + noise * (1 - alpha_t) ** 0.5
        )  # Noise corrupt the data (eq14)

        x = torch.concat([model_in, t.unsqueeze(1).to(DEVICE)], dim=-1)
        out = model(x)
        loss = torch.mean(
            (noise[:, :output_size] - out) ** 2
        )  # Compute loss on prediction (eq14)
        loss = model.loss(out, noise)

        # losses.append(loss.detach().cpu().numpy())
        running_loss += loss.detach().cpu().numpy()
        # Bwd pass
        loss.backward()
        optimizer.step()

    model.info["train_metrics"]["loss/train"].append(running_loss / (i + 1))


def val_epoch(
    validation_loader: DataLoader,
    model: Net,
):

    running_vloss = 0.0
    running_violations = 0.0
    running_violation_score = 0.0

    with torch.no_grad():
        for i, [vdata] in enumerate(validation_loader):
            regular = vdata[:, : model.output_size]
            conditioning = vdata[:, model.output_size :]
            pred = sample(model, len(vdata), conditioning=conditioning)
            val_loss = model.loss(regular, pred).detach().cpu().numpy()
            running_vloss += val_loss
            violations, violation_score = get_violations(pred.detach().cpu().numpy())
            running_violations += violations
            running_violation_score += violation_score

    avg_vloss = running_vloss / (i + 1)
    avg_violations = running_violations / (i + 1)
    avg_violation_score = running_violation_score / (i + 1)

    model.info["val_metrics"]["loss/val"].append(avg_vloss)
    model.info["val_metrics"]["viol/percent"].append(avg_violations)
    model.info["val_metrics"]["viol/avg"].append(avg_violation_score)


def test_epoch(
    test_loader: DataLoader,
    model: Net,
):

    assert len(test_loader) == 1
    with torch.no_grad():
        for [test_data] in test_loader:
            regular = test_data[:, : model.output_size]
            conditioning = test_data[:, model.output_size :]
            pred = sample(model, len(test_data), conditioning=conditioning)
            # mmd = mmd_rbf(regular, pred)
            emd = wasserstein_distance_pytorch(regular, pred)

    # model.info["test_metrics"]["loss/mmd"].append(mmd)
    model.info["test_metrics"]["loss/emd"].append(emd)


def train(
    model: Net,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    tb_writer: SummaryWriter,
    nepochs: int = 10,
    denoising_steps: int = 100,
):
    """Alg 1 from the DDPM paper"""
    model.to(DEVICE)
    lr = 1e-3
    model.info["lr"] = lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    alpha_bars, _ = get_alpha_betas(denoising_steps)  # Precompute alphas

    pbar = tqdm(range(nepochs))
    model.info["train_metrics"] = {"loss/train": []}
    model.info["val_metrics"] = {
        "loss/val": [],
        "viol/percent": [],
        "viol/avg": [],
    }
    model.info["test_metrics"] = {"loss/emd": []}
    try:
        for epoch in pbar:
            train_epoch(
                training_loader,
                model,
                optimizer,
                alpha_bars,
                denoising_steps,
            )
            for train_metric in model.info["train_metrics"].keys():
                tb_writer.add_scalar(
                    train_metric,
                    model.info["train_metrics"][train_metric][-1],
                    epoch + 1,
                )
            if (epoch + 1) % 10 != 0:
                continue
            model.eval()
            val_epoch(validation_loader, model)
            for val_metric in model.info["val_metrics"].keys():
                tb_writer.add_scalar(
                    val_metric, model.info["val_metrics"][val_metric][-1], epoch + 1
                )
            if (epoch + 1) % 20 != 0:
                continue
            test_epoch(test_loader, model)
            for test_metric in model.info["test_metrics"].keys():
                tb_writer.add_scalar(
                    test_metric, model.info["test_metrics"][test_metric][-1], epoch + 1
                )
            if (
                np.argmin(model.info["test_metrics"]["loss/emd"])
                == len(model.info["test_metrics"]["loss/emd"]) - 1
            ):
                model.save(epoch + 1, pbar)

            tb_writer.flush()

    except KeyboardInterrupt:
        logger.info("Cancelled training (Keyboard Interrupt)")

    model.info["epochs"] = epoch + 1

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
            if conditioning.shape[1] < 2:
                conditioning = conditioning.reshape(-1, 1)
            # model_prediction = trained_model(
            #     torch.concat([x_t, conditioning], dim=-1), ts
            # )
            model_prediction = trained_model(
                torch.concat([x_t, conditioning, ts], dim=-1)
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
