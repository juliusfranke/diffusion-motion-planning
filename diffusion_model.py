import time
from typing import Dict, List

import numpy as np
from functools import partial
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
    wasserstein_distance_pytorch,
    mse,
    mae,
    log_cosh_loss,
    clipped_mse_loss,
    boundary_aware_loss,
    weighted_mse_loss,
    mmd_rbf,
    sinkhorn,
    mse_theta,
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
        # self.loss_fn = {"actions": self._loss_actions, "theta_0": self._loss_theta_0}
        max = torch.tensor([0.5] * 10 + [torch.pi, torch.inf], device=DEVICE)
        loss_fns = {
            "mse": mse,
            "mae": mae,
            "log_cosh": log_cosh_loss,
            "clipped_mse": partial(clipped_mse_loss, max_val=max, min_val=-max),
            "bound_aware": partial(boundary_aware_loss, max_val=max, min_val=-max),
            "w_mse": partial(weighted_mse_loss, max_val=max[:-1], min_val=-max[:-1]),
            "mmd": mmd_rbf,
            "sinkhorn": sinkhorn,
            "mse_theta": mse_theta,
        }
        self.loss = loss_fns[config["loss"]]
        logger.debug("Created Net")

    def save(self, epoch, pbar: tqdm):
        if isinstance(self.path, Path):
            torch.save(self.state_dict(), self.path)
            pbar.set_description(f"Saved @ epoch {epoch}")
        else:
            logger.error("Model path not set")
            raise Exception

    # def loss(self, output: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    #     error = noise[:, : self.output_size] - output
    #     return torch.mean(error**2)
    # weights = torch.ones(self.output_size, device=DEVICE)
    # weights[-1] = 1 / (2 * torch.pi)
    # error = noise[:, : self.output_size] - output * weights
    # return torch.mean(error**2)
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
        # out = self.linears[-1](x)
        # out[:, -1] = (out[:, -1] + torch.pi) % (2 * torch.pi) - torch.pi
        # return out
        out = self.linears[-1](x)
        # max = torch.tensor([0.5] * 10 + [torch.pi], device=DEVICE)
        # out = torch.clamp(out, min=-max, max=max)
        return out

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
    # betas = np.random.uniform(10e-4, 0.02, N)  # schedule from the 2020 paper
    alpha_bars = np.cumprod(1 - betas)
    return alpha_bars, betas


def run_epoch(
    data_loader: DataLoader,
    model: Net,
    optimizer,
    alpha_bars,
    denoising_steps: int,
    validate=False,
):

    output_size = model.output_size
    running_loss = 0.0
    for i, [data] in enumerate(data_loader):
        data = data.to(DEVICE)
        if not validate:
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
        # max = torch.tensor([0.5] * 10 + [torch.pi, torch.inf], device=DEVICE)
        # noise = torch.clamp(noise, min=-max, max=max)
        # noise = (
        #     torch.rand(*data.shape, device=DEVICE) - 0.5
        # )  # Sample DIFFERENT random noise for each datapoint
        # weight = torch.ones(noise.shape[1], device=DEVICE)
        # weight[-1] = 2 * torch.pi
        # noise = noise * weight
        model_in = (
            alpha_t**0.5 * data + noise * (1 - alpha_t) ** 0.5
        )  # Noise corrupt the data (eq14)

        x = torch.concat([model_in, t.unsqueeze(1).to(DEVICE)], dim=-1)
        out = model(x)
        # loss = torch.mean(
        #     (noise[:, :output_size] - out) ** 2
        # )  # Compute loss on prediction (eq14)
        loss = model.loss(out, noise[:, :output_size])

        # losses.append(loss.detach().cpu().numpy())
        running_loss += loss.detach().cpu().numpy()
        # Bwd pass
        if not validate:
            loss.backward()
            optimizer.step()

    if validate:
        model.info["val_metrics"]["loss/val"].append(running_loss / (i + 1))
    else:
        model.info["train_metrics"]["loss/train"].append(running_loss / (i + 1))


def val_epoch(
    validation_loader: DataLoader,
    model: Net,
    denoising_steps: int,
):

    running_vloss = 0.0
    running_violations = 0.0
    running_violation_score = 0.0

    hist_data = {
        "s_real": [],
        "s_pred": [],
        "phi_real": [],
        "phi_pred": [],
        "theta_0_real": [],
        "theta_0_pred": [],
    }
    with torch.no_grad():
        for i, [vdata] in enumerate(validation_loader):
            regular = vdata[:, : model.output_size]
            conditioning = vdata[:, model.output_size :]
            pred = sample(
                model,
                len(vdata),
                n_steps=denoising_steps,
                conditioning=conditioning,
            )
            hist_data["s_real"].extend(regular[:, ::2][:, :5].flatten().detach().cpu())
            hist_data["s_pred"].extend(pred[:, ::2][:, :5].flatten().detach().cpu())
            hist_data["phi_real"].extend(
                regular[:, 1::2][:, :5].flatten().detach().cpu()
            )
            hist_data["phi_pred"].extend(pred[:, 1::2][:, :5].flatten().detach().cpu())
            if pred.shape[1] == 11:
                hist_data["theta_0_real"].extend(
                    regular[:, 10].flatten().detach().cpu()
                )
                hist_data["theta_0_pred"].extend(pred[:, 10].flatten().detach().cpu())
            else:
                hist_data["theta_0_real"].append(0)
                hist_data["theta_0_pred"].append(0)
            # phi = pred[:, 1::2][:, :5]
            # theta_0 = pred[:, 10]
            val_loss = model.loss(pred, regular).detach().cpu().numpy()
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

    return hist_data


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
    tb_writer: SummaryWriter,
    nepochs: int = 10,
    denoising_steps: int = 50,
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
        # "viol/percent": [],
        # "viol/avg": [],
    }
    model.info["test_metrics"] = {"loss/emd": []}
    try:
        for epoch in pbar:
            tb_writer.flush()
            model.train()
            run_epoch(
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
            run_epoch(
                validation_loader,
                model,
                optimizer,
                alpha_bars,
                denoising_steps,
                validate=True,
            )
            for val_metric in model.info["val_metrics"].keys():
                tb_writer.add_scalar(
                    val_metric,
                    model.info["val_metrics"][val_metric][-1],
                    epoch + 1,
                )
            if (
                np.argmin(model.info["val_metrics"]["loss/val"])
                == len(model.info["val_metrics"]["loss/val"]) - 1
            ):
                model.save(epoch + 1, pbar)
            # hist_data = val_epoch(validation_loader, model, denoising_steps)
            # for val_metric in model.info["val_metrics"].keys():
            #     tb_writer.add_scalar(
            #         val_metric, model.info["val_metrics"][val_metric][-1], epoch + 1
            #     )
            # if (epoch + 1) % 20 != 0:
            #     continue
            # test_epoch(test_loader, model)
            # for test_metric in model.info["test_metrics"].keys():
            #     tb_writer.add_scalar(
            #         test_metric, model.info["test_metrics"][test_metric][-1], epoch + 1
            #     )
            # if (
            #     np.argmin(model.info["val_metrics"]["loss/val"])
            #     == len(model.info["val_metrics"]["loss/val"]) - 1
            # ):
            #     for key, hist in hist_data.items():
            #         if "theta" in key:
            #             clip = 5
            #         else:
            #             clip = 1
            #         tb_writer.add_histogram(
            #             key,
            #             np.clip(np.array(hist), a_min=-clip, a_max=clip),
            #             epoch + 1,
            #             bins=50,
            #         )
            #     model.save(epoch + 1, pbar)

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
    # max = torch.tensor([0.5] * 10 + [torch.pi], device=DEVICE)
    # x_t = torch.clamp(x_t, min=-max, max=max)

    # x_t = torch.rand((n_samples, trained_model.output_size)).to(DEVICE) - 0.5
    # weight = torch.ones(trained_model.output_size, device=DEVICE)
    # weight[-1] = 2 * torch.pi
    # x_t = x_t * weight

    alpha_bars, betas = get_alpha_betas(n_steps)
    alphas = np.clip(1 - betas, 1e-8, np.inf)
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
        # z = torch.clamp(z, min=-max, max=max)
        # z = (
        #     torch.rand((n_samples, trained_model.output_size)) - 0.5
        #     if t > 1
        #     else torch.zeros((n_samples, trained_model.output_size))
        # ).to(DEVICE)

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
