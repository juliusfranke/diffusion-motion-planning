import time
from typing import Dict, List
import tempfile
import numpy as np
from functools import partial
import torch.nn as nn
import torch.utils.data
from icecream import ic
from torch.utils.data import DataLoader, random_split
from ray import train
from ray.train import Checkpoint, get_checkpoint
import ray.cloudpickle as pickle
from tqdm import tqdm
from pathlib import Path
import logging
from data import (
    load_data,
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
    post_SVD,
    passthrough,
)
from torch.utils.tensorboard import SummaryWriter

torch.autograd.set_detect_anomaly(True)
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
        if "R4SVD" in self.regular.keys() or "R2SVD" in self.regular.keys():
            self.postprocessor = post_SVD
        else:
            self.postprocessor = passthrough

        logger.debug("Created Net")

    def postprocess(self, x):
        return self.postprocessor(x, self.regular)

    def save(self, epoch, pbar: tqdm):
        if isinstance(self.path, Path):
            torch.save(self.state_dict(), self.path)
            pbar.set_description(f"Saved @ epoch {epoch}")
        else:
            logger.error("Model path not set")
            raise Exception

    def forward(self, x):

        for layer in self.linears[:-1]:
            x = nn.ReLU()(layer(x))
        out = self.linears[-1](x)
        return out


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
        # model_in = model.postprocess(model_in)
        x = torch.concat([model_in, t.unsqueeze(1).to(DEVICE)], dim=-1)
        out = model(x)

        loss = model.loss(
            out,
            noise[:, :output_size],
        )
        # loss = model.loss(
        #     model.postprocess(out),
        #     model.postprocess(noise[:, :output_size]),
        # )
        # breakpoint()

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


def train_raytune(
    config: Dict,
    model_static: Dict,
    nepochs: int = 10,
):
    """Alg 1 from the DDPM paper"""
    data_dict = config | model_static
    denoising_steps = data_dict["denoising_steps"]
    lr = data_dict["lr"]
    model = Net(data_dict)
    model.to(DEVICE)
    model.info["lr"] = lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    alpha_bars, _ = get_alpha_betas(denoising_steps)  # Precompute alphas

    model.info["train_metrics"] = {"loss/train": []}
    model.info["val_metrics"] = {
        "loss/val": [],
    }
    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0
    pbar = tqdm(range(start_epoch, nepochs))

    trainset, testset = load_data(data_dict)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )

    training_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(data_dict["batch_size"]),
        shuffle=True,
        num_workers=8,
    )
    validation_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=int(data_dict["batch_size"]), shuffle=True, num_workers=8
    )

    for epoch in pbar:
        model.train()
        run_epoch(
            training_loader,
            model,
            optimizer,
            alpha_bars,
            denoising_steps,
            validate=False,
        )

        model.eval()
        run_epoch(
            validation_loader,
            model,
            optimizer,
            alpha_bars,
            denoising_steps,
            validate=True,
        )
        # idx_best = np.argmin(model.info["val_metrics"]["loss/val"])
        # val_step = len(model.info["val_metrics"]["loss/val"]) - 1

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {"loss": model.info["val_metrics"]["loss/val"][-1]},
                checkpoint=checkpoint,
            )

    pbar.close()

    model.info["epochs"] = epoch + 1

    # return model


def train_normal(
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
            idx_best = np.argmin(model.info["val_metrics"]["loss/val"])
            val_step = len(model.info["val_metrics"]["loss/val"]) - 1
            if idx_best == val_step:
                model.save(epoch + 1, pbar)
            elif val_step - idx_best >= 100:
                logger.info("Cancelled training after 1000 epochs with no improvements")
                break

    except KeyboardInterrupt:
        logger.info("Cancelled training (Keyboard Interrupt)")
    pbar.close()

    model.info["epochs"] = epoch + 1

    return model


def sample(
    trained_model: Net,
    n_samples: int,
    n_steps: int = 100,
    conditioning: torch.Tensor | None = None,
):
    """Alg 2 from the DDPM paper."""
    x_t = torch.randn((n_samples, trained_model.output_size), device=DEVICE)
    # max = torch.tensor([0.5] * 10 + [torch.pi], device=DEVICE)
    # x_t = torch.clamp(x_t, min=-max, max=max)

    # x_t = torch.rand((n_samples, trained_model.output_size)).to(DEVICE) - 0.5
    # weight = torch.ones(trained_model.output_size, device=DEVICE)
    # weight[-1] = 2 * torch.pi
    # x_t = x_t * weight

    alpha_bars, betas = get_alpha_betas(n_steps)
    alphas = np.clip(1 - betas, 1e-8, np.inf)
    for t in range(len(alphas))[::-1]:
        ts = t * torch.ones((n_samples, 1), device=DEVICE)
        ab_t = alpha_bars[t] * torch.ones((n_samples, 1), device=DEVICE)
        # Tile the alpha to the number of samples
        z = (
            torch.randn((n_samples, trained_model.output_size), device=DEVICE)
            if t > 1
            else torch.zeros((n_samples, trained_model.output_size), device=DEVICE)
        )
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
            # breakpoint()
            x_t += betas[t] ** 0.5 * z
        else:
            model_prediction = trained_model(x_t, ts)
            x_t = (
                1
                / alphas[t] ** 0.5
                * (x_t - betas[t] / (1 - ab_t) ** 0.5 * model_prediction)
            )
            x_t += betas[t] ** 0.5 * z

    return trained_model.postprocess(x_t)


if __name__ == "__main__":
    pass
