# import torch
import torch.utils.data
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict
from icecream import ic
from data import (
    data_gen,
    car_val,
    gen_car_state_area,
    metric,
    read_yaml,
    calc_unicycle_states,
    spiral_points,
    circle_SO2,
)
import time
import matplotlib.pyplot as plt

import sys
from db import Database

# import alphashape

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

    def forward(self, x, t):
        # x = torch.concat([tensor for tensor in args], dim=-1)
        x = torch.concat([x, t], dim=-1)

        for layer in self.linears[:-1]:
            x = nn.ReLU()(layer(x))
        return self.linears[-1](x)


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
    time_start = time.time()

    output_size = model.output_size
    for epoch in range(nepochs):
        # ic(epoch)
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
            # ic(data.shape)
            # ic(config["actions"].shape)

            # breakpoint()
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
            losses.append(loss.detach().cpu().numpy())

            # Bwd pass
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            time_passed = time.time() - time_start
            time_per_epoch = time_passed / (epoch + 1)
            mean_loss = np.mean(np.array(losses))
            losses = []
            # print("Epoch %d,\t Loss %f " % (epoch + 1, mean_loss))
            if time_per_epoch < 1:
                epoch_per_second = 1 / time_per_epoch
                ic(epoch + 1, mean_loss, epoch_per_second)
            else:
                ic(epoch + 1, mean_loss, time_per_epoch)

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
            model_prediction = trained_model(x_t, conditioning, ts)
            x_t = (
                1
                / alphas[t] ** 0.5
                * (
                    x_t[:, : trained_model.output_size]
                    - betas[t] / (1 - ab_t) ** 0.5 * model_prediction
                )
            )
            x_t += betas[t] ** 0.5 * z
            x_t = torch.concat([x_t, conditioning], dim=-1)
        else:
            model_prediction = trained_model(x_t, ts)
            x_t = (
                1
                / alphas[t] ** 0.5
                * (x_t - betas[t] / (1 - ab_t) ** 0.5 * model_prediction)
            )
            x_t += betas[t] ** 0.5 * z

    return x_t


def plotTraining(trainingData) -> None:
    plt.scatter(trainingData[:, 0], trainingData[:, 1], label="training")


def plotSamples(samples, sample_data, n_plot: int = 5) -> None:
    start_arr = sample_data["start"]
    goal_arr = sample_data["goal"]
    pred_goals = []
    indices = np.linspace(0, N_SAMPLES - 1, n_plot, dtype=int)
    for index in range(n_plot):
        i = indices[index]
        start = start_arr[i]
        # ic(start)
        # ic(samples[i])
        state = calc_unicycle_states(samples[i], start=start)
        pred_goals.append(state[-1])
        # ic(state)
        plt.scatter(goal_arr[i, 0], goal_arr[i, 1], label=f"goal{index}")
        plt.plot(state[:, 0], state[:, 1], label=f"primitive {index}")
    pred_goals = np.array(pred_goals)
    # breakpoint()
    mse = np.mean(metric(goal_arr, pred_goals))
    ic(mse)
    max = 1.1 * np.max(np.abs(np.concatenate([start_arr[:, :2], goal_arr[:, :2]])))

    # breakpoint()
    plt.xlim(-max, max)
    plt.ylim(-max, max)
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.show()


def plotHist(data: np.ndarray, samples: np.ndarray):
    data_actions = data[:, :10].reshape(data.shape[0] * 5, 2)
    sample_actions = samples[:, :10].reshape(samples.shape[0] * 5, 2)

    data_s = data_actions[:, 0]
    data_phi = data_actions[:, 1]
    data_theta_0 = data[:, 10]

    sample_s = sample_actions[:, 0]
    sample_phi = sample_actions[:, 1]
    sample_theta_0 = samples[:, 10]
    sample_list = [sample_s, sample_phi, sample_theta_0]
    data_list = [data_s, data_phi, data_theta_0]

    n_bins = 50
    titles = ["s", "phi", "theta_0"]
    fig, ax = plt.subplots(3)
    for i in range(3):
        breakpoint()
        data_current = data_list[i]
        sample_current = sample_list[i]

        bins = np.linspace(
            np.floor(np.min(data_current)), np.ceil(np.max(data_current)), n_bins
        )
        ax[i].hist(
            data_current,
            bins=bins,
            alpha=0.5,
            label="training data",
            density=True,
        )
        ax[i].hist(
            sample_current, bins=bins, alpha=0.5, label="predicted", density=True
        )
        ax[i].legend()
        ax[i].set_title(f"distribution of {titles[i]}")
    plt.show()


def plotError(errorData) -> None:
    pass


def trainRun(args):
    ic(DEVICE)
    args = vars(args)
    if args["generate"]:
        data = data_gen(args["trainingsize"])
        data_dict = {
            "type": "car",
            "n_hidden": args["nhidden"],
            "s_hidden": args["shidden"],
            "dim_action": 2,
            "dim_state": 3,
            "action_in": True,
            "action_out": True,
            "state_in": True,
            "state_out": True,
            # "theta_0_in":True,
            # "theta_0_out":True,
        }
    else:
        data_dict = {
            "type": "car",
            "n_hidden": args["nhidden"],
            "s_hidden": args["shidden"],
            "regular": {"actions": 10, "theta_0": 1},
            "conditioning": {},
        }
        data = read_yaml(
            args["load"], **data_dict["regular"], **data_dict["conditioning"]
        )

        ic(data.shape)

    # breakpoint()
    dataset = TensorDataset(torch.Tensor(data))
    loader = DataLoader(dataset, batch_size=50, shuffle=True)

    model = Net(data_dict)
    trained_model = train(model, loader, data_dict, args["epochs"])
    # torch.save(trained_model.state_dict(), "unicycle_larger.pt")
    samples = sample(trained_model, N_SAMPLES).detach().cpu().numpy()
    # ic(samples)
    # for s in samples:
    #     ic(s)
    #     for i in range(5):
    #         ic(s[i * 2 : i * 2 + 2])

    # sample_state = calc_unicycle_states(samples[0, :10])
    # ic(sample_state)
    # start = [0.0, 0.0, np.pi]
    plotHist(data, samples)
    circle = circle_SO2(np.pi / 2, N_SAMPLES)
    start_arr = circle["start"]
    goal_arr = circle["goal"]
    p = spiral_points()
    n_plot = N_SAMPLES
    points = [next(p)]
    ic(points)
    indices = np.linspace(0, N_SAMPLES - 1, n_plot, dtype=int)
    for index in range(n_plot):
        i = indices[index]
        # start = start_arr[i]
        start = [0, 0, samples[i, -1]]
        # goal = goal_arr[i]
        ic(start)
        ic(samples[i])
        state = calc_unicycle_states(samples[i, :10], start=start)
        ic(state)
        # start = next(p)
        # points.append(start)
        # start = state[-1]
        # plt.scatter(goal_arr[i, 0], goal_arr[i, 1], label=f"goal{index}")
        plt.plot(state[:, 0], state[:, 1], label=f"primitive {index}")

    # plt.scatter(goal_arr[:n_plot, 0], goal_arr[:n_plot, 1], label="goals")
    # points = np.array(points)
    # plt.plot(points[:, 0], points[:, 1], label="spiral")
    plt.legend()
    plt.axis("equal")
    plt.show()
    sys.exit()


def loadRun(args):
    pass


def listRun():
    db = Database("data.json")
    data = db.tabulate(keys=["uuid", "type", "s_hidden"])
    print(data)


if __name__ == "__main__":
    pass
