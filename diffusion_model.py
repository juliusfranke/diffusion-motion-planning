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
    read_yaml,
    calc_unicycle_states,
    spiral_points,
    circle_SO2,
)
import matplotlib.pyplot as plt
import shapely

import sys
from db import Database

# import alphashape
import geopandas as gpd

N_SEQ = 1_000
EPOCHS = 200
N_SAMPLES = 1_000
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        # self.diff_in = config["diff_in"]
        # self.state_in = config["state_in"]
        # self.theta_0_in = config["theta_0_in"]
        # self.theta_0_out = config["theta_0_out"]
        self.input = config["input"]
        self.conditioning = config["conditioning"]
        self.validate: Dict[str, int] = self.input | self.conditioning

        self.output_size = sum(self.input.values())

        self.input_size = self.output_size + sum(self.conditioning.values()) + 1

        ic(self.input_size, self.output_size)
        layers = [
            nn.Linear(self.input_size, config["s_hidden"])
        ]  # Change this to 6 if you want to use the fourier embeddings of t
        for _ in range(config["n_hidden"]):
            layers.append(nn.Linear(config["s_hidden"], config["s_hidden"]))
        layers.append(nn.Linear(config["s_hidden"], self.output_size))
        self.linears = nn.ModuleList(layers)

        for layer in self.linears:
            # init using kaiming
            nn.init.kaiming_uniform_(layer.weight)

    def validate_input(self, **kwargs: torch.Tensor) -> bool:
        return kwargs.keys() == self.validate.keys()

    def forward(
        self,
        **kwargs: torch.Tensor,
        # x: torch.Tensor,
        # start: torch.Tensor,
        # goal: torch.Tensor,
        # diff: torch.Tensor,
        # t: torch.Tensor,
    ):
        x = torch.concat([tensor for tensor in kwargs.values()], dim=-1)
        # if self.diff_in:
        #     x = torch.concat([x, diff, t], dim=-1)
        # elif self.state_in:
        #     x = torch.concat([x, start, goal, t], dim=-1)
        # else:
        #     x = torch.concat([x, t], dim=-1)

        # Optional: Use Fourier feature embeddings for t, cf. transformers
        # t = torch.concat([t - 0.5, torch.cos(2*torch.pi*t), torch.sin(2*torch.pi*t), -torch.cos(4*torch.pi*t)], axis=1)
        # ic(x.shape, start.shape, goal.shape, t.shape)

        # diff_xy = goal[:, :2] - start[:, :2]
        # # pos_diff =
        # a = (goal[:, 2] - start[:, 2]) % (2 * np.pi)
        # b = (start[:, 2] - goal[:, 2]) % (2 * np.pi)
        # diff_theta = -a * (a < b) + b * (b <= a)
        # diff_theta = diff_theta.reshape(-1, 1)
        # breakpoint()
        # diff = torch.concat([diff_xy, diff_theta], dim=-1)
        # diff_theta = goal[:, :2] - start[:, :2]
        # x = torch.concat([x, diff, t], dim=-1)
        # ic(x.shape)
        # breakpoint()
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
    output_size = (
        config["dim_state"] * config["state_out"]
        + config["dim_action"] * config["action_out"]
    )
    for epoch in range(nepochs):
        for [data] in loader:
            data = data.to(DEVICE)
            start = data[:, 10:13]
            goal = data[:, 13:16]
            diff = data[:, 16:]
            data = data[:, :10]
            data = torch.concat([data, start[:, 2].reshape(-1, 1)], dim=-1)
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
            out = model(model_in, start, goal, diff, t.unsqueeze(1).to(DEVICE))
            loss = torch.mean(
                (noise[:, :11] - out) ** 2
            )  # Compute loss on prediction (eq14)
            losses.append(loss.detach().cpu().numpy())

            # Bwd pass
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 200 == 0:
            mean_loss = np.mean(np.array(losses))
            losses = []
            # print("Epoch %d,\t Loss %f " % (epoch + 1, mean_loss))
            ic(epoch + 1, mean_loss)

    return model


def sample(trained_model, n_samples: int, n_steps: int = 100):
    """Alg 2 from the DDPM paper."""
    x_t = torch.randn((n_samples, 11)).to(DEVICE)

    # p = spiral_points()
    # points = [next(p) for _ in range(n_samples + 1)]
    # start = torch.Tensor(points[:n_samples])
    # goal = torch.Tensor(points[1:])

    circle = circle_SO2(np.pi / 2, n_samples)
    start = torch.Tensor(circle["start"]).to(DEVICE)
    goal = torch.Tensor(circle["goal"]).to(DEVICE)
    diff = torch.Tensor(circle["diff"]).to(DEVICE)
    # start = torch.randn((n_samples, 3)).to(DEVICE)
    # ic(start, goal, diff)
    # cond = np.array([next(p) for _ in range(20)])
    # xy = -0.25 + 0.5 * torch.rand(n_samples, 2).to(DEVICE)
    # th = -np.pi + 2 * np.pi * torch.rand(n_samples, 2).to(DEVICE)
    # cond = torch.concat((xy, th), dim=-1)

    alpha_bars, betas = get_alpha_betas(n_steps)
    alphas = 1 - betas
    for t in range(len(alphas))[::-1]:
        ts = t * torch.ones((n_samples, 1)).to(DEVICE)
        ab_t = alpha_bars[t] * torch.ones((n_samples, 1)).to(
            DEVICE
        )  # Tile the alpha to the number of samples
        z_dim = 11
        z = (
            torch.randn((n_samples, z_dim))
            if t > 1
            else torch.zeros((n_samples, z_dim))
        ).to(DEVICE)
        model_prediction = trained_model(x_t, start, goal, diff, ts)
        x_t = (
            1
            / alphas[t] ** 0.5
            * (x_t - betas[t] / (1 - ab_t) ** 0.5 * model_prediction)
        )
        x_t += betas[t] ** 0.5 * z
        # x_t = torch.concat((x_t,), dim=-1)

    return x_t


def plotTraining(trainingData) -> None:
    plt.scatter(trainingData[:, 0], trainingData[:, 1], label="training")


def plotError(errorData) -> None:
    pass


def trainRun(args):
    ic(DEVICE)
    # ic(args)
    args = vars(args)
    # db = Database(filename="data.json")
    data_0 = {}
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
        data_0 = read_yaml(args["load"], actions=10,states=18, theta_0=1)
        data_dict = {
            "type": "car",
            "n_hidden": args["nhidden"],
            "s_hidden": args["shidden"],
            # "dim_action": data_0["action_dim"],
            # "dim_state": data_0["state_dim"],
            "diff_in": False,
            "dim_action": 10,
            "dim_state": 3,
            "action_in": True,
            "action_out": True,
            "state_in": False,
            "state_out": False,
            "theta_0_in": True,
            "theta_0_out": True,
            "input": {"actions": 10, "theta_0": 1},
            "conditioning": {},
        }
        data = np.concatenate(
            [data_0["actions"], data_0["start"], data_0["goal"], data_0["diff"]],
            axis=-1,
        )

        ic(data.shape)
    # for i in range(len(data_0["start"])):
    #     st = data_0["start"][i]
    #     stx = [st[0], st[0] + 0.1 * np.cos(st[2])]
    #     sty = [st[1], st[1] + 0.1 * np.sin(st[2])]
    #     gl = data_0["goal"][i]
    #     df = data_0["diff"][i]
    #     acts = data_0["actions"][i]
    #     ic(st, gl, df, acts)
    #     plt.scatter(0, 0, marker=".", color="red")
    #     plt.plot(stx, sty, label="start")
    #     plt.scatter(gl[0], gl[1], label="goal")
    #     plt.scatter(df[0], df[1], label="diff")
    #     plt.legend()
    #     plt.show()
    # uuid = db.getUUID(data_dict)
    # data_dict["uuid"] = uuid
    # data_max = np.max(data[:, 3:], axis=0)
    # data_min = np.min(data[:, 3:], axis=0)

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

    sample_state = calc_unicycle_states(samples[0, :10])
    ic(sample_state)
    a_max = np.maximum(
        np.max(samples[:, :2], axis=0), np.max(data_0["actions"][:, :2], axis=0)
    )
    a_min = np.minimum(
        np.min(samples[:, :2], axis=0), np.min(data_0["actions"][:, :2], axis=0)
    )
    n_bins = 100
    bins_s = np.linspace(np.floor(a_min[0]), np.ceil(a_max[0]), n_bins)
    bins_phi = np.linspace(np.floor(a_min[1]), np.ceil(a_max[1]), n_bins)
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.hist(
        data_0["actions"][:, 0],
        bins=bins_s,
        alpha=0.5,
        label="training data",
        density=True,
    )
    ax1.hist(samples[:, 0], bins=bins_s, alpha=0.5, label="predicted", density=True)
    ax1.legend()
    ax1.set_title("distribution of s")
    ax2.hist(
        data_0["actions"][:, 1],
        bins=bins_phi,
        alpha=0.5,
        label="training data",
        density=True,
    )
    ax2.hist(samples[:, 1], bins=bins_phi, alpha=0.5, label="predicted", density=True)
    ax2.legend()
    ax2.set_title("distribution of phi")
    plt.show()
    # start = [0.0, 0.0, np.pi]
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
