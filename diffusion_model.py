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
import geopandas as gpd

N_SEQ = 1_000
EPOCHS = 200
N_SAMPLES = 20
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.diff_in = config["diff_in"]
        input_size = (
            config["dim_state"] * config["state_in"]
            + config["dim_action"] * config["action_in"]
            + 1
        )
        output_size = (
            config["dim_state"] * config["state_out"]
            + config["dim_action"] * config["action_out"]
        )

        ic(input_size, output_size)
        layers = [
            nn.Linear(input_size, config["s_hidden"])
        ]  # Change this to 6 if you want to use the fourier embeddings of t
        for _ in range(config["n_hidden"]):
            layers.append(nn.Linear(config["s_hidden"], config["s_hidden"]))
        layers.append(nn.Linear(config["s_hidden"], output_size))
        self.linears = nn.ModuleList(layers)

        for layer in self.linears:
            # init using kaiming
            nn.init.kaiming_uniform_(layer.weight)

    def forward(
        self,
        x: torch.Tensor,
        start: torch.Tensor,
        goal: torch.Tensor,
        diff: torch.Tensor,
        t: torch.Tensor,
    ):
        if self.diff_in:
            x = torch.concat([x, diff, t], dim=-1)
        else:
            x = torch.concat([x, start, goal, t], dim=-1)

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
    time_start = time.time()
    for epoch in range(nepochs):
        for [data] in loader:
            data = data.to(DEVICE)
            start = data[:, 10:13]
            goal = data[:, 13:16]
            diff = data[:, 16:]
            data = data[:, :10]
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
            noise = torch.randn(
                *data.shape, device=DEVICE
            )  # Sample DIFFERENT random noise for each datapoint
            model_in = (
                alpha_t**0.5 * data + noise * (1 - alpha_t) ** 0.5
            )  # Noise corrupt the data (eq14)
            out = model(model_in, start, goal, diff, t.unsqueeze(1).to(DEVICE))
            loss = torch.mean(
                (noise[:, :10] - out) ** 2
            )  # Compute loss on prediction (eq14)
            losses.append(loss.detach().cpu().numpy())

            # Bwd pass
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 1000 == 0:
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


def sample(trained_model, sample_data, n_samples: int, n_steps: int = 100):
    """Alg 2 from the DDPM paper."""
    x_t = torch.randn((n_samples, 10)).to(DEVICE)

    # p = spiral_points()
    # points = [next(p) for _ in range(n_samples + 1)]
    # start = torch.Tensor(points[:n_samples])
    # goal = torch.Tensor(points[1:])

    # sample_data = circle_SO2(np.pi / 2, n_samples)
    # sample_data = spiral_points(n_samples)
    start = torch.Tensor(sample_data["start"]).to(DEVICE)
    goal = torch.Tensor(sample_data["goal"]).to(DEVICE)
    diff = torch.Tensor(sample_data["diff"]).to(DEVICE)
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
        z = (
            torch.randn((n_samples, 10)) if t > 1 else torch.zeros((n_samples, 10))
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


def plotHist(data_0, samples):
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
    ax1.hist(data_0["actions"][:, 0], bins=bins_s, alpha=0.5, label="training data")
    ax1.hist(samples[:, 0], bins=bins_s, alpha=0.5, label="predicted")
    ax1.legend()
    ax1.set_title("distribution of s")
    ax2.hist(data_0["actions"][:, 1], bins=bins_phi, alpha=0.5, label="training data")
    ax2.hist(samples[:, 1], bins=bins_phi, alpha=0.5, label="predicted")
    ax2.legend()
    ax2.set_title("distribution of phi")
    plt.show()


def plotError(errorData) -> None:
    pass


def trainRun(args):
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
        }
    else:
        data_0 = read_yaml(args["load"])
        ic(args["nhidden"])
        ic(args["shidden"])
        data_dict = {
            "type": "car",
            "n_hidden": args["nhidden"],
            "s_hidden": args["shidden"],
            "dim_action": data_0["action_dim"],
            # "dim_state": data_0["state_dim"],
            "diff_in": True,
            "dim_state": 3,
            "action_in": True,
            "action_out": True,
            "state_in": True,
            "state_out": False,
        }
        data = np.concatenate(
            [data_0["actions"], data_0["start"], data_0["goal"], data_0["diff"]],
            axis=-1,
        )
        # data = data_0["actions"]
        # ic(data_0["actions"].shape)
        # breakpoint()
        # ic(data)
        ic(data.shape)

    dataset = TensorDataset(torch.Tensor(data))
    loader = DataLoader(dataset, batch_size=50, shuffle=True)

    model = Net(data_dict)
    trained_model = train(model, loader, data_dict, args["epochs"])
    # sample_data = circle_SO2(np.pi / 2, N_SAMPLES)
    sample_data = spiral_points(N_SAMPLES)
    samples = sample(trained_model, sample_data, N_SAMPLES).detach().cpu().numpy()

    # sample_state = calc_unicycle_states(samples[0])
    # ic(sample_state)
    plotSamples(samples, sample_data, N_SAMPLES)

    sample_data = circle_SO2(np.pi * 2 / 3, N_SAMPLES)
    samples = sample(trained_model, sample_data, N_SAMPLES).detach().cpu().numpy()
    plotSamples(samples, sample_data, N_SAMPLES)
    print("Save ?")
    if input() == "y":
        print("input filename")
        name = input()
        torch.save(trained_model.state_dict(), name)
    sys.exit()
    pred = samples[:, :3]
    actions = samples[:, 3:]
    max_actions = np.max(actions, axis=0)
    min_actions = np.min(actions, axis=0)
    for n_action in range(len(max_actions)):
        pred_min = min_actions[n_action]
        pred_max = max_actions[n_action]
        action_min = data_min[n_action]
        action_max = data_max[n_action]

        ic(n_action, pred_min, action_min, pred_max, action_max)

    val = car_val(pred, actions)

    max_error = np.max(val["error"])
    min_error = np.min(val["error"])

    ic(val["mse"])
    ic(max_error, min_error)
    if uuid in db.check:
        if val["mse"] > db.data["data"][str(uuid)]["mse"]:
            print("not saving")
            sys.exit()
    print("saving")
    data_dict["mse"] = val["mse"]
    db.addEntry(data_dict, uuid)
    torch.save(trained_model.state_dict(), str(uuid))


def loadRun(args):
    pass


def listRun():
    db = Database("data.json")
    data = db.tabulate(keys=["uuid", "type", "s_hidden"])
    print(data)


if __name__ == "__main__":
    main()
