# import torch
import torch.utils.data
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict
from icecream import ic
from data import data_gen, car_val, gen_car_state_area, read_yaml, calc_unicycle_states
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

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # Optional: Use Fourier feature embeddings for t, cf. transformers
        # t = torch.concat([t - 0.5, torch.cos(2*torch.pi*t), torch.sin(2*torch.pi*t), -torch.cos(4*torch.pi*t)], axis=1)
        x = torch.concat([x, t], dim=-1)
        # ic(x.shape)
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
            out = model(model_in, t.unsqueeze(1).to(DEVICE))
            loss = torch.mean(
                (noise[:, :10] - out) ** 2
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
    x_t = torch.randn((n_samples, 10)).to(DEVICE)
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
        model_prediction = trained_model(x_t, ts)
        x_t = (
            1
            / alphas[t] ** 0.5
            * (x_t - betas[t] / (1 - ab_t) ** 0.5 * model_prediction)
        )
        x_t += betas[t] ** 0.5 * z
        # ext = torch.randn((n_samples, 18)).to(DEVICE)
        # x_t = torch.concat((x_t, ext), dim=-1)

    return x_t


def main():
    ic(DEVICE)
    data = data_gen(N_SEQ)
    data_max = np.max(data[:, 3:], axis=0)
    data_min = np.min(data[:, 3:], axis=0)
    # plt.show()
    dataset = TensorDataset(torch.Tensor(data))
    loader = DataLoader(dataset, batch_size=50, shuffle=True)
    trained_model = train(loader, EPOCHS)

    samples = sample(trained_model, N_SAMPLES).detach().cpu().numpy()
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

    error_sel = val["error"] >= val["mse"]

    ic(val["mse"])
    ic(max_error, min_error)

    torch.save(trained_model.state_dict(), "model.pt")
    xy_states = val["states"][:, :2][error_sel]
    xy_pred = pred[:, :2][error_sel]
    # ic(xy_states, xy_pred)
    # plt.figure()
    # forward = actions[:, 0] > 0
    xy_states_poly = gen_car_state_area(60)
    # ic(xy_states_poly[:10, :])
    shape = shapely.Polygon(xy_states_poly)
    # ic(shape.area)
    plt.scatter(data[:, 0], data[:, 1], label="training")
    # plt.figure()
    g = gpd.GeoSeries(shape)
    g.plot(alpha=0.2, color="blue")
    # plt.plot(*zip(*xy_states_poly), label="test")
    # plt.figure()
    # plt.hist(val["error"])

    # plt.figure()

    plt.quiver(
        xy_states[:, 0],
        xy_states[:, 1],
        xy_pred[:, 0] - xy_states[:, 0],
        xy_pred[:, 1] - xy_states[:, 1],
        # xy_pred[:, 0],
        # xy_pred[:, 1],
        label="error",
        alpha=0.5,
        scale=1,
        scale_units="xy",
        color="grey",
    )
    plt.axis("equal")
    plt.scatter(
        xy_states[:, 0],
        xy_states[:, 1],
        label="states",
        alpha=0.7,
        marker="+",
        c="green",
    )
    plt.scatter(
        xy_pred[:, 0],
        xy_pred[:, 1],
        label="prediction",
        alpha=0.7,
        marker="x",
        c="orange",
    )
    plt.legend()
    # plt.figure()
    # plt.scatter(actions[:, 0], val["error"])
    # plt.figure()
    # plt.scatter(actions[:, 1], val["error"])

    plt.show()


def plotTraining(trainingData) -> None:
    plt.scatter(trainingData[:, 0], trainingData[:, 1], label="training")


def plotError(errorData) -> None:
    pass


def trainRun(args):
    # ic(args)
    args = vars(args)
    db = Database(filename="data.json")
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
        data_dict = {
            "type": "car",
            "n_hidden": args["nhidden"],
            "s_hidden": args["shidden"],
            "dim_action": data_0["action_dim"],
            # "dim_state": data_0["state_dim"],
            "dim_state": 0,
            "action_in": True,
            "action_out": True,
            "state_in": True,
            "state_out": False,
        }
        # data = np.concatenate([data_0["states"], data_0["actions"]], axis=-1)
        data = data_0["actions"]
        ic(data_0["actions"].shape)
        # ic(data)
        # ic(data.shape)

    # sys.exit()
    # uuid = db.getUUID(data_dict)
    # data_dict["uuid"] = uuid
    # data_max = np.max(data[:, 3:], axis=0)
    # data_min = np.min(data[:, 3:], axis=0)

    dataset = TensorDataset(torch.Tensor(data))
    loader = DataLoader(dataset, batch_size=50, shuffle=True)

    model = Net(data_dict)
    trained_model = train(model, loader, data_dict, args["epochs"])
    torch.save(trained_model.state_dict(), "unicycle.pt")
    samples = sample(trained_model, N_SAMPLES).detach().cpu().numpy()
    # ic(samples)
    # for s in samples:
    #     ic(s)
    #     for i in range(5):
    #         ic(s[i * 2 : i * 2 + 2])

    sample_state = calc_unicycle_states(samples[0])
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
    ax1.hist(data_0["actions"][:, 0], bins=bins_s, alpha=0.5, label="training data")
    ax1.hist(samples[:, 0], bins=bins_s, alpha=0.5, label="predicted")
    ax1.legend()
    ax1.set_title("distribution of s")
    ax2.hist(data_0["actions"][:, 1], bins=bins_phi, alpha=0.5, label="training data")
    ax2.hist(samples[:, 1], bins=bins_phi, alpha=0.5, label="predicted")
    ax2.legend()
    ax2.set_title("distribution of phi")
    plt.show()
    for i in range(5):
        state = calc_unicycle_states(samples[i])
        plt.plot(state[:, 1], state[:, 2])

    plt.show()
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
