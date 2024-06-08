import torch
import torch.utils.data
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from icecream import ic
from data import data_gen, car_val
import matplotlib.pyplot as plt
import alphashape
import geopandas as gpd

N_SEQ = 1_000
EPOCHS = 2_00
N_SAMPLES = 1_000
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self, nhidden: int = 256):
        super().__init__()
        layers = [
            nn.Linear(6, nhidden)
        ]  # Change this to 6 if you want to use the fourier embeddings of t
        for _ in range(3):
            layers.append(nn.Linear(nhidden, nhidden))
        layers.append(nn.Linear(nhidden, 5))
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


def train(loader: DataLoader, nepochs: int = 10, denoising_steps: int = 100):
    """Alg 1 from the DDPM paper"""
    model = Net()
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    alpha_bars, _ = get_alpha_betas(denoising_steps)  # Precompute alphas
    losses = []
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
            noise = torch.randn(
                *data.shape, device=DEVICE
            )  # Sample DIFFERENT random noise for each datapoint
            model_in = (
                alpha_t**0.5 * data + noise * (1 - alpha_t) ** 0.5
            )  # Noise corrupt the data (eq14)
            out = model(model_in, t.unsqueeze(1).to(DEVICE))
            loss = torch.mean((noise - out) ** 2)  # Compute loss on prediction (eq14)
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
    x_t = torch.randn((n_samples, 5)).to(DEVICE)
    alpha_bars, betas = get_alpha_betas(n_steps)
    alphas = 1 - betas
    for t in range(len(alphas))[::-1]:
        ts = t * torch.ones((n_samples, 1)).to(DEVICE)
        ab_t = alpha_bars[t] * torch.ones((n_samples, 1)).to(
            DEVICE
        )  # Tile the alpha to the number of samples
        z = (torch.randn((n_samples, 5)) if t > 1 else torch.zeros((n_samples, 5))).to(
            DEVICE
        )
        model_prediction = trained_model(x_t, ts)
        x_t = (
            1
            / alphas[t] ** 0.5
            * (x_t - betas[t] / (1 - ab_t) ** 0.5 * model_prediction)
        )
        x_t += betas[t] ** 0.5 * z

    return x_t


def main():
    ic(DEVICE)
    data = data_gen(N_SEQ)
    # plt.show()
    dataset = TensorDataset(torch.Tensor(data))
    loader = DataLoader(dataset, batch_size=50, shuffle=True)
    trained_model = train(loader, EPOCHS)
    torch.save(trained_model.state_dict(), "model.pt")

    samples = sample(trained_model, N_SAMPLES).detach().cpu().numpy()
    pred = samples[:, :3]
    actions = samples[:, 3:]
    max_actions = np.max(actions, axis=0)
    min_actions = np.min(actions, axis=0)
    for n_action in range(len(max_actions)):
        action_min = min_actions[n_action]
        action_max = max_actions[n_action]
        ic(n_action, action_min, action_max)

    val = car_val(pred, actions)

    max_error = np.max(val["error"])
    min_error = np.min(val["error"])

    error_sel = val["error"] >= val["mse"]

    ic(val["mse"])
    ic(max_error, min_error)

    xy_states = val["states"][:, :2][error_sel]
    xy_pred = pred[:, :2][error_sel]
    # plt.figure()
    xy_states_poly = val["states"][:, :2]
    shape = alphashape.alphashape(xy_states_poly, 0.0)
    g = gpd.GeoSeries(shape)
    g.plot(alpha=0.2)
    plt.scatter(data[:, 0], data[:, 1], label="training")
    # plt.figure()
    # plt.hist(val["error"])

    # plt.figure()

    # plt.quiver(
    #     xy_states[:, 0],
    #     xy_states[:, 1],
    #     xy_pred[:, 0] - xy_states[:, 0],
    #     xy_pred[:, 1] - xy_states[:, 1],
    #     label="error",
    # )
    # plt.scatter(xy_states[:, 0], xy_states[:, 1], label="states")
    # plt.scatter(xy_pred[:, 0], xy_pred[:, 1], label="prediction", alpha=0.5)
    # plt.legend()
    # plt.figure()
    # plt.scatter(actions[:, 0], val["error"])
    # plt.figure()
    # plt.scatter(actions[:, 1], val["error"])

    plt.show()


if __name__ == "__main__":
    main()
