from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
import torch
import torch.utils.data
from icecream import ic
import sys
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict
from data import (
    calc_unicycle_states,
    circle_SO2,
    data_gen,
    metric,
    read_yaml,
    spiral_points,
)
from diffusion_model import DEVICE, Net, sample, train


def plotTraining(trainingData) -> None:
    plt.scatter(trainingData[:, 0], trainingData[:, 1], label="training")


def plotSamples(samples, sample_data, n_samples=1000, n_plot: int = 5) -> None:
    start_arr = sample_data["start"]
    goal_arr = sample_data["goal"]
    pred_goals = []
    indices = np.linspace(0, n_samples - 1, n_plot, dtype=int)
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
    # TODO - get model output size and name from Net class
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
        data_current = data_list[i]
        sample_current = sample_list[i]

        bins = np.linspace(
            np.floor(np.min(data_current)), np.ceil(np.max(data_current)), n_bins
        )
        # h, _bins= np.histogram(sample_current, bins=bins, density=True)
        # width = bins[1]-bins[0]
        # ax[i].bar(_bins[:-1]+width/2, h, label="HELLO", width=width)
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


def loadModel(modelPath: Path, dataDict: Dict) -> Net:
    model = Net(dataDict)
    model.load_state_dict(torch.load(modelPath))
    return model


def outputToDict(modelOutput: np.ndarray, dataDict: Dict) -> Dict[str, np.ndarray]:
    returnDict = {}
    idx = 0
    sampleSize = modelOutput.shape[0]
    for outputType, length in dataDict["regular"].items():
        # breakpoint()
        if outputType == "actions":
            returnDict[outputType] = modelOutput[:, idx : idx + length].reshape(
                sampleSize, 5, 2
            )
        else:
            returnDict[outputType] = modelOutput[:,idx : idx + length]
        idx += length
    return returnDict


def plotError(errorData) -> None:
    pass


def trainRun(args: Dict):
    ic(DEVICE)
    training_size = args["trainingsize"]
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
    dataset = TensorDataset(torch.Tensor(data))
    dataset_split = torch.utils.data.random_split(
        dataset, [training_size, data.shape[0] - training_size]
    )[0]
    ic(len(dataset_split))
    loader = DataLoader(dataset_split, batch_size=50, shuffle=True)

    model = Net(data_dict)
    trained_model = train(model, loader, data_dict, args["epochs"])
    model_save = "alcove_unicycle.pt"
    torch.save(trained_model.state_dict(), model_save)
    print(f"Model saved as {model_save}")


def loadRun(args: Dict):
    dataDict = {
        "type": "car",
        "n_hidden": 4,
        "s_hidden": 256,
        "regular": {"actions": 10, "theta_0": 1},
        "conditioning": {},
    }
    model = loadModel(modelPath=Path(args["model"]), dataDict=dataDict)
    samples = sample(model, args["samples"]).detach().cpu().numpy()
    data = read_yaml(args["load"], **dataDict["regular"], **dataDict["conditioning"])
    plotHist(data, samples)


def export(args: Dict) -> None:
    ic(args)
    dataDict = {
        "type": "car",
        "n_hidden": 4,
        "s_hidden": 256,
        "regular": {"actions": 10, "theta_0": 1},
        "conditioning": {},
    }
    model = loadModel(modelPath=Path(args["model"]), dataDict=dataDict)

    samples = sample(model, args["samples"]).detach().cpu().numpy()
    sampleDict = outputToDict(samples, dataDict)
    sampleDict["actions"] = np.clip(sampleDict["actions"], -0.5, 0.5)
    # s_min = np.min(sampleDict["actions"][:,:,0])
    # s_max = np.max(sampleDict["actions"][:,:,0])
    # phi_min = np.min(sampleDict["actions"][:,:,1])
    # phi_max = np.max(sampleDict["actions"][:,:,1])

    # breakpoint()
    dt = 0.1
    outputList = []
    breakpoint()
    for actions, theta_0 in zip(sampleDict["actions"], sampleDict["theta_0"]):
        tempDict = {
            "actions": actions.tolist(),
            "time_stamp": 0,
            "feasible": 1,
            "traj_feas": 1,
            "goal_feas": 1,
            "start_feas": 1,
            "col_feas": 1,
            "x_bounds_feas": 1,
            "u_bounds_feas": 1,
            "max_jump": 0,
            "max_collision": 0,
            "start_distance": 0,
            "goal_distance": 0,
            "x_bound_distance": 0,
            "u_bound_distance": 0,
        }
        states = calc_unicycle_states(actions, dt=dt, start=[0, 0, float(theta_0)])
        numStates = len(states)
        numActions = len(actions)
        tempDict["cost"] = numActions * dt
        tempDict["start"] = states[0].tolist()
        tempDict["goal"] = states[-1].tolist()
        tempDict["num_states"] = numStates
        tempDict["states"] = states.tolist()
        tempDict["num_actions"] = numActions
        outputList.append(tempDict)

    # breakpoint()
    with open("output/unicycle_alcove.yaml", "w") as file:
        yaml.safe_dump(outputList, file, default_flow_style=None)


def listRun():
    raise NotImplementedError


#     db = Database("data.json")
#     data = db.tabulate(keys=["uuid", "type", "s_hidden"])
#     print(data)
