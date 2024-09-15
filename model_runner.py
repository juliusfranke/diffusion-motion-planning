from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.utils.data
import yaml
from data import (
    WeightSampler,
    calc_unicycle_states,
    data_gen,
    metric,
    pruneDataset,
    read_yaml,
)
from diffusion_model import DEVICE, Net, sample, train
from icecream import ic
from torch.utils.data import DataLoader, TensorDataset


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


def plotHist(data: np.ndarray, samples: np.ndarray, dataDict: Dict):
    action_length = dataDict["regular"]["actions"]
    mp_length = action_length // 2

    data_actions_mean = np.mean(
        data[:, :action_length].reshape(data.shape[0], mp_length, 2), axis=1
    )
    sample_actions_mean = np.clip(
        np.mean(
            samples[:, :action_length].reshape(samples.shape[0], mp_length, 2), axis=1
        ),
        -0.5,
        0.5,
    )
    actions_mean = np.concatenate([data_actions_mean, sample_actions_mean])

    data_theta_0 = data[:, action_length]
    sample_theta_0 = samples[:, action_length]

    # ws = WeightSampler()
    # data_weights = ws.ppf(data[:, action_length+1])
    data_weights = data[:, action_length + 1]
    pltdict = {
        "s": actions_mean[:, 0].flatten(),
        "phi": actions_mean[:, 1].flatten(),
        "theta_0": np.concatenate([data_theta_0, sample_theta_0]).flatten(),
        "weights": data_weights.tolist() + samples.shape[0] * [1],
        "source": data.shape[0] * ["training data"] + samples.shape[0] * ["predicted"],
    }
    pltdf = pd.DataFrame(pltdict)

    sns.set_theme()

    g = sns.PairGrid(pltdf, hue="source", corner=False, vars=["s", "phi", "theta_0"])

    g.map_diag(
        sns.histplot,
        element="poly",
        common_norm=False,
        stat="percent",
        weights=pltdf["weights"],
        bins=50,
    )
    g.map_lower(sns.kdeplot, levels=4, common_norm=False, weights=pltdf["weights"])
    g.map_upper(sns.scatterplot, s=20, alpha=1, marker="x")
    g.add_legend()

    plt.show()


def loadModel(modelName: str) -> Tuple[Net, Dict]:
    modelPath = Path(__file__).parents[0] / "data" / "models" / modelName
    weightsPath = modelPath.with_suffix(".pt")
    configPath = modelPath.with_suffix(".yaml")

    if not weightsPath.exists() and not configPath.exists():
        raise FileNotFoundError

    with open(configPath, "r") as file:
        dataDict = yaml.safe_load(file)
    model = Net(dataDict)
    model.load_state_dict(torch.load(weightsPath))
    return model, dataDict


def outputToDict(modelOutput: np.ndarray, dataDict: Dict) -> Dict[str, np.ndarray]:
    returnDict = {}
    idx = 0
    sampleSize = modelOutput.shape[0]
    for outputType, length in dataDict["regular"].items():
        if outputType == "actions":
            returnDict[outputType] = modelOutput[:, idx : idx + length].reshape(
                sampleSize, length // 2, 2
            )
        else:
            returnDict[outputType] = modelOutput[:, idx : idx + length]
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
            "conditioning": {
                "delta_0": 1,
                "rel_probability": 1,
                "area": 1,
                "area_blocked": 1,
            },
        }
        data = read_yaml(
            args["load"], **data_dict["regular"], **data_dict["conditioning"]
        )

    ic(data.shape)
    # data = prune(data, 0.1)
    dataset = TensorDataset(torch.Tensor(data))
    if training_size < data.shape[0] and training_size != -1:
        dataset_split = torch.utils.data.random_split(
            dataset, [training_size, data.shape[0] - training_size]
        )[0]
    else:
        dataset_split = dataset
    ic(len(dataset_split))
    loader = DataLoader(dataset_split, batch_size=50, shuffle=True)

    model = Net(data_dict)
    trained_model = train(model, loader, data_dict, args["epochs"])
    model_save = "data/models/rand_env_l5.pt"
    torch.save(trained_model.state_dict(), model_save)
    print(f"Model saved as {model_save}")


def loadRun(args: Dict):
    model, dataDict = loadModel(modelName=args["model"])
    ws = WeightSampler()
    cdf = torch.Tensor(ws.rvs(size=args["samples"])).to(DEVICE)
    # cdf = torch.Tensor(ws.ppf(np.linspace(0,1,args["samples"]))).to(DEVICE)
    # cdf = torch.Tensor(np.linspace(0,1,args["samples"])**10).to(DEVICE)
    samples = sample(model, args["samples"], conditioning=cdf).detach().cpu().numpy()
    data = read_yaml(args["load"], **dataDict["regular"], **dataDict["conditioning"])
    ic(data.shape)
    samples[:, 10] = np.mod(np.abs(samples[:, 10]), np.pi) * np.sign(samples[:, 10])
    plotHist(data, samples, dataDict)


def export(args: Dict) -> None:
    model, dataDict = loadModel(modelName=args["model"])

    ws = WeightSampler()
    if dataDict["conditioning"]:
        instance_path = args["instance"]
        with open(instance_path, "r") as file:
            instance_data = yaml.safe_load(file)["environment"]

        conditioning = []
        for condition, size in dataDict["conditioning"].items():
            if condition == "rel_probability":
                cond = torch.Tensor(
                    ws.rvs(size=args["samples"]), device=DEVICE
                ).reshape(-1, 1)
            elif condition in instance_data.keys():
                cond = (
                    torch.ones(size=(args["samples"], size), device=DEVICE)
                    * instance_data[condition]
                )
            else:
                cond = (
                    torch.ones(size=(args["samples"], size), device=DEVICE)
                    * args[condition]
                )
            conditioning.append(cond)
        # conditioning = torch.concat([cdf, delta_0], dim=1)
        conditioning = torch.concat(conditioning, dim=1)
        samples = (
            sample(model, args["samples"], conditioning=conditioning)
            .detach()
            .cpu()
            .numpy()
        )
    else:
        samples = sample(model, args["samples"]).detach().cpu().numpy()

    sampleDict = outputToDict(samples, dataDict)
    dt = 0.1
    outputList = []
    length = args["samples"]
    dataset, limit = pruneDataset(
        sampleDict["actions"], sampleDict["theta_0"], length=length
    )
    for actions, states in dataset:
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
    # ic(len(outputList))
    if args["out"] is None:
        out = f"output/model_unicycle_bugtrap_n{length}_l5.yaml"
    else:
        out = args["out"]
    Path(out).parents[0].mkdir(parents=True, exist_ok=True)
    with open(out, "w") as file:
        yaml.safe_dump(outputList, file, default_flow_style=None)


def listRun():
    raise NotImplementedError
