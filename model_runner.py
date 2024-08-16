from pathlib import Path
import pandas as pd
import seaborn as sns
import yaml
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
import torch
import torch.utils.data
from icecream import ic
import sys
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple
from data import (
    WeightSampler,
    calc_unicycle_states,
    circle_SO2,
    data_gen,
    metric,
    pruneDataset,
    read_yaml,
    spiral_points,
    prune,
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


def plotHist(data: np.ndarray, samples: np.ndarray, dataDict: Dict):
    action_length = dataDict["regular"]["actions"]
    mp_length = action_length // 2

    data_actions_mean = np.mean(
        data[:, :action_length].reshape(data.shape[0], mp_length, 2), axis=1
    )
    sample_actions_mean = np.mean(
        samples[:, :action_length].reshape(samples.shape[0], mp_length, 2), axis=1
    )
    actions_mean = np.concatenate([data_actions_mean, sample_actions_mean])

    data_theta_0 = data[:, action_length]
    sample_theta_0 = samples[:, action_length]

    # ws = WeightSampler()
    # data_weights = ws.ppf(data[:, action_length+1])
    data_weights = data[:, action_length+1]
    pltdict = {
        "s": actions_mean[:,0].flatten(),
        "phi": actions_mean[:,1].flatten(),
        "theta_0": np.concatenate([data_theta_0, sample_theta_0]).flatten(),
        "weights": data_weights.tolist() + samples.shape[0] * [1],
        "source": data.shape[0] * ["training data"] + samples.shape[0] * ["predicted"],
    }
    pltdf = pd.DataFrame(pltdict)
    # breakpoint()
    sns.set_theme()
    # breakpoint()
    # sns.displot(pltdf, x="s", y="phi", hue="source", kind="kde", thresh=0.2, levels=4)
    g = sns.PairGrid(pltdf, hue="source", corner=True,vars=["s", "phi", "theta_0"])
    # g = sns.PairGrid(pltdf, hue="source", corner=True)
    # g.map_upper(sns.histplot)
    g.map_diag(sns.histplot, element="poly", common_norm=False, stat="percent", weights=pltdf["weights"],bins=50)
    g.map_lower(sns.kdeplot, levels=4, common_norm=False, weights=pltdf["weights"])
    g.add_legend()

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
            "conditioning": {"rel_probability": 1},
        }
        data = read_yaml(
            args["load"], **data_dict["regular"], **data_dict["conditioning"]
        )

    ic(data.shape)
    # data = prune(data, 0.1)
    ic(data.shape)
    dataset = TensorDataset(torch.Tensor(data))
    if training_size < data.shape[0]:
        dataset_split = torch.utils.data.random_split(
            dataset, [training_size, data.shape[0] - training_size]
        )[0]
    else:
        dataset_split = dataset
    ic(len(dataset_split))
    loader = DataLoader(dataset_split, batch_size=50, shuffle=True)

    model = Net(data_dict)
    trained_model = train(model, loader, data_dict, args["epochs"])
    model_save = "bugtrap_rel_logistic.pt"
    torch.save(trained_model.state_dict(), model_save)
    print(f"Model saved as {model_save}")


def loadRun(args: Dict):
    dataDict = {
        "type": "car",
        "n_hidden": 6,
        "s_hidden": 256,
        "regular": {"actions": 10, "theta_0": 1},
        "conditioning": {"rel_probability": 1},
    }
    model = loadModel(modelPath=Path(args["model"]), dataDict=dataDict)
    ws = WeightSampler()
    cdf = torch.Tensor(ws.rvs(size=args["samples"])).to(DEVICE)
    samples = sample(model, args["samples"], conditioning=cdf).detach().cpu().numpy()
    data = read_yaml(args["load"], **dataDict["regular"], **dataDict["conditioning"])
    ic(data.shape)
    prunedData = prune(data, 0.1)
    uniqueData = np.unique(data, axis=0)
    ic(prunedData.shape)
    ic(uniqueData.shape)
    plotHist(data, samples)


def export(args: Dict) -> None:
    # ic(args)
    dataDict = {
        "type": "car",
        "n_hidden": 6,
        "s_hidden": 256,
        "regular": {"actions": 10, "theta_0": 1},
        "conditioning": {"rel_probability": 1},
    }
    model = loadModel(modelPath=Path(args["model"]), dataDict=dataDict)

    ws = WeightSampler()
    cdf = torch.Tensor(ws.rvs(size=args["samples"])).to(DEVICE)
    samples = sample(model, args["samples"], conditioning=cdf).detach().cpu().numpy()
    sampleDict = outputToDict(samples, dataDict)
    # sampleDict["actions"] = np.clip(sampleDict["actions"], -0.5, 0.5)
    # s_min = np.min(sampleDict["actions"][:,:,0])
    # s_max = np.max(sampleDict["actions"][:,:,0])
    # phi_min = np.min(sampleDict["actions"][:,:,1])
    # phi_max = np.max(sampleDict["actions"][:,:,1])

    # breakpoint()
    dt = 0.1
    outputList = []
    # breakpoint()
    # statesCheck = []
    # ic(args["samples"])
    length = args["samples"]
    dataset, limit = pruneDataset(
        sampleDict["actions"], sampleDict["theta_0"], length=length
    )
    # breakpoint()
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
        # states = calc_unicycle_states(actions, dt=dt, start=[0, 0, float(theta_0)])
        # if statesCheck:
        #     diff = np.linalg.norm(np.array(statesCheck) - states, axis=(1, 2))
        #
        #     if (diff < 0.2).any():
        #         continue
        # statesCheck.append(states)
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
    with open(out, "w") as file:
        yaml.safe_dump(outputList, file, default_flow_style=None)


def listRun():
    raise NotImplementedError


#     db = Database("data.json")
#     data = db.tabulate(keys=["uuid", "type", "s_hidden"])
#     print(data)
