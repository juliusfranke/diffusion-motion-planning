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
    pruneDataset,
    load_dataset,
    SUPP_COMPLETE,
)
from diffusion_model import DEVICE, Net, sample, train
from icecream import ic
from torch.utils.data import DataLoader, TensorDataset
import logging

logger = logging.getLogger(__name__)


def plotTraining(trainingData) -> None:
    plt.scatter(trainingData[:, 0], trainingData[:, 1], label="training")


def getViolations(samples):
    samples = samples[:, :10]
    is_violation = np.abs(samples) > 0.5

    total_violations = np.sum(is_violation)
    total = np.prod(samples.shape)
    violations = total_violations / total
    violation_score = np.sum(is_violation * (np.abs(samples[:, :10]) - 0.5)) / total
    return violations, violation_score


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
    model.load_state_dict(torch.load(weightsPath, weights_only=True))
    model.to(DEVICE)

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
    logger.info(f"Device: {DEVICE}")
    training_size = args["trainingsize"]
    if args["generate"]:
        raise NotImplementedError
    else:
        data_dict = {
            "type": "unicycle1_v0",
            "n_hidden": args["nhidden"],
            "s_hidden": args["shidden"],
            "regular": {"actions": 10, "theta_0": 1},
            "conditioning": {
                # "delta_0": 1,
                "rel_probability": 1,
                "env_theta_start": 1,
                "env_theta_goal": 1,
                # "area": 1,
                # "env_width": 1,
                # "env_height": 1,
                # "area_blocked": 1,
                # "avg_node_connectivity": 1,
                # "avg_clustering": 1,
                # "cost": 1,
                # "avg_shortest_path": 1,
            },
        }
        data = load_dataset(
            args["load"],
            regular=data_dict["regular"],
            conditioning=data_dict["conditioning"],
        )

    logger.debug(f"Dataset size: {data.shape[0]}")

    dataset = TensorDataset(torch.tensor(data, device=DEVICE))
    if training_size < data.shape[0] and training_size != -1:
        dataset_split = torch.utils.data.random_split(
            dataset, [training_size, data.shape[0] - training_size]
        )[0]
        logger.debug(f"Split Dataset - size: {len(dataset_split)}")
    else:
        dataset_split = dataset
    batch_size = len(dataset_split)
    logger.debug(f"Batch size: {batch_size}")

    loader = DataLoader(dataset_split, batch_size=batch_size, shuffle=True)

    model = Net(data_dict)

    trained_model, losses = train(model, loader, args["epochs"])
    model_path = Path("data/models/test_rand_env_theta_l5___.pt")
    # model_path = Path("data/models/qtest.pt")
    if model_path.exists():
        if input(f"{model_path} already exists, overwrite? (y/n) ") != "y":
            return None
    torch.save(trained_model.state_dict(), model_path)
    with open(model_path.with_suffix(".yaml"), "w") as file:
        yaml.safe_dump(data_dict, file, default_flow_style=None)

    logger.info(f"Model saved as {model_path}")
    plt.plot(np.array(losses))
    plt.yscale("log")
    plt.savefig(model_path.with_suffix(".png"))
    pd.DataFrame(np.array(losses), columns=["training_loss"]).to_csv(
        model_path.with_suffix(".csv")
    )


def loadRun(args: Dict):
    model, data_dict = loadModel(modelName=args["model"])
    ws = WeightSampler()
    cdf = torch.Tensor(ws.rvs(size=args["samples"])).to(DEVICE)
    # cdf = torch.Tensor(ws.ppf(np.linspace(0,1,args["samples"]))).to(DEVICE)
    # cdf = torch.Tensor(np.linspace(0,1,args["samples"])**10).to(DEVICE)
    samples = sample(model, args["samples"], conditioning=cdf).detach().cpu().numpy()
    data = load_dataset(
        args["load"],
        regular=data_dict["regular"],
        conditioning=data_dict["conditioning"],
    )
    ic(data.shape)
    samples[:, 10] = np.mod(np.abs(samples[:, 10]), np.pi) * np.sign(samples[:, 10])
    plotHist(data, samples, data_dict)


def export(args: Dict) -> None:
    model, data_dict = loadModel(modelName=args["model"])

    ws = WeightSampler()
    if data_dict["conditioning"]:
        instance_path = args["instance"]
        with open(instance_path, "r") as file:
            instance_data = yaml.safe_load(file)
        env_data = instance_data["environment"]

        conditions = sorted(
            [
                key
                for key in data_dict["conditioning"].keys()
                if key != "rel_probability"
            ]
        )
        conditioning = []
        for condition in conditions:
            # if condition not in dataDict["conditioning"].keys():
            #     continue
            # else:
            size = data_dict["conditioning"][condition]
            print(f"Adding {condition}")
            # if condition == "rel_probability":
            #    cond = torch.tensor(
            #        ws.rvs(size=args["samples"]), device=DEVICE
            #    ).reshape(-1, 1)
            if condition in env_data.keys():
                cond = (
                    torch.ones(
                        size=(args["samples"], size), device=DEVICE, dtype=torch.float64
                    )
                    * env_data[condition]
                )
                print(env_data[condition])
            elif condition == "env_theta_start":
                cond = (
                    torch.ones(
                        size=(args["samples"], size), device=DEVICE, dtype=torch.float64
                    )
                    * instance_data["robots"][0]["start"][2]
                )
                print(instance_data["robots"][0]["start"][2])
            elif condition == "env_theta_goal":
                cond = (
                    torch.ones(
                        size=(args["samples"], size), device=DEVICE, dtype=torch.float64
                    )
                    * instance_data["robots"][0]["goal"][2]
                )
                print(instance_data["robots"][0]["goal"][2])
            else:
                cond = (
                    torch.ones(
                        size=(args["samples"], size), device=DEVICE, dtype=torch.float64
                    )
                    * args[condition]
                )
                print(args[condition])
            conditioning.append(cond)
        if "rel_probability" in data_dict["conditioning"].keys():
            print("Adding rel_probability")
            conditioning.append(
                torch.tensor(ws.rvs(size=args["samples"]), device=DEVICE).reshape(-1, 1)
            )
        # conditioning = torch.concat([cdf, delta_0], dim=1)
        conditioning = torch.concat(conditioning, dim=1).to(DEVICE)
        # samples = (
        #         sample(model, args["samples"], conditioning=conditioning)
        #         .detach()
        #         .cpu()
        #         .numpy()
        #     )
        # breakpoint()
        samples = (
            sample(model, args["samples"], conditioning=conditioning)
            .detach()
            .cpu()
            .numpy()
        )
        # samples = []
        # best_score = np.inf
        # best_steps = None

        # for steps in [50, 100, 150, 200, 250, 300, 350, 400]:
        #     samples_ = (
        #         sample(model, args["samples"], conditioning=conditioning,n_steps=steps)
        #         .detach()
        #         .cpu()
        #         .numpy()
        #     )
        #     violations, violation_score = getViolations(samples_)
        #     score = violation_score * violations
        #     print(f"{steps} steps : {violation_score} * {violations} = {score}" )
        #     if score < best_score:
        #         samples = samples_
        #         best_score = score
        #         best_steps = steps

    else:
        samples = sample(model, args["samples"]).detach().cpu().numpy()
    # print(f"Best steps: {best_steps} with {best_score}")
    # breakpoint()
    violations, violation_score = getViolations(samples)
    score = violation_score * violations
    print(f"{violation_score} * {violations} = {score}")
    sampleDict = outputToDict(samples, data_dict)
    sampleDict["actions"] = np.clip(sampleDict["actions"], -0.5, 0.5)
    # breakpoint()
    # sampleDict["theta_0"] = np.sign(sampleDict["theta_0"]) * np.abs(sampleDict["theta_0"]) % np.pi

    ic(max(sampleDict["theta_0"]))
    ic(min(sampleDict["theta_0"]))
    sampleDict["theta_0"] = (sampleDict["theta_0"] + np.pi) % (2 * np.pi) - np.pi
    ic(max(sampleDict["theta_0"]))
    ic(min(sampleDict["theta_0"]))
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
