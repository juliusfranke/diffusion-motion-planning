from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.utils.data
import yaml
import time
import msgpack
from data import (
    WeightSampler,
    pruneDataset,
    load_dataset,
    load_allocator_dataset,
    sorted_nicely,
    get_violations,
    SUPP_COMPLETE,
    symmetric_orthogonalization,
)
from diffusion_model import DEVICE, Net, sample, train_normal
from icecream import ic
from torch.utils.data import DataLoader, TensorDataset
import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

logger = logging.getLogger(__name__)
torch.manual_seed(0)


def plotTraining(trainingData) -> None:
    plt.scatter(trainingData[:, 0], trainingData[:, 1], label="training")


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
    for outputType in sorted_nicely(dataDict["regular"].keys()):
        length = dataDict["regular"][outputType]
        # print(outputType, length, idx)
        if outputType == "actions":
            returnDict[outputType] = modelOutput[:, idx : idx + length].reshape(
                sampleSize, length // 2, 2
            )
        # elif outputType == "R4SVD":
        #     returnDict[outputType] = modelOutput[:, idx : idx + length].reshape(
        #         sampleSize, 2, 2
        #     )
        else:
            returnDict[outputType] = modelOutput[:, idx : idx + length]
        idx += length
    return returnDict


def plotError(errorData) -> None:
    pass


def trainRun(args: Dict):
    with open(args["config"], "r") as file:
        data_dict = yaml.safe_load(file)
    if "cascade" in data_dict.keys():
        trainComposite(args, data_dict)
    else:
        trainSingle(args, data_dict)


def trainComposite(args: Dict, data_dict: Dict):
    assert set(data_dict["regular"].keys()) == set(data_dict["cascade"].keys())

    dataset_folder = Path(args["dataset"])
    model_path = (Path("data/models/") / args["name"]).with_suffix(".pt")
    if model_path.exists():
        if input(f"{model_path} already exists, overwrite? (y/n) ") != "y":
            return None

    with open(model_path.with_suffix(".yaml"), "w") as file:
        yaml.safe_dump(data_dict, file, default_flow_style=None)
    # Train allocator
    data = load_allocator_dataset(
        dataset_folder, data_dict["regular"], data_dict["conditioning"]
    ).to_numpy()
    trainModel(args, data_dict, data, model_path)


def trainSingle(args: Dict, data_dict: Dict):
    name = args["name"]
    model_path = (Path("data/models/") / name).with_suffix(".pt")
    if model_path.exists():
        if input(f"{model_path} already exists, overwrite? (y/n) ") != "y":
            return None
    logger.info(f"Device: {DEVICE}")

    with open(model_path.with_suffix(".yaml"), "w") as file:
        yaml.safe_dump(data_dict, file, default_flow_style=None)

    data = load_dataset(
        args["dataset"],
        regular=data_dict["regular"],
        conditioning=data_dict["conditioning"],
    ).to_numpy()
    trainModel(args, data_dict, data, model_path)


def trainModel(args: Dict, data_dict: Dict, data: np.ndarray, model_path: Path):
    name = args["name"]
    training_size = args["trainingsize"]
    val_split = args["valsplit"]

    conditions = [
        key for key in data_dict["conditioning"].keys() if key != "rel_probability"
    ]
    cond_str = " + ".join(conditions)
    logger.info(f"Dataset size: {data.shape[0]}")

    dataset = TensorDataset(torch.tensor(data, device=DEVICE))
    # test_dataset = TensorDataset(torch.tensor(test_data, device=DEVICE))

    if training_size == -1 or training_size > len(dataset):
        training_size = len(dataset)
    else:
        dataset = torch.utils.data.random_split(
            dataset, [training_size, len(dataset) - training_size]
        )[0]
        logger.info(f"Selected dataset size: {training_size}")
    # if training_size < len(test_dataset):
    #     test_dataset = torch.utils.data.random_split(
    #         test_dataset, [training_size, len(test_dataset) - training_size]
    #     )[0]

    len_training = int(training_size * val_split)
    len_validation = int(training_size - len_training)

    training_set, validation_set = torch.utils.data.random_split(
        dataset, [len_training, len_validation]
    )
    batch_size_train = min(len_training // 10, 4096)
    batch_size_val = min(len_validation // 10, 4096)
    logger.debug(f"Batch size: {batch_size_train}")
    # breakpoint()
    training_loader = DataLoader(
        training_set, batch_size=batch_size_train, shuffle=True
    )
    validation_loader = DataLoader(
        validation_set, batch_size=batch_size_val, shuffle=True
    )
    # test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

    model = Net(data_dict)
    model.path = model_path
    timestamp = datetime.now().strftime("%d-%m-%y %H:%M")
    tb_writer = SummaryWriter(f"runs/{name}/{timestamp}")
    # example = iter(validation_loader)
    # ts = torch.ones((batch_size_val, 1)).to("cpu")
    # example_data = torch.concat([example._next_data()[0].to("cpu"), ts], dim=-1)
    # tb_writer.add_graph(model, example_data)
    # for regular, size in data_dict["regular"].items():
    #     tb_writer.add_text(str(regular), str(size))
    start = time.time()
    trained_model = train_normal(
        model,
        training_loader,
        validation_loader,
        tb_writer,
        args["epochs"],
        data_dict["denoising_steps"],
    )
    end = time.time()
    duration = end - start
    # hparams = {
    #     key: value
    #     for key, value in (
    #         data_dict["regular"] | data_dict["conditioning"] | hparams
    #     ).items()
    # }
    # print(hparams)

    all_keys = list(data_dict["regular"].keys()) + list(
        data_dict["conditioning"].keys()
    )

    tb_writer.add_hparams(
        {
            "n_hidden": model.info["n_hidden"],
            "s_hidden": model.info["s_hidden"],
            "type": model.info["type"],
            "lr": model.info["lr"],
            "epochs": model.info["epochs"],
            "val_split": val_split,
            "dataset_size": training_size,
            "batch_size": batch_size_train,
            "training_time": duration,
            "epoch/s": model.info["epochs"] / duration,
            "conditioning": cond_str,
            "loss_fn": data_dict["loss"],
            "denoising_steps": data_dict["denoising_steps"],
        }
        | {f"z_{key}": str(key in all_keys) for key in SUPP_COMPLETE},
        {f"min_{key}": min(value) for key, value in model.info["train_metrics"].items()}
        | {
            f"min_{key}": min(value) for key, value in model.info["val_metrics"].items()
        },
        # | {
        #     f"min_{key}": min(value)
        #     for key, value in model.info["test_metrics"].items()
        # },
        run_name=".",
        hparam_domain_discrete={f"z_{key}": ["True", "False"] for key in SUPP_COMPLETE},
    )

    tb_writer.close()
    # model_path = Path("data/models/qtest.pt")

    # logger.info(f"Model saved as {model_path}")


def loadRun(args: Dict):
    model, data_dict = loadModel(modelName=args["model"])
    ws = WeightSampler()
    cdf = torch.Tensor(ws.rvs(size=args["samples"]), device=DEVICE)
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
    instance_path = args["instance"]
    length = args["samples"]
    with open(instance_path, "r") as file:
        instance_data = yaml.safe_load(file)

    if "cascade" in data_dict.keys():
        data = sampleCascade(args, data_dict, instance_data, model)
    else:
        data = sampleToPrimitive(
            sampleSingle(args, data_dict, instance_data, model), data_dict
        )

    if args["out"] is None:
        out = f"output/model_unicycle_bugtrap_n{length}_l5.msgpack"
    else:
        out = args["out"]
    Path(out).parents[0].mkdir(parents=True, exist_ok=True)
    # with open(out, "wb") as file:
    #     packed = msgpack.packb(outputList)
    #     file.write(packed)
    with open(out, "w") as file:
        yaml.safe_dump(data, file, default_flow_style=None)


def sampleCascade(
    args: Dict, allocator_dict: Dict, instance_data: Dict, allocator_model: Net
):
    allocator = sampleSingle(args, allocator_dict, instance_data, allocator_model)
    _, counts = np.unique(np.argmax(allocator, axis=1), return_counts=True)
    data = []
    for model_name, count in zip(
        sorted_nicely(allocator_dict["cascade"].values()), counts
    ):
        model, data_dict = loadModel(modelName=model_name)
        args["samples"] = count
        samples = sampleSingle(args, data_dict, instance_data, model)
        data.extend(sampleToPrimitive(samples, data_dict))

    return data


def sampleSingle(args: Dict, data_dict: Dict, instance_data: Dict, model: Net):
    ws = WeightSampler()
    if data_dict["conditioning"]:
        env_data = instance_data["environment"]

        conditions = sorted_nicely([key for key in data_dict["conditioning"].keys()])
        conditioning = []
        for condition in conditions:
            size = data_dict["conditioning"][condition]
            # print(f"Adding {condition}")
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
                # print(env_data[condition])
            elif condition == "env_theta_start":
                cond = (
                    torch.ones(
                        size=(args["samples"], size), device=DEVICE, dtype=torch.float64
                    )
                    * instance_data["robots"][0]["start"][2]
                )
                # print(instance_data["robots"][0]["start"][2])
            elif condition == "env_theta_goal":
                cond = (
                    torch.ones(
                        size=(args["samples"], size), device=DEVICE, dtype=torch.float64
                    )
                    * instance_data["robots"][0]["goal"][2]
                )
                # print(instance_data["robots"][0]["goal"][2])
            elif condition == "rel_probability":
                cond = torch.tensor(
                    ws.rvs(size=args["samples"]), device=DEVICE, dtype=torch.float64
                ).reshape(-1, 1)
            elif condition == "location":
                cond = torch.linspace(
                    0, 1, args["samples"], device=DEVICE, dtype=torch.float64
                ).reshape(-1, 1)

            else:
                cond = (
                    torch.ones(
                        size=(args["samples"], size), device=DEVICE, dtype=torch.float64
                    )
                    * args[condition]
                )
                # print(args[condition])
            conditioning.append(cond)
        # if "rel_probability" in data_dict["conditioning"].keys():
        #     # print("Adding rel_probability")
        #     conditioning.append(
        #         torch.tensor(ws.rvs(size=args["samples"]), device=DEVICE).reshape(-1, 1)
        #     )
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
            sample(
                model,
                args["samples"],
                conditioning=conditioning,
                n_steps=data_dict["denoising_steps"],
            )
            .detach()
            .cpu()
            .numpy()
        )

    else:
        samples = sample(model, args["samples"]).detach().cpu().numpy()
    return samples


def sampleToPrimitive(samples, data_dict):
    sampleDict = outputToDict(samples, data_dict)
    sampleDict["actions"] = np.clip(sampleDict["actions"], -0.5, 0.5)

    if "theta_0" in sampleDict.keys():
        sampleDict["theta_0"] = (sampleDict["theta_0"] + np.pi) % (2 * np.pi) - np.pi
    elif "R4SVD" in sampleDict.keys():
        x = sampleDict["R4SVD"][:, 0]
        y = sampleDict["R4SVD"][:, 2]
        sampleDict["theta_0"] = np.arctan2(y, x)
        sampleDict.pop("R4SVD")
    elif "R2SVD" in sampleDict.keys():
        x = sampleDict["R2SVD"][:, 0]
        y = sampleDict["R2SVD"][:, 1]
        sampleDict["theta_0"] = np.arctan2(y, x)
        sampleDict.pop("R2SVD")
    # ic(max(sampleDict["theta_0"]))
    # ic(min(sampleDict["theta_0"]))
    # breakpoint()
    dt = 0.1
    outputList = []

    # TODO remove this
    dataset, limit = pruneDataset(
        sampleDict["actions"], sampleDict["theta_0"], length=len(sampleDict["actions"])
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
    return outputList
    # for i, (actions, states) in enumerate(dataset):
    #     tempDict = {
    #         "actions": actions.tolist(),
    #         "states": states.tolist(),
    #         "T": len(actions),
    #         "x0": states[0].tolist(),
    #         "xf": states[-1].tolist(),
    #         "name": f"m{i}",
    #     }
    # numActions = len(actions)
    # tempDict["T"] = numActions
    # tempDict["states"] = states.tolist()
    # tempDict["x0"] = states[0].tolist()
    # tempDict["xf"] = states[-1].tolist()
    # tempDict["name"] = f"m{i}"
    # outputList.append(tempDict)

    # breakpoint()
    # ic(len(outputList))


def listRun():
    raise NotImplementedError
