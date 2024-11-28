from functools import partial
import os
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.tune.stopper import ExperimentPlateauStopper
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
import ray.cloudpickle as pickle
from diffusion_model import train_raytune
from data import load_dataset


if torch.cuda.is_available():
    DEVICE = "cuda:0"
    torch.cuda.set_device(0)
    torch.cuda.init()
else:
    DEVICE = "cpu"


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    # torch.manual_seed(0)
    path = Path().absolute()
    # config = {
    #     "n_hidden": tune.choice([1, 2]),
    #     "s_hidden": tune.choice([2**i for i in range(3, 9)]),
    #     "lr": tune.loguniform(1e-4, 1e-1),
    #     "batch_size": tune.choice([128, 512]),
    # }
    config = {
        "lr": tune.loguniform(1e-4, 1e-2),
        # "batch_size": tune.randint([128, 256, 512]),
        "denoising_steps": tune.randint(20, 100),
        # "conditioning": {
        #     "rel_probability": tune.choice([0, 1]),
        #     "location": tune.choice([0, 1]),
        #     "p_obstacles": tune.choice([0, 1]),
        #     "env_width": tune.choice([0, 1]),
        #     "env_height": tune.choice([0, 1]),
        #     "area": tune.choice([0, 1]),
        #     "area_blocked": tune.choice([0, 1]),
        #     "env_theta_start": tune.choice([0, 1]),
        #     "env_theta_goal": tune.choice([0, 1]),
        # "avg_clustering": tune.choice([0, 1]),
        # "avg_node_connectivity": tune.choice([0, 1]),
        # },
        # "n_hidden": tune.quniform(2, 4, 1),
        "n_hidden": tune.randint(2, 4),
        # "s_hidden": tune.randint(128, 600),
        "s_hidden": tune.choice([2**n for n in range(7, 10)]),
    }
    model_static = {
        "type": "unicycle1_v0",
        # "lr": 1e-3,
        "batch_size": 512,
        # "denoising_steps": 30,
        # "n_hidden": 3,
        # "s_hidden": 128,
        "regular": {"actions": 10, "R2SVD": 2},
        "conditioning": {"location": 1, "p_obstacles": 1, "env_theta_goal": 1},
        "loss": "mse",
        "dataset": path / "data/training_datasets/rand_env_40k.parquet",
        "device": DEVICE,
        "dataset_size": 5000,
    }
    # model_static = {
    #     "type": "unicycle1_v0",
    #     "lr": 1e-3,
    #     "batch_size": 512,
    #     "denoising_steps": 30,
    #     "n_hidden": 2,
    #     "s_hidden": 256,
    #     "regular": {"l5": 1, "l10": 1, "l15": 1},
    #     "cascade": {"l5": "abc", "l10": "abc", "l15": "abc"},
    #     # "conditioning": {"location": 1, "p_obstacles": 1},
    #     "loss": "mse",
    #     "dataset": path / "data/training_datasets/cascade",
    #     "device": DEVICE,
    #     "dataset_size": 10000,
    # }
    # current_best = [
    #     {
    #         # "n_hidden": 2,
    #         # "s_hidden": 512,
    #         "denoising_steps": 50,
    #         "lr": 1e-3,
    #     }
    # ]
    # scheduler = PopulationBasedTraining(
    #     time_attr="training_iteration",
    #     metric="loss",
    #     mode="min",
    #     perturbation_interval=20,
    #     hyperparam_mutations=config,
    # )
    search_alg = OptunaSearch(metric="loss", mode="min")
    # search_alg = HyperOptSearch(
    #     metric="loss", mode="min", points_to_evaluate=current_best
    # )
    # search_alg = TuneBOHB(metric="loss", mode="min")
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=8)
    # scheduler = HyperBandForBOHB(
    #     time_attr="training_iteration",
    #     metric="loss",
    #     mode="min",
    #     max_t=max_num_epochs,
    #     reduction_factor=2,
    #     stop_last_trials=False,
    # )
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=25,
        reduction_factor=2,
    )
    trainable = tune.with_resources(
        partial(
            train_raytune,
            model_static=model_static,
            nepochs=max_num_epochs + 1,
        ),
        {"cpu": 1, "gpu": gpus_per_trial},
    )
    # stopper = ExperimentPlateauStopper(metric="loss", mode="min", patience=10)
    # tuner = tune.Tuner(
    #     trainable,
    #     tune_config=tune.TuneConfig(
    #         num_samples=num_samples,
    #         scheduler=scheduler,
    #         search_alg=search_alg,
    #         # num_samples=num_samples,
    #         # scheduler=scheduler,
    #     ),
    #     # run_config=train.RunConfig(stop=stopper),
    #     param_space=config,
    # )
    # result = tune.run(
    #     partial(
    #         train_raytune,
    #         model_static=model_static,
    #         nepochs=max_num_epochs,
    #     ),
    #     resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
    #     config=config,
    #     num_samples=num_samples,
    #     scheduler=scheduler,
    # )

    result = tune.Tuner.restore(
        "/home/julius/ray_results/train_raytune_2024-11-14_13-41-14",
        trainable=trainable,
    ).fit()
    # result = tuner.fit()
    # breakpoint()
    best_trial = result.get_best_result(metric="loss", mode="min", scope="all")

    print(f"Best trial config: {best_trial.config}")
    print(f"Best Loss: {best_trial.metrics['loss']}")
    breakpoint()
    df = result.get_dataframe(metric="loss", mode="min")
    df.to_parquet("ray_tune.parquet")
    print(df)
    # df = best_trial.metrics_dataframe
    # print(f"Best trial final validation loss: {best_trial.metrics['loss']}")
    # print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    # best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     if gpus_per_trial > 1:
    #         best_trained_model = nn.DataParallel(best_trained_model)
    # best_trained_model.to(device)

    # best_checkpoint = result.get_best_checkpoint(
    #     trial=best_trial, metric="accuracy", mode="max"
    # )
    # with best_checkpoint.as_directory() as checkpoint_dir:
    #     data_path = Path(checkpoint_dir) / "data.pkl"
    #     with open(data_path, "rb") as fp:
    #         best_checkpoint_data = pickle.load(fp)

    #     best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
    #     test_acc = test_accuracy(best_trained_model, device)
    #     print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=128, max_num_epochs=2000, gpus_per_trial=0)
