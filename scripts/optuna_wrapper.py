from typing import Literal, Optional
import optuna
import sys
import logging
import diffmp
import itertools
from pathlib import Path
import torch
import numpy as np
import random

from diffmp.utils.reporting import OptunaReporter

DYNAMICS = "unicycle1_v0"
N_ROBOTS = 2
OBJECTIVE = "Composite"
MAX_EPOCH = 500
MIN_EPOCH = 200
N_TRIALS = 50
# DYNAMICS = "car1_v0"
# DYNAMICS = "unicycle2_v0"
DATASET = Path("data/training_datasets/unicycle1_v0_x2.h5")


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Avoids nondeterministic behavior


# OPTIMIZER = "adam"
def get_condition(
    trial: optuna.Trial,
    dynamics: Literal["unicycle1_v0", "unicycle2_v0", "car1_v0"],
    length: Optional[int] = None,
):
    opts_condition = [
        # "rel_l",
        # "p_obstacles",
        "Theta_s",
        "Theta_g",
        # "Theta_2_s",
        # "Theta_2_g",
        # "env_width",
        # "env_height",
        # "area",
        # "area_free",
        # "area_blocked",
    ]
    if dynamics == "car1_v0":
        opts_condition.extend(["Theta_2_s", "Theta_2_g"])
    conditions_chosen = []
    for condition in opts_condition:
        if isinstance(length, int):
            chosen = trial.suggest_categorical(f"{condition}_l{length:02d}", [0, 1])
        else:
            chosen = trial.suggest_categorical(f"{condition}", [0, 1])
        if not chosen:
            continue
        conditions_chosen.append(condition)
    return conditions_chosen


def get_regular(
    trial: optuna.Trial, dynamics: Literal["unicycle1_v0", "unicycle2_v0", "car1_v0"]
):
    regular = ["actions", "Theta_0"]
    if dynamics == "unicycle2_v0":
        regular.extend(["s_0", "phi_0"])
    elif dynamics == "car1_v0":
        regular.extend(["Theta_2_0"])
    return regular


def objective_composite(trial: optuna.Trial) -> float:
    regular = get_regular(trial, DYNAMICS)
    lengths = [5, 10, 15, 20]
    configs = []
    # denoising_steps = trial.suggest_int("denoising_steps", 10, 50, step=5)
    classify = trial.suggest_categorical("classify_actions", [True, False])
    # rel_c_weight = trial.suggest_int("rel_c_weight", 0, 20, step=5)
    # rel_c_weight = 10
    # conditions = get_condition(trial, DYNAMICS)
    conditions = [
        "rel_l",
        "x_s",
        "x_g",
        "y_s",
        "y_g",
        "Theta_s",
        "Theta_g",
        # "env_width",
        # "env_height",
    ]
    n_hidden = {5: 3, 10: 3, 15: 4, 20: 4}
    # denoise = {5: 30, 10: 30, 15: 30, 20: 30}
    # lr = {5: 0.00311, 10: 0.001699, 15: 0.000758, 20: 0.0003744}
    for length in lengths:
        # this_cond = get_condition(trial, DYNAMICS, length)
        this_cond = conditions
        # this_cond = conditions.copy()
        # if trial.suggest_categorical(f"rel_l{length}", [1, 0]):
        #     this_cond = conditions + ["rel_l"]
        # if trial.suggest_categorical(f"p_o_{length}", [1, 0]):
        #     this_cond = this_cond + ["p_obstacles"]
        data = {
            "dynamics": DYNAMICS,
            "timesteps": length,
            "n_robots": N_ROBOTS,
            "problem": "Any",
            "s_hidden": trial.suggest_int(
                f"s_hidden_l{length:02d}", 128, 512, log=True
            ),
            # "s_hidden": 512,
            "n_hidden": n_hidden[length],
            # "n_hidden": trial.suggest_int(f"n_hidden_l{length}", 3, 7),
            "regular": regular,
            "conditioning": this_cond,
            "loss_fn": "mse",
            "dataset": DATASET,
            "time_embedding_size": 16,
            # "denoising_steps": denoise[length],
            "denoising_steps": trial.suggest_int(
                f"denoising_steps_l{length:02d}", 25, 50, step=5
            ),
            "lr": trial.suggest_float(f"lr_l{length:02d}", 1e-4, 5e-3, log=True),
            # "lr": lr[length],
            "batch_size": 256,
            # "noise_schedule": "sigmoid",
            "noise_schedule": trial.suggest_categorical(
                f"noise_schedule_l{length:02d}", ["linear_scaled", "cosine"]
            ),
            "dataset_size": 1000,
            "reporters": [],
            "weights": {},
            "classify_actions": classify,
            # "weights": {"misc": {"rel_c": rel_c_weight}},
        }
        if trial.suggest_categorical(f"discretize_l{length:02d}", [True, False]):
            data["discretize"] = {"method": "sd", "resolution": 16}

        config = diffmp.torch.Config.from_dict(data)
        configs.append(config)
    models = [diffmp.torch.Model(c) for c in configs]
    optuna_reporter = OptunaReporter()
    optuna_reporter.trial = trial

    # n_mp = 100

    # num_5 = trial.suggest_int("num_5", 10, 70)
    # num_10 = trial.suggest_int("num_10", 10, n_mp - num_5 - 20)
    # num_15 = trial.suggest_int("num_15", 10, n_mp - num_5 - num_10 - 10)
    # num_20 = n_mp - (num_5 + num_10 + num_15)

    # allocation = {5: num_5, 10: num_10, 15: num_15, 20: num_20}
    # allocation = {5: 51, 10: 19, 15: 16, 20: 14}

    # composite_config = diffmp.torch.CompositeConfig(
    #     DYNAMICS, models, optuna_reporter, allocation
    # )
    composite_config = diffmp.torch.CompositeConfig(
        DYNAMICS, models, N_ROBOTS, optuna_reporter
    )
    diffmp.torch.train_composite(composite_config, MAX_EPOCH, auto_save=False)
    return optuna_reporter.best_test


def objective_standard(trial: optuna.Trial):
    iterable = [
        # "rel_l",
        # "p_obstacles",
        "Theta_s",
        "Theta_g",
        # "Theta_2_s",
        # "Theta_2_g",
        # "env_width",
        # "env_height",
        # "area",
        # "area_free",
        # "area_blocked",
    ]
    conditions_chosen = []
    for condition in iterable:
        if trial.suggest_categorical(condition, [0, 1]):
            continue
        conditions_chosen.append(condition)
    regular = ["actions", "Theta_0"]
    # regular.append(trial.suggest_categorical("theta_0_repr", ["theta_0", "Theta_0"]))
    # regular.append(
    #     trial.suggest_categorical("theta_2_0_repr", ["theta_2_0", "Theta_2_0"])
    # )
    data = {
        "dynamics": DYNAMICS,
        "n_robots": N_ROBOTS,
        "timesteps": 10,
        "problem": "Any",
        # "s_hidden": trial.suggest_categorical("s_hidden", [128, 256, 512, 1024]),
        "s_hidden": trial.suggest_int("s_hidden", 128, 512, log=True),
        "n_hidden": trial.suggest_int("n_hidden", 3, 7),
        "regular": regular,
        "conditioning": conditions_chosen,
        "loss_fn": "mse",
        "dataset": DATASET,
        "denoising_steps": trial.suggest_int("denoising_steps", 25, 50, step=5),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "batch_size": 256,
        "noise_schedule": "linear_scaled",
        # "noise_schedule": trial.suggest_categorical(
        #     "noise_schedule", ["linear_scaled", "linear", "sigmoid"]
        # ),
        "dataset_size": 10000,
        "reporters": ["aim", "tqdm"],
        "weights": {},
        "discretize": {
            "method": "sd",
            "resolution": trial.suggest_int("discrete_res", 16, 64, log=True),
        },
        # "weights": {
        #     "misc": {"rel_c": trial.suggest_float("rel_c_weight", 0.0, 20.0, step=0.5)}
        # },
        "classify_actions": True,
    }
    config = diffmp.torch.Config.from_dict(data)
    optuna_reporter = diffmp.utils.OptunaReporter(4)
    optuna_reporter.trial = trial
    config.reporters.append(optuna_reporter)
    model = diffmp.torch.Model(config)
    model.path = Path(f"data/models/optuna/{trial._trial_id}")
    diffmp.torch.train(model, MAX_EPOCH, test_epoch=25)
    return optuna_reporter.best_test


def main():
    # diffmp.utils.DEVICE = "cpu"
    diffmp.utils.DEVICE = "cpu"
    set_seed(42)
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    # study_name = "unicycle2_v0"  # Unique identifier of the study.

    study_name = f"{DYNAMICS}_x{N_ROBOTS} - {OBJECTIVE}"
    storage_name = "sqlite:///data/optuna/optuna.db"

    study = optuna.create_study(
        direction="maximize",
        storage=storage_name,
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=10),
        # sampler=optuna.samplers.RandomSampler(),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=MIN_EPOCH, max_resource=MAX_EPOCH, reduction_factor=3
        ),
        # pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        load_if_exists=True,
    )
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.FAIL:
            study.enqueue_trial(trial.params, skip_if_exists=True)
    if OBJECTIVE == "Standard":
        study.optimize(objective_standard, n_trials=N_TRIALS)
    elif OBJECTIVE == "Composite":
        study.optimize(objective_composite, n_trials=N_TRIALS)
    else:
        raise Exception
    pruned_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.PRUNED]
    )
    complete_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
    )

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
