import optuna
import diffmp
import itertools
from pathlib import Path

DYNAMICS = "unicycle2_v0"
DATASET = Path("data/training_datasets/new_unicycle2_v0.parquet")
# OPTIMIZER = "adam"

# "dynamics": "unicycle1_v0"
# "timesteps": 5
# "problem": "bugtrap"
# "n_hidden": 3
# "s_hidden": 128
# "regular":
#   - "actions"
#   - "Theta_0"
# "conditioning":
#   - "rel_l"
#   - "p_obstacles"
#   - "theta_g"
#   # - name: "rel_p"
#   #   weight: 1.0
#   # - name: "delta_0"
# "loss_fn": "mse"
# "dataset": "data/training_datasets/new_unicycle1_v0.parquet"
# "denoising_steps": 30
# "batch_size": 256
# "lr": 1.0e-3
# "dataset_size": 10000
# "reporters":
#   - "tqdm"
# "noise_schedule": "linear_scaled"


def objective(trial: optuna.Trial):
    iterable = [
        "rel_l",
        "p_obstacles",
        "Theta_s",
        "Theta_g",
        "env_width",
        "env_height",
        "area",
        "area_free",
        "area_blocked",
    ]
    combinations = []

    for r in range(1, len(iterable) + 1):
        combinations.extend(
            [list(x) for x in itertools.combinations(iterable=iterable, r=r)]
        )
    optuna_reporter = diffmp.utils.OptunaReporter()
    optuna_reporter.trial = trial
    data = {
        "dynamics": DYNAMICS,
        "timesteps": 5,
        "problem": "Any",
        "s_hidden": trial.suggest_categorical("s_hidden", [64, 128, 256, 512]),
        "n_hidden": trial.suggest_categorical("n_hidden", [2, 3, 4, 5]),
        # "n_hidden":
        "regular": ["actions", "s_0", "phi_0"]
        + [trial.suggest_categorical("theta_repr", ["theta_0", "Theta_0"])],
        "conditioning": trial.suggest_categorical("conditioning", combinations),
        # "conditioning": ["Theta_g, rel_l"],
        # "loss_fn": trial.suggest_categorical("loss_fn", ["mse", "mae", "sinkhorn"]),
        "loss_fn": "mse",
        "dataset": DATASET,
        # "denoising_steps": trial.suggest_int("denoising_steps", 25, 40),
        "denoising_steps": 30,
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        # "lr": 1e-3,
        "batch_size": 256,
        "noise_schedule": "linear_scaled",
        "dataset_size": 5000,
        "reporters": ["aim", "tqdm"],
    }
    config = diffmp.torch.Config.from_dict(data)
    config.reporters.append(optuna_reporter)
    model = diffmp.torch.Model(config)
    model.path = Path(f"data/models/optuna/{trial._trial_id}")
    return diffmp.torch.train(model, 500)


def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
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
