from pathlib import Path

import numpy.typing as npt
import torch
import yaml

import diffmp


def out_to_mps(actions: npt.NDArray, states: npt.NDArray, out_path: Path):
    length = actions.shape[1]
    mps = []
    # breakpoint()
    for i in range(length):
        mp = {
            "time_stamp": 0,
            "cost": 0.5,
            "feasible": 1,
            "traj_feas": 1,
            "goal_feas": 1,
            "start_feas": 1,
            "col_feas": 1,
            "x_bounds_feas": 1,
            "u_bounds_feas": 1,
            "start": states[0, i, :].tolist(),
            "goal": states[-1, i, :].tolist(),
            "max_jump": 0,
            "max_collision": 0,
            "start_distance": 0,
            "goal_distance": 0,
            "x_bound_distance": 0,
            "u_bound_distance": 0,
            "num_states": 6,
            "num_actions": 5,
            "info": "i:480-s:0-l:5",
            "actions": actions[:, i, :].tolist(),
            "states": states[:, i, :].tolist(),
        }
        mps.append(mp)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as file:
        yaml.safe_dump(mps, file, default_flow_style=None)


def train(config_path: Path, model_path: Path, epochs: int) -> diffmp.torch.Model:
    config = diffmp.torch.Config.from_yaml(config_path)
    model = diffmp.torch.Model(config)
    model.path = model_path
    diffmp.torch.train(model, epochs)
    model.save()
    return model


def load(config_path: Path, model_path: Path) -> diffmp.torch.Model:
    config = diffmp.torch.Config.from_yaml(config_path)
    model = diffmp.torch.Model(config)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model


def export(
    model: diffmp.torch.Model, instance: diffmp.problems.Instance, out_path: Path
) -> None:
    out = diffmp.torch.sample(model, 100, instance).detach().numpy()
    mps = model.dynamics.to_mp(out)
    out_to_mps(mps["actions"], mps["states"], out_path)


if __name__ == "__main__":
    dynamics = "unicycle1_v0"
    # dynamics = "unicycle2_v0"
    config_path = Path(f"scripts/{dynamics}.yaml")
    model_path = Path(f"data/models/{dynamics}.pt")
    if dynamics == "unicycle1_v0":
        instance = diffmp.problems.Instance.from_dict(
            diffmp.utils.load_yaml(Path("../example/bugtrap.yaml"))
        )
    elif dynamics == "unicycle2_v0":
        instance = diffmp.problems.Instance.from_dict(
            diffmp.utils.load_yaml(Path("../example/bugtrap_2.yaml"))
        )
    else:
        raise NotImplementedError

    epochs = 25000
    exports = 10
    # model = train(config_path, model_path, epochs)
    model = load(config_path, model_path)
    for i in range(exports):
        export(model, instance, Path(f"data/output/{dynamics}/{i}.yaml"))
