from pathlib import Path
import sys

import numpy.typing as npt
import torch
import yaml

import diffmp


def train(config_path: Path, model_path: Path, epochs: int) -> diffmp.torch.Model:
    config = diffmp.torch.Config.from_yaml(config_path)
    model = diffmp.torch.Model(config)
    model.path = model_path
    diffmp.torch.train(model, epochs)
    return model


def load(config_path: Path, model_path: Path) -> diffmp.torch.Model:
    config = diffmp.torch.Config.from_yaml(config_path)
    model = diffmp.torch.Model(config)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model


# def export(
#     model: diffmp.torch.Model, instance: diffmp.problems.Instance, out_path: Path
# ) -> None:
#     out = diffmp.torch.sample(model, 100, instance).detach().cpu().numpy()
#     mps = model.dynamics.to_mp(out)
#     out_to_mps(mps["actions"], mps["states"], out_path)


if __name__ == "__main__":
    # dynamics = "unicycle1_v0"
    # dynamics = "unicycle2_v0"
    config_path = Path(sys.argv[1])
    # model_name = sys.argv[2]
    # model_path = Path(f"data/models/{model_name}")
    epochs = 500
    if config_path.suffixes[0] == ".standard":
        config = diffmp.torch.Config.from_yaml(config_path)
        dynamics = config.dynamics.name
        model = diffmp.torch.Model(config)
        # model.path = model_path

        diffmp.torch.train(model, epochs)
    elif config_path.suffixes[0] == ".composite":
        composite_config = diffmp.torch.CompositeConfig.from_yaml(config_path)
        diffmp.torch.train_composite(composite_config, 1000)

    # diffmp.torch.Model.load(Path("data/models/yeehaaa"))
    # breakpoint()
    # if dynamics == "unicycle1_v0":
    #     instance = diffmp.problems.Instance.from_dict(
    #         diffmp.utils.load_yaml(Path("../example/bugtrap.yaml"))
    #     )
    # elif dynamics == "unicycle2_v0":
    #     instance = diffmp.problems.Instance.from_dict(
    #         diffmp.utils.load_yaml(Path("../example/bugtrap_2.yaml"))
    #     )
    # else:
    #     raise NotImplementedError

    # for i in range(exports):
    #     diffmp.utils.export(
    #         model, instance, Path(f"data/output/{dynamics}/rand_{i}.yaml")
    # )
