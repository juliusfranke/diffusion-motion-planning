from pathlib import Path

import torch

import diffmp

model_path = Path("data/models/test.pt")


def train():
    config = diffmp.torch.Config.from_yaml(Path("scripts/conf.yaml"))
    model = diffmp.torch.Model(config)
    model.path = model_path
    diffmp.torch.train(model, 100)
    model.save()
    # model.load_state_dict()
    # print(a)


def load():
    instance = diffmp.problems.Instance.from_dict(
        diffmp.utils.load_yaml(Path("data/instances/alcove_unicycle_single.yaml"))
    )
    config = diffmp.torch.Config.from_yaml(Path("scripts/conf.yaml"))
    model = diffmp.torch.Model(config)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    # model = torch.load(model_path, weights_only=False)
    a = diffmp.torch.sample(model, 100, instance)
    print(a)


if __name__ == "__main__":
    train()
    load()
