from pathlib import Path

import diffmp


def train():
    instance = diffmp.problems.Instance.from_dict(
        diffmp.utils.load_yaml(Path("data/instances/alcove_unicycle_single.yaml"))
    )
    config = diffmp.torch.Config.from_yaml(Path("scripts/conf.yaml"))
    model = diffmp.torch.Model(config)
    model.path = Path("data/models/test.pt")
    diffmp.torch.train(model, 100)
    model.save()
    a = diffmp.torch.sample(model, 100, instance)
    print(a)


if __name__ == "__main__":
    train()
