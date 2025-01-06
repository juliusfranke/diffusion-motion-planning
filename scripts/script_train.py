from pathlib import Path

import diffmp


def train():
    dynamics = diffmp.dynamics.get_dynamics("unicycle1_v0")
    instance = diffmp.problems.Instance.from_dict(
        diffmp.utils.load_yaml(Path("data/instances/alcove_unicycle_single.yaml"))
    )
    config = diffmp.torch.Config(
        dynamics=dynamics,
        timesteps=5,
        problem="bugtrap",
        n_hidden=2,
        s_hidden=128,
        regular=[diffmp.utils.ParameterRegular.actions],
        conditioning=[],
        loss_fn=diffmp.torch.Loss.mse,
        dataset=Path("data/training_datasets/alcove.parquet"),
        denoising_steps=30,
        batch_size=100,
        lr=1e-3,
        dataset_size=1000,
        reporters=[diffmp.utils.TQDMReporter()],
        noise_schedule=diffmp.torch.NoiseSchedule.linear,
    )
    model = diffmp.torch.Model(config)
    model.path = Path("data/models/test.pt")
    diffmp.torch.train(model, 100)
    # model.save()
    a = diffmp.torch.sample(model, 100, instance)
    print(a)


if __name__ == "__main__":
    train()
