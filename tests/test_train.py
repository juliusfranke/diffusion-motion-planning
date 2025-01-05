from pathlib import Path

import diffmp


def test_train():
    dynamics = diffmp.dynamics.get_dynamics("unicycle1_v0")
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
        noise_schedule=diffmp.torch.NoiseSchedule.linear,
    )
    model = diffmp.torch.Model(config)
    diffmp.torch.train(model, 10)
