from pathlib import Path

import diffmp


def test_train():
    dynamics = diffmp.dynamics.get_dynamics("unicycle1_v0")
    config = diffmp.torch.Config(
        dynamics=dynamics,
        timesteps=5,
        problem="bugtrap",
        n_hidden=2,
        s_hidden=2,
        regular=[diffmp.utils.ParameterRegular.actions],
        loss_fn=diffmp.torch.Loss.mse,
        dataset=Path("data/training_datasets/bugtrap_single_l5.yaml"),
        denoising_steps=30,
        batch_size=100,
        lr=1e-3,
        noise_schedule=diffmp.torch.NoiseSchedule.linear,
    )
    model = diffmp.torch.Model(config)
    diffmp.torch.train(model, 100)
