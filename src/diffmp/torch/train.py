from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, random_split

import diffmp

from .model import Model


def run_epoch(
    model: Model,
    training_loader: DataLoader[Any],
    alpha_bars: npt.NDArray[np.floating],
    validate: bool,
) -> float:
    running_loss = 0.0
    i = 0
    for i, data in enumerate(training_loader):
        regular = data["regular"]
        conditioning = data["conditioning"]
        if not validate:
            model.optimizer.zero_grad()
        t = torch.randint(
            model.config.denoising_steps,
            size=(regular.shape[0],),
            device=diffmp.utils.DEVICE,
        )
        alpha_t = (
            torch.index_select(
                torch.tensor(alpha_bars, device=diffmp.utils.DEVICE), 0, t
            )
            .unsqueeze(1)
            .to(diffmp.utils.DEVICE)
        )

        epsilon = torch.randn(*regular.shape, device=diffmp.utils.DEVICE)

        data_noised = alpha_t**0.5 * regular + epsilon * (1 - alpha_t) ** 0.5

        if conditioning.shape[1] != 0:
            x = torch.concat([data_noised, conditioning, t.unsqueeze(1)], dim=-1)
        else:
            x = torch.concat([data_noised, t.unsqueeze(1)], dim=-1)

        out = model(x)

        loss = model.loss_fn(out, epsilon)

        running_loss += loss.detach().cpu().numpy()

        if not validate:
            loss.backward()
            model.optimizer.step()

    return running_loss / (i + 1)


def train(model: Model, n_epochs: int):
    model.to(diffmp.utils.DEVICE)

    dataset = diffmp.utils.load_dataset(model.config)

    test_abs = int(len(dataset) * model.config.validation_split)
    train_subset, val_subset = random_split(
        dataset, [test_abs, len(dataset) - test_abs]
    )

    training_loader = DataLoader(
        train_subset, batch_size=model.config.batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        val_subset, batch_size=model.config.batch_size, shuffle=True
    )

    alpha_bars, _ = model.noise_schedule(model.config.denoising_steps)
    for reporter in model.config.reporters:
        if hasattr(reporter, "start"):
            reporter.start(n_epochs)
    try:
        for epoch in range(n_epochs):
            # TODO implement reporting
            train_loss = run_epoch(model, training_loader, alpha_bars, validate=False)
            val_loss = run_epoch(model, validation_loader, alpha_bars, validate=True)
            for reporter in model.config.reporters:
                reporter.report_loss(train_loss, val_loss, epoch)
    except KeyboardInterrupt:
        print("Stopped training")
    finally:
        for reporter in model.config.reporters:
            if hasattr(reporter, "close"):
                reporter.close()
