from typing import Any
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import diffmp

from .model import Model


def run_epoch(
    model: Model,
    training_loader: DataLoader[Any],
    alpha_bars: npt.NDArray[np.floating],
    validate: bool,
):
    running_loss = 0.0

    for i, [data, conditioning] in enumerate(training_loader):
        if not validate:
            model.optimizer.zero_grad()
        t = torch.randint(model.config.denoising_steps, size=data.shape[0])
        alpha_t = (
            torch.index_select(torch.Tensor(alpha_bars), 0, t)
            .unsqueeze(1)
            .to(diffmp.utils.DEVICE)
        )

        epsilon = torch.randn(*data.shape, device=diffmp.utils.DEVICE)

        data_noised = alpha_t**0.5 * data + epsilon * (1 - alpha_t) ** 0.5

        x = torch.concat([data_noised, conditioning, t], dim=-1)

        out = model(x)

        loss = model.loss_fn(out, epsilon)

        running_loss += loss.detach().cpu().numpy()

        if not validate:
            loss.backward()
            model.optimizer.step()

    # TODO implement reporting


def train(model: Model, n_epochs: int):
    dataset = diffmp.utils.load_dataset(model.config)
    dataset.tensors[0].to(diffmp.utils.DEVICE)

    test_abs = len(dataset) * model.config.validation_split
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
    pbar = tqdm(range(n_epochs))
    try:
        for epoch in pbar:
            run_epoch(model, training_loader, alpha_bars, validate=False)
            run_epoch(model, validation_loader, alpha_bars, validate=True)
    except KeyboardInterrupt:
        pbar.close()
        print("Stopped training")
