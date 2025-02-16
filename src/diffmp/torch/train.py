import math
import tempfile
import multiprocessing as mp
import gc
from typing import Any, Dict

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, random_split

import diffmp
from pathlib import Path

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

        running_loss += float(loss.detach().cpu().numpy())

        if not validate:
            loss.backward()
            model.optimizer.step()

    return running_loss / (i + 1)


def run_test(model: Model, trials: int = 1) -> Dict[str, float]:
    r_success = []
    r_cost = []
    r_duration = []
    tasks = []
    tmp_paths = []
    for instance in model.config.test_instances:
        for _ in range(trials):
            tmp_path = tempfile.NamedTemporaryFile(suffix=".yaml")
            tmp_paths.append(tmp_path)
            diffmp.utils.export(
                model,
                instance,
                Path(tmp_path.name),
                n_mp=diffmp.utils.DEFAULT_CONFIG["num_primitives_0"],
            )
            cfg = diffmp.utils.DEFAULT_CONFIG | {"mp_path": str(tmp_path.name)}
            task = diffmp.utils.Task(instance, cfg, 1000, 1500, [])
            tasks.append(task)
    with mp.Pool(4, maxtasksperchild=10) as p:
        for executed_task in p.imap_unordered(diffmp.utils.execute_task, tasks):
            baseline = executed_task.instance.baseline
            assert isinstance(baseline, diffmp.problems.Baseline)
            if len(executed_task.solutions) == 0:
                r_success.append((0 - baseline.success) / baseline.success)
                continue
            r_success.append((1 - baseline.success) / baseline.success)
            r_duration.append(
                (executed_task.solutions[0].runtime - baseline.duration)
                / baseline.duration
            )
            r_cost.append(
                (min([s.cost for s in executed_task.solutions]) - baseline.cost)
                / baseline.cost
            )

    for tmp_path in tmp_paths:
        tmp_path.close()
    if r_duration:
        duration = float(np.median(r_duration))
        cost = float(np.median(r_cost))
    else:
        duration = 1
        cost = 1
    return {
        "success": float(np.mean(r_success)),
        "duration": duration,
        "cost": cost,
    }


def train(model: Model, n_epochs: int):
    val_loss = np.inf
    best_val_loss = np.inf
    model.to(diffmp.utils.DEVICE)
    best_success = 0.0
    best_cost = np.inf
    best_duration = np.inf
    best_test = np.inf

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
        reporter.report_hparams(model.config)
        if hasattr(reporter, "start"):
            reporter.start(n_epochs)
    try:
        for epoch in range(n_epochs):
            train_loss = run_epoch(model, training_loader, alpha_bars, validate=False)
            for reporter in model.config.reporters:
                reporter.report_train(train_loss, epoch)
            if (epoch + 1) % 10 == 0:
                val_loss = run_epoch(
                    model, validation_loader, alpha_bars, validate=True
                )
                for reporter in model.config.reporters:
                    reporter.report_validate(val_loss, epoch)
            # if epoch < 200 or (epoch + 1) % 100 != 0:
            #     continue
            # continue
            if epoch < 200:
                continue
            if val_loss > best_val_loss:
                continue
            ratio = val_loss / best_val_loss
            best_val_loss = val_loss
            if ratio > 0.99:
                continue
            # model.save()
            # continue
            test_results = run_test(model, trials=5)

            for reporter in model.config.reporters:
                reporter.report_test(test_results, epoch)
            success = test_results["success"]
            duration = test_results["duration"]
            cost = test_results["cost"]
            test_loss = (-success) + cost + duration
            if test_loss > best_test:
                continue
            # if success < best_success:
            #     continue
            # if duration * cost > best_duration * best_cost:
            #     continue
            # best_success = success
            # best_duration = duration
            # best_cost = cost
            best_test = test_loss
            model.save()

    except KeyboardInterrupt:
        print("Stopped training")
    finally:
        for reporter in model.config.reporters:
            if hasattr(reporter, "close"):
                reporter.close()
    return best_test
