from functools import partial
import math
import tempfile
import multiprocessing as mp
import gc
from typing import Any, Dict, Optional, Tuple
import random

from aim.sdk.run import defaultdict
from pandas.io.sql import com
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, random_split

import diffmp
from pathlib import Path


from . import CompositeConfig, Model, compute_test_loss, ExponentialMovingAverage


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


def run_test_composite(
    composite_config: CompositeConfig, trials: int = 1, n_instances: int = 1
) -> float:
    # test_losses = []
    instance_results = defaultdict(list)
    tasks = []
    tmp_paths = []
    instances = random.sample(
        composite_config.models[0].config.test_instances, n_instances
    )
    for instance in instances:
        for _ in range(trials):
            tmp_path = tempfile.NamedTemporaryFile(suffix=".yaml")
            tmp_paths.append(tmp_path)
            diffmp.utils.export_composite(
                composite_config,
                instance,
                Path(tmp_path.name),
                n_mp=diffmp.utils.DEFAULT_CONFIG["num_primitives_0"],
            )
            cfg = diffmp.utils.DEFAULT_CONFIG | {"mp_path": str(tmp_path.name)}
            if instance.robots[0].dynamics == "car1_v0":
                cfg["delta_0"] = 0.9
            task = diffmp.utils.Task(instance, cfg, 1000, 1500, [])
            tasks.append(task)
    with mp.Pool(4, maxtasksperchild=10) as p:
        for executed_task in p.imap_unordered(diffmp.utils.execute_task, tasks):
            instance = executed_task.instance
            baseline = instance.baseline
            assert isinstance(baseline, diffmp.problems.Baseline)
            # assert isinstance(instance.results, list)
            if len(executed_task.solutions) == 0:

                instance_results[instance.name].append(
                    diffmp.problems.Baseline(0, 0, 0)
                )
                # test_loss = compute_test_loss(
                #     0, baseline.success, 0, baseline.duration, 0, baseline.cost
                # )
                # test_losses.append(test_loss)
                continue
            this_success = 1
            this_duration = executed_task.solutions[0].runtime
            this_cost = min([s.cost for s in executed_task.solutions])
            # this_cost = executed_task.solutions[0].cost
            instance_results[instance.name].append(
                diffmp.problems.Baseline(this_success, this_duration, this_cost)
            )
            # test_loss = compute_test_loss(
            #     this_success,
            #     baseline.success,
            #     this_duration,
            #     baseline.duration,
            #     this_cost,
            #     baseline.cost,
            # )
            # test_losses.append(test_loss)
    test_losses = []
    for instance in instances:
        # assert isinstance(instance.results, list)
        results = instance_results[instance.name]
        assert isinstance(instance.baseline, diffmp.problems.Baseline)
        successes = []
        durations = []
        costs = []
        for result in results:
            successes.append(result.success)
            if result.success:
                durations.append(result.duration)
                costs.append(result.cost)
        st = np.mean(successes)
        if len(durations) == 0:
            test_losses.append(0)
            continue
        dt = np.median(durations)
        ct = np.median(costs)
        test_loss = compute_test_loss(
            st,
            instance.baseline.success,
            dt,
            instance.baseline.duration,
            ct,
            instance.baseline.cost,
        )
        test_losses.append(test_loss)

    for tmp_path in tmp_paths:
        tmp_path.close()
    return float(np.median(test_losses))


def run_test(model: Model, trials: int = 1, n_instances: int = 1) -> float:
    test_losses = []
    tasks = []
    tmp_paths = []
    instances = random.sample(model.config.test_instances, n_instances)
    for instance in instances:
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
            if instance.robots[0].dynamics == "car1_v0":
                cfg["delta_0"] = 0.9
            task = diffmp.utils.Task(instance, cfg, 1000, 1500, [])
            tasks.append(task)
    with mp.Pool(4, maxtasksperchild=10) as p:
        for executed_task in p.imap_unordered(diffmp.utils.execute_task, tasks):
            baseline = executed_task.instance.baseline
            assert isinstance(baseline, diffmp.problems.Baseline)
            if len(executed_task.solutions) == 0:
                test_loss = compute_test_loss(
                    0, baseline.success, 0, baseline.duration, 0, baseline.cost
                )
                test_losses.append(test_loss)
                continue
            this_success = 1
            this_duration = executed_task.solutions[0].runtime
            this_cost = min([s.cost for s in executed_task.solutions])
            test_loss = compute_test_loss(
                this_success,
                baseline.success,
                this_duration,
                baseline.duration,
                this_cost,
                baseline.cost,
            )
            test_losses.append(test_loss)

    for tmp_path in tmp_paths:
        tmp_path.close()
    return float(np.median(test_losses))


def get_data_loader(model: Model) -> Tuple[DataLoader, DataLoader]:
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
    return (training_loader, validation_loader)


def train_composite(
    composite_config: CompositeConfig, n_epochs, auto_save: bool = True
):
    epoch_test = 10
    best_test = 0
    ema_test_loss = ExponentialMovingAverage(alpha=0.2)
    pbar = tqdm(total=n_epochs)
    data_loaders = {
        model.dynamics.timesteps: get_data_loader(model)
        for model in composite_config.models
    }
    for i in range(n_epochs // epoch_test):
        epoch = i * epoch_test + epoch_test
        for model in composite_config.models:
            training_loader, validation_loader = data_loaders[model.dynamics.timesteps]
            train(model, epoch_test, False, False, training_loader, validation_loader)
            # train_single(model)
        # with mp.Pool(4) as p:
        #     for _ in p.imap_unordered(train_single, composite_config.models):
        #         pass
        pbar.update(epoch_test // 2)
        try:
            test_loss = run_test_composite(composite_config, trials=1, n_instances=6)
        except AssertionError:
            test_loss = 0.0
        ema_test_loss.update(test_loss)
        assert isinstance(ema_test_loss.ema, float) or isinstance(
            ema_test_loss.ema, int
        )
        if ema_test_loss.ema > best_test:
            best_test = ema_test_loss.ema
            if auto_save:
                [m.save() for m in composite_config.models]
        pbar.update(epoch_test // 2)
        pbar.write(f"{epoch}: {ema_test_loss.ema} ({best_test})")
        if not isinstance(composite_config.optuna, diffmp.utils.OptunaReporter):
            continue
        composite_config.optuna.report_test(ema_test_loss.ema, epoch)


def train(
    model: Model,
    n_epochs: int,
    run_validation: bool = True,
    benchmark_test: bool = True,
    training_loader: Optional[DataLoader] = None,
    validation_loader: Optional[DataLoader] = None,
) -> None:
    val_loss = np.inf
    best_val_loss = np.inf
    model.to(diffmp.utils.DEVICE)
    best_test = 0
    ema_test_loss = ExponentialMovingAverage(alpha=0.2)

    if not training_loader and not validation_loader:
        training_loader, validation_loader = get_data_loader(model)
    assert isinstance(training_loader, DataLoader)
    assert isinstance(validation_loader, DataLoader)

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
            if run_validation:
                if (epoch + 1) % 10 == 0:
                    val_loss = run_epoch(
                        model, validation_loader, alpha_bars, validate=True
                    )
                    for reporter in model.config.reporters:
                        reporter.report_validate(val_loss, epoch)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            if not benchmark_test:
                continue
            if epoch < 1:
                continue
            if (epoch + 1) % 10 != 0:
                continue
            test_loss = run_test(model, trials=3, n_instances=2)
            ema_test_loss.update(test_loss)

            assert isinstance(ema_test_loss.ema, float)
            if ema_test_loss.ema > best_test:
                best_test = ema_test_loss.ema
                model.save()
            for reporter in model.config.reporters:
                reporter.report_test(ema_test_loss.ema, epoch)

    except KeyboardInterrupt:
        print("Stopped training")
    finally:
        for reporter in model.config.reporters:
            if hasattr(reporter, "close"):
                reporter.close()
    return None
