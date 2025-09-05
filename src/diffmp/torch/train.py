from __future__ import annotations

import multiprocessing as mp
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

import diffmp
import diffmp.problems as pb
import diffmp.utils as du

from . import CompositeConfig, ExponentialMovingAverage, Model, compute_test_loss
from .classifier import compute_log_posterior


def run_epoch(
    model: Model,
    training_loader: DataLoader[Any],
    alpha_bars: npt.NDArray[np.floating],
    validate: bool,
    num_action_classes: int = 3,
    loss_weights: Optional[dict[str, float]] = None,
) -> float:
    loss_weights = loss_weights or {"gaussian": 1.0, "categorical": 1.0}
    running_loss = 0.0
    i = 0
    for i, data in enumerate(training_loader):
        regular = data["regular"]
        conditioning = data["conditioning"]
        discretized = data["discretized"]
        robot_id = data["robot_id"]
        x0_cat = data["actions_classes"]
        interior_mask = x0_cat == 1

        if not validate:
            model.optimizer.zero_grad()

        t = torch.randint(
            model.config.denoising_steps,
            size=(regular.shape[0],),
            device=du.DEVICE,
        )
        alpha_t = (
            torch.index_select(torch.tensor(alpha_bars, device=du.DEVICE), 0, t)
            .unsqueeze(1)
            .to(du.DEVICE)
        )

        epsilon = torch.randn(*regular.shape, device=du.DEVICE)

        x_cont = alpha_t**0.5 * regular + epsilon * (1 - alpha_t) ** 0.5

        if conditioning.shape[1] != 0:
            x_cont = torch.concat([x_cont, conditioning], dim=-1)

        if discretized is not None:
            # TODO this is wrong lol
            scale = torch.ones(regular.shape[0], dtype=torch.int) * 0
        else:
            scale = None

        out_gauss, logits_gauss = model(
            x_cont, t.unsqueeze(1), x0_cat, discretized, scale, robot_id
        )
        # breakpoint()

        if model.config.classify_actions:
            # Mask scalar loss to only include the values that are not at action limits
            out_interior = out_gauss[:, : model.actions_dim][interior_mask]
            eps_interior = epsilon[:, : model.actions_dim][interior_mask]
            # scalars, which are categorized (actions)
            scalar_masked_loss = model.loss_fn(
                out_interior,
                eps_interior,
            )
            # scalars which are not categorized (theta etc.)
            out_misc = out_gauss[:, model.actions_dim :]
            eps_misc = epsilon[:, model.actions_dim :]
            scalar_loss = model.loss_fn(out_misc, eps_misc)
            loss_gaussian = scalar_masked_loss + scalar_loss
            # Compute multimonial branch
            alpha_bar_t = torch.index_select(
                torch.tensor(alpha_bars, device=du.DEVICE), 0, t
            )
            log_x0 = (
                F.one_hot(x0_cat, num_classes=num_action_classes)
                .float()
                .clamp(min=1e-40)
                .log()
            )

            log_alpha = torch.log(alpha_bar_t).unsqueeze(1).unsqueeze(2)
            log_rest = torch.log1p(-alpha_bar_t).unsqueeze(1).unsqueeze(2) - np.log(
                num_action_classes
            )
            log_q_xt = torch.logsumexp(
                torch.stack([log_x0 + log_alpha, log_rest.expand_as(log_x0)], dim=-1),
                dim=-1,
            )

            probs_xt = torch.exp(log_q_xt)
            xt = torch.distributions.Categorical(
                probs=probs_xt.view(-1, num_action_classes)
            ).sample()
            xt = xt.view(x0_cat.shape)
            log_xt = (
                F.one_hot(xt, num_classes=num_action_classes)
                .float()
                .clamp(min=1e-40)
                .log()
            )

            log_hat_x0 = F.log_softmax(logits_gauss, dim=-1)
            log_q_post = compute_log_posterior(
                log_xt, log_x0, t, alpha_bars, num_action_classes
            )
            log_p_post = compute_log_posterior(
                log_xt, log_hat_x0, t, alpha_bars, num_action_classes
            )

            q = torch.exp(log_q_post)
            loss_categorical = torch.sum(q * (log_q_post - log_p_post), dim=-1).mean()

            loss = (
                loss_weights["gaussian"] * loss_gaussian
                + loss_weights["categorical"] * loss_categorical
            )
        else:
            loss = model.loss_fn(out_gauss, epsilon)

        running_loss += float(loss.detach().cpu().numpy())

        if not validate:
            loss.backward()
            model.optimizer.step()

    return running_loss / (i + 1)


def run_test_composite(
    composite_config: CompositeConfig, pbar: tqdm, trials: int = 1, n_instances: int = 1
) -> float:
    tasks = []
    tmp_paths = []
    instances = random.sample(
        composite_config.models[0].config.test_instances, n_instances
    )
    test_results = {instance.name: [] for instance in instances}
    pbar0 = tqdm(desc="Sampling", total=n_instances * trials, leave=False)
    for instance in instances:
        for _ in range(trials):
            mp_paths = []
            for robot_idx in range(composite_config.n_robots):
                tmp_path = tempfile.NamedTemporaryFile(suffix=".msgpack")
                tmp_paths.append(tmp_path)
                du.export_composite(
                    composite_config,
                    instance,
                    Path(tmp_path.name),
                    n_mp=du.DEFAULT_CONFIG["num_primitives_0"],
                    robot_idx=robot_idx,
                )
                mp_paths.append(str(tmp_path.name))
            cfg = du.DEFAULT_CONFIG | {"mp_path": mp_paths}
            if instance.robots[0].dynamics == "car1_v0":
                cfg["delta_0"] = 0.9
            task = du.Task(instance, cfg, 1000, 1500, [])
            tasks.append(task)
            pbar0.update()
    pbar0.close()
    for task in tasks:
        task.instance = task.instance.to_dict()

    pbar1 = tqdm(desc="Testing", total=len(tasks), leave=False)
    executed_tasks = du.execute_tasks(tasks, timeout=10, pbar=pbar1)
    pbar1.close()
    # with mp.Pool(4, maxtasksperchild=10) as p:
    for executed_task in executed_tasks:
        instance = executed_task.instance
        if isinstance(instance, pb.Instance):
            baseline = instance.baseline
            name = instance.name
        else:
            baseline = pb.Baseline.from_dict(instance["baseline"])
            name = instance["name"]
        assert isinstance(baseline, diffmp.problems.Baseline)
        if len(executed_task.solutions) == 0:
            test_results[name].append([0, 0, 0])
            # instance_results[name].append(diffmp.problems.Baseline(0, 0, 0))
            continue
        this_success = 1
        this_duration = executed_task.solutions[0].runtime
        this_cost = min([s.cost for s in executed_task.solutions])
        test_results[name].append([1, this_duration, this_cost])
        # this_cost = executed_task.solutions[0].cost
        # instance_results[name].append(
        # diffmp.problems.Baseline(this_success, this_duration, this_cost)
        # )
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
        name = instance.name
        baseline = instance.baseline
        # assert isinstance(instance.results, list)
        # results = instance_results[instance.name]
        assert isinstance(baseline, pb.Baseline)
        results = np.array(test_results[name])
        try:
            successes = np.sum(results[:, 0])
        except IndexError:
            successes = 0

        debug = True
        if successes == 0:
            loss = compute_test_loss(
                0,
                baseline.success,
                0,
                baseline.duration,
                0,
                baseline.cost,
                pbar=pbar,
                debug=debug,
            )
            test_losses.append(loss)
            continue
        success = successes / trials
        duration = np.sum(results[:, 1]) / successes
        cost = np.sum(results[:, 2]) / successes
        loss = compute_test_loss(
            success,
            baseline.success,
            duration,
            baseline.duration,
            cost,
            baseline.cost,
            pbar=pbar,
            debug=debug,
        )
        test_losses.append(loss)

    for tmp_path in tmp_paths:
        tmp_path.close()
    return float(np.mean(test_losses))


def run_test(
    model: Model, trials: int = 1, n_instances: int = 1, pbar: Optional[tqdm] = None
) -> float:
    tasks = []
    tmp_paths = []
    instances = random.sample(model.config.test_instances, n_instances)
    test_results = {instance.name: [] for instance in instances}
    pbar0 = tqdm(desc="Sampling", total=n_instances * trials, leave=False)
    for instance in instances:
        for _ in range(trials):
            mp_paths = []
            for robot_idx in range(model.config.n_robots):
                # tmp_path = tempfile.NamedTemporaryFile(suffix=".yaml")
                tmp_path = tempfile.NamedTemporaryFile(suffix=".msgpack")
                tmp_paths.append(tmp_path)
                du.export(
                    model,
                    instance,
                    Path(tmp_path.name),
                    n_mp=du.DEFAULT_CONFIG["num_primitives_0"] * 10,
                    robot_idx=robot_idx,
                )
                mp_paths.append(str(tmp_path.name))
            cfg = du.DEFAULT_CONFIG | {"mp_path": mp_paths}
            if instance.robots[0].dynamics == "car1_v0":
                cfg["delta_0"] = 0.9
            # task = du.Task(instance, cfg, 1000, 1500, [])
            task = du.Task(instance, cfg, 5000, 5000, [])
            tasks.append(task)
            pbar0.update()
    pbar0.close()
    for task in tasks:
        task.instance = task.instance.to_dict()
    pbar1 = tqdm(desc="Testing", total=len(tasks), leave=False)
    executed_tasks = du.execute_tasks(tasks, timeout=10, pbar=pbar1)
    # with mp.Pool(4, maxtasksperchild=10) as p:
    pbar1.close()
    # for executed_task in p.imap_unordered(du.execute_task, tasks):
    for executed_task in executed_tasks:
        instance = executed_task.instance
        if isinstance(instance, pb.Instance):
            baseline = instance.baseline
            name = instance.name
        else:
            baseline = pb.Baseline.from_dict(instance["baseline"])
            name = instance["name"]
        assert isinstance(baseline, diffmp.problems.Baseline)
        if len(executed_task.solutions) == 0:
            test_results[name].append([0, 0, 0])
            # test_loss = compute_test_loss(
            #     0, baseline.success, 0, baseline.duration, 0, baseline.cost
            # )
            # test_losses.append(test_loss)
            continue
        this_success = 1
        this_duration = executed_task.solutions[0].runtime
        this_cost = min([s.cost for s in executed_task.solutions])
        # breakpoint()
        test_results[name].append([1, this_duration, this_cost])
        # print(this_duration, baseline.duration, this_cost, baseline.cost)
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
        name = instance.name
        baseline = instance.baseline
        assert isinstance(baseline, pb.Baseline)
        results = np.array(test_results[name])
        successes = np.sum(results[:, 0])
        debug = True
        if successes == 0:
            loss = compute_test_loss(
                0,
                baseline.success,
                0,
                baseline.duration,
                0,
                baseline.cost,
                pbar=pbar,
                debug=debug,
            )
            test_losses.append(loss)
            continue
        success = successes / trials
        duration = np.sum(results[:, 1]) / successes
        cost = np.sum(results[:, 2]) / successes
        loss = compute_test_loss(
            success,
            baseline.success,
            duration,
            baseline.duration,
            cost,
            baseline.cost,
            pbar=pbar,
            debug=debug,
        )
        test_losses.append(loss)

    for tmp_path in tmp_paths:
        tmp_path.close()
    return float(np.mean(test_losses))


def get_data_loader(model: Model) -> Tuple[DataLoader, DataLoader]:
    dataset = du.load_dataset(model.config)
    # breakpoint()
    model.config.set_norm_vals(dataset)
    dataset.regular = model.config.normalize_regular(dataset.regular)

    if dataset.conditioning is not None:
        dataset.conditioning = model.config.normalize_conditioning(dataset.conditioning)

    if dataset.discretized is not None:
        dataset.discretized = model.config.normalize_discretized(dataset.discretized)

    test_abs = int(len(dataset) * model.config.validation_split)
    if dataset.row_to_env is None:
        train_subset, val_subset = random_split(
            dataset, [test_abs, len(dataset) - test_abs]
        )
    else:
        unique, counts = np.unique(dataset.row_to_env, return_counts=True)
        total = counts.sum()

        # Sort by frequency (descending)
        sorted_idx = np.argsort(-counts)
        unique_sorted = unique[sorted_idx]
        counts_sorted = counts[sorted_idx]

        # Greedily accumulate until target reached
        validation_env_idx = []
        training_env_idx = []
        cumulative = 0
        for u, c in zip(unique_sorted, counts_sorted):
            if cumulative / total >= (1 - model.config.validation_split):
                training_env_idx.append(u)
                continue
            validation_env_idx.append(u)
            cumulative += c

        coverage = cumulative / total
        training_idx: list[int] = np.where(
            np.isin(dataset.row_to_env, training_env_idx)
        )[0].tolist()
        validation_idx: list[int] = np.where(
            np.isin(dataset.row_to_env, validation_env_idx)
        )[0].tolist()

        train_subset = Subset(dataset, training_idx)

        val_subset = Subset(dataset, validation_idx)

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
    epoch_test = 20
    best_test = 0
    ema_test_loss = ExponentialMovingAverage(alpha=0.2)
    pbar = tqdm(desc="Training", total=n_epochs)
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
            test_loss = run_test_composite(
                composite_config, pbar=pbar, trials=3, n_instances=3
            )
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
        if not isinstance(composite_config.optuna, du.OptunaReporter):
            continue
        composite_config.optuna.report_test(ema_test_loss.ema, epoch)


def train(
    model: Model,
    n_epochs: int,
    run_validation: bool = True,
    benchmark_test: bool = True,
    training_loader: Optional[DataLoader] = None,
    validation_loader: Optional[DataLoader] = None,
    test_epoch: int = 100,
) -> None:
    val_loss = np.inf
    best_val_loss = np.inf
    model.to(du.DEVICE)
    best_test = 0
    ema_test_loss = ExponentialMovingAverage(alpha=0.2)

    if not training_loader and not validation_loader:
        training_loader, validation_loader = get_data_loader(model)
    assert isinstance(training_loader, DataLoader)
    assert isinstance(validation_loader, DataLoader)

    alpha_bars, _ = model.noise_schedule(model.config.denoising_steps)
    pbar = None
    for reporter in model.config.reporters:
        reporter.report_hparams(model.config)
        if hasattr(reporter, "start"):
            reporter.start(n_epochs)
        if isinstance(reporter, du.TQDMReporter):
            pbar = reporter.pbar
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
            if (epoch + 1) % test_epoch != 0:
                continue
            test_loss = run_test(model, pbar=pbar, trials=4, n_instances=8)
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
