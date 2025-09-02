from __future__ import annotations
from enum import Enum
import numpy as np
from aim import Run
from typing import Dict, Optional
import optuna
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import diffmp


class Reporter:
    def __init__(self) -> None:
        self.started = False

    def start(self, total_steps: int) -> None:
        return None

    def check_start(self):
        if getattr(self, "requires_start", False) and not self.started:
            raise RuntimeError(
                f"{self.__class__.__name__} requires 'start()' to be called before reporting"
            )

    def report_train(self, train_loss: float, step: int, **kwargs) -> None:
        self.check_start()
        return None

    def report_validate(self, validation_loss: float, step: int, **kwargs) -> None:
        self.check_start()
        return None

    def report_test(self, test_loss: float, step: int, **kwargs) -> None:
        self.check_start()
        return None

    def report_hparams(self, config: diffmp.torch.Config) -> None:
        return None

    def close(self) -> None:
        return None


class ConsoleReporter(Reporter):
    def __init__(self) -> None:
        super().__init__()
        self.val = np.inf
        self.train = np.inf

    def report_train(self, train_loss: float, step: int, **kwargs) -> None:
        self.train = train_loss
        self.__print(step)
        return super().report_train(train_loss, step, **kwargs)

    def report_validate(self, validation_loss: float, step: int, **kwargs) -> None:
        self.val = validation_loss
        self.__print(step)
        return super().report_validate(validation_loss, step, **kwargs)

    def __print(self, step: int) -> None:
        print(
            f"Step {step}: Training Loss = {self.train:.4f}, Validation Loss = {self.val:.4f}"
        )


class TQDMReporter(Reporter):
    def __init__(self) -> None:
        super().__init__()
        self.pbar: Optional[tqdm] = None
        self.requires_start = True
        self.postfix = {"train": np.inf, "val": np.inf}
        self.step = 0

    def start(self, total_steps: int) -> None:
        self.pbar = tqdm(total=total_steps, desc="Training", unit="step")
        self.started = True

    def __update_postfix(self, step: int) -> None:
        assert isinstance(self.pbar, tqdm)
        self.pbar.set_postfix(self.postfix)
        while self.step < step + 1:
            self.step += 1
            self.pbar.update()

    def report_train(self, train_loss: float, step: int, **kwargs) -> None:
        super().report_train(train_loss, step, **kwargs)
        self.postfix["train"] = train_loss
        self.__update_postfix(step)
        return None

    def report_validate(self, validation_loss: float, step: int, **kwargs) -> None:
        super().report_validate(validation_loss, step, **kwargs)
        self.postfix["val"] = validation_loss
        self.__update_postfix(step)
        return None

    def report_test(self, test_loss: float, step: int, **kwargs) -> None:
        super().report_test(test_loss, step, **kwargs)
        if self.pbar is not None:
            self.pbar.write(
                f"Epoch {step+1:04d}: Test: {test_loss:.4f} - Train: {self.postfix['train']:.4f} - Val: {self.postfix['val']:.4f}"
            )
        return None

    def close(self) -> None:
        if self.pbar is not None:
            self.pbar.close()


class TensorBoardReporter(Reporter):
    def __init__(self, log_dir="logs"):
        super().__init__()
        self.writer = SummaryWriter(log_dir=log_dir)

    # def report_loss(self, train_loss: float, val_loss: float, step, **kwargs):
    #     super().report_loss(train_loss, val_loss, step, **kwargs)
    #     self.writer.add_scalar("loss/train", train_loss, step)
    #     self.writer.add_scalar("loss/val", val_loss, step)

    def close(self):
        self.writer.close()


class AimReporter(Reporter):
    def __init__(self) -> None:
        super().__init__()
        self.run = Run()
        self.best_train = np.inf
        self.best_val = np.inf
        self.best_test = 0
        self.best_success = -1
        self.best_duration = np.inf
        self.best_cost = np.inf

    def report_hparams(self, config: diffmp.torch.Config) -> None:
        config_dict = config.to_dict()
        config_dict["conditioning"] = {key: True for key in config_dict["conditioning"]}
        config_dict["regular"] = {key: True for key in config_dict["regular"]}
        config_dict["reporters"] = {key: True for key in config_dict["reporters"]}
        weights = {}
        for cols, value in config_dict["weights"].items():
            if cols[0] not in weights.keys():
                weights[cols[0]] = {}
            weights[cols[0]][cols[1]] = value
        config_dict["weights"] = weights
        self.run["hparams"] = config_dict
        return super().report_hparams(config)

    def report_train(self, train_loss: float, step: int, **kwargs) -> None:
        super().report_train(train_loss, step, **kwargs)
        self.run.track(
            train_loss, name="loss", context={"subset": "train"}, epoch=step, step=step
        )
        self.best_train = min(train_loss, self.best_train)
        self.run.track(
            self.best_train,
            name="best_loss",
            context={"subset": "train"},
            epoch=step,
            step=step,
        )
        return None

    def report_validate(self, validation_loss: float, step: int, **kwargs) -> None:
        super().report_validate(validation_loss, step, **kwargs)
        self.run.track(
            validation_loss,
            name="loss",
            context={"subset": "validation"},
            epoch=step,
            step=step,
        )
        self.best_val = min(validation_loss, self.best_val)
        self.run.track(
            self.best_val,
            name="best_loss",
            context={"subset": "validation"},
            epoch=step,
            step=step,
        )
        return None

    def report_test(self, test_loss, step: int, **kwargs) -> None:
        super().report_test(test_loss, step, **kwargs)
        self.run.track(
            test_loss,
            name="performance",
            # context={"subset": "test"},
            epoch=step,
            step=step,
        )
        self.best_test = max(test_loss, self.best_test)
        self.run.track(
            self.best_test,
            name="best_performance",
            # context={"subset": "test"},
            epoch=step,
            step=step,
        )
        return None


class OptunaReporter(Reporter):
    def __init__(self, reported_min: int = 5) -> None:
        self.trial: Optional[optuna.Trial] = None
        self.reported = 0
        self.reported_min = reported_min
        self.best_test = 0

    def report_test(self, test_loss: float, step: int, **kwargs) -> None:
        assert isinstance(self.trial, optuna.Trial)

        self.trial.report(test_loss, step)
        self.best_test = max(test_loss, self.best_test)
        self.reported += 1
        if self.reported > self.reported_min:
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return super().report_test(test_loss, step, **kwargs)


class Reporters(Enum):
    console = ConsoleReporter
    tqdm = TQDMReporter
    tensorboard = TensorBoardReporter
    aim = AimReporter
    optuna = OptunaReporter
