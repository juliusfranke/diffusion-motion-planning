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

    def report_test(self, test_results: Dict, step: int, **kwargs) -> None:
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
        self.postfix = {"training": np.inf, "validation": np.inf}
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
        self.postfix["training"] = train_loss
        self.__update_postfix(step)
        return None

    def report_validate(self, validation_loss: float, step: int, **kwargs) -> None:
        super().report_validate(validation_loss, step, **kwargs)
        self.postfix["validation"] = validation_loss
        self.__update_postfix(step)
        return None

    def report_test(self, test_results: Dict[str, float], step: int, **kwargs) -> None:
        super().report_test(test_results, step, **kwargs)
        if self.pbar is not None:
            self.pbar.write(str(test_results))
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
        self.best_test = np.inf
        self.best_success = -1
        self.best_duration = np.inf
        self.best_cost = np.inf

    def report_hparams(self, config: diffmp.torch.Config) -> None:
        config_dict = config.to_dict()
        config_dict["conditioning"] = {key: True for key in config_dict["conditioning"]}
        config_dict["regular"] = {key: True for key in config_dict["regular"]}
        config_dict["reporters"] = {key: True for key in config_dict["reporters"]}
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

    def report_test(self, test_results: Dict, step: int, **kwargs) -> None:
        super().report_test(test_results, step, **kwargs)
        self.run.track(test_results, context={"subset": "test"}, epoch=step, step=step)
        success = test_results["success"]
        duration = test_results["duration"]
        cost = test_results["cost"]
        self.best_success = max(success, self.best_success)
        self.best_duration = min(duration, self.best_duration)
        self.best_cost = min(cost, self.best_cost)
        self.run.track(
            {
                "best_success": self.best_success,
                "best_duration": self.best_duration,
                "best_cost": self.best_cost,
            },
            context={"subset": "test"},
            epoch=step,
            step=step,
        )
        # test_loss = (duration * cost) / ((success * 1000) ** 2)
        test_loss = (-success) + cost + duration
        self.best_test = min(test_loss, self.best_test)
        self.run.track(
            test_loss, name="loss", context={"subset": "test"}, epoch=step, step=step
        )
        self.run.track(
            self.best_test,
            name="best_loss",
            context={"subset": "test"},
            epoch=step,
            step=step,
        )
        return None

    # def report_loss(
    #     self, train_loss: float, val_loss: float, step: int, **kwargs
    # ) -> None:
    #     self.run.track(
    #         train_loss,
    #     )
    #     return super().report_loss(train_loss, val_loss, step, **kwargs)

    # def report_test(self, test_results: Dict[str, float]) -> None:
    #     return super().report_test(test_results)


class OptunaReporter(Reporter):
    def __init__(self) -> None:
        self.trial: Optional[optuna.Trial] = None

    def report_test(self, test_results: Dict, step: int, **kwargs) -> None:
        assert isinstance(self.trial, optuna.Trial)
        success = test_results["success"]
        cost = test_results["cost"]
        duration = test_results["duration"]
        test_loss = (-success) + cost + duration
        self.trial.report(test_loss, step)
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return super().report_test(test_results, step, **kwargs)


class Reporters(Enum):
    console = ConsoleReporter
    tqdm = TQDMReporter
    tensorboard = TensorBoardReporter
    aim = AimReporter
    optuna = OptunaReporter
