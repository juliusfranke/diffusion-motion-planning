from typing import Optional
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm


class Reporter:
    def __init__(self) -> None:
        self.started = False

    def start(self, total_steps: int) -> None:
        return None

    def report_loss(
        self, train_loss: float, val_loss: float, step: int, **kwargs
    ) -> None:
        if getattr(self, "requires_start", False) and not self.started:
            raise RuntimeError(
                f"{self.__class__.__name__} requires 'start()' to be called before reporting"
            )

    def close(self) -> None:
        return None


class ConsoleReporter(Reporter):
    def report_loss(self, train_loss: float, val_loss: float, step, **kwargs):
        super().report_loss(train_loss, val_loss, step, **kwargs)
        print(
            f"Step {step}: Training Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}"
        )


class TQDMReporter(Reporter):
    def __init__(self) -> None:
        super().__init__()
        self.pbar: Optional[tqdm] = None
        self.requires_start = True

    def start(self, total_steps: int) -> None:
        self.pbar = tqdm(total=total_steps, desc="Training", unit="step")
        self.started = True

    def report_loss(
        self, train_loss: float, val_loss: float, step: int, **kwargs
    ) -> None:
        super().report_loss(train_loss, val_loss, step, **kwargs)
        if self.pbar is not None:
            self.pbar.set_postfix(
                trainloss=f"{train_loss:.4f}", valloss=f"{val_loss:.4f}"
            )
            self.pbar.update(1)
        else:
            raise RuntimeError("Progress bar is not initialized. Call 'start()' first.")

    def close(self) -> None:
        if self.pbar is not None:
            self.pbar.close()


class TensorBoardReporter(Reporter):
    def __init__(self, log_dir="logs"):
        super().__init__()
        self.writer = SummaryWriter(log_dir=log_dir)

    def report_loss(self, train_loss: float, val_loss: float, step, **kwargs):
        super().report_loss(train_loss, val_loss, step, **kwargs)
        self.writer.add_scalar("loss/train", train_loss, step)
        self.writer.add_scalar("loss/val", val_loss, step)

    def close(self):
        self.writer.close()
