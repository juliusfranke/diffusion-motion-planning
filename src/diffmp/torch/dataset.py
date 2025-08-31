from typing import Literal, Optional
import torch.utils.data
import numpy.typing as npt
import numpy as np
import diffmp
import torch


class DiffusionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        regular: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        discretized: Optional[torch.Tensor] = None,
        row_to_env: Optional[npt.NDArray[np.floating]] = None,
        row_to_id: Optional[torch.Tensor] = None,
        action_classes: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert conditioning is None or len(regular) == len(conditioning)
        is_discretized = discretized is not None
        if is_discretized:
            assert row_to_env is not None
            assert len(regular) == row_to_env.shape[0]

        self.regular = regular.to(diffmp.utils.DEVICE)
        self.conditioning = (
            conditioning.to(diffmp.utils.DEVICE) if conditioning is not None else None
        )
        self.discretized = (
            discretized.to(diffmp.utils.DEVICE) if is_discretized else None
        )
        self.row_to_env = row_to_env
        self.row_to_id = row_to_id
        self.is_discretized = is_discretized

        self.actions_classes = (
            action_classes if action_classes is None else action_classes.to(torch.long)
        )

    def __getitem__(self, idx) -> dict[str, torch.Tensor | Literal[0]]:
        discretized: Literal[0] | torch.Tensor = 0
        if self.is_discretized:
            env_id = int(self.row_to_env[idx])  # type:ignore
            discretized = self.discretized[env_id]  # type:ignore

        return {
            "regular": self.regular[idx],
            "conditioning": 0 if self.conditioning is None else self.conditioning[idx],
            "discretized": discretized,
            "robot_id": 0 if self.row_to_id is None else self.row_to_id[idx],
            "actions_classes": (
                0 if self.actions_classes is None else self.actions_classes[idx]
            ),
        }

    def __len__(self):
        return len(self.regular)
