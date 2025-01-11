from typing import Optional
import torch.utils.data
import diffmp
import torch


class DiffusionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        regular: torch.Tensor,
        conditioning: Optional[torch.Tensor],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert conditioning is None or len(regular) == len(conditioning)

        self.regular: torch.Tensor = regular.to(diffmp.utils.DEVICE)
        self.conditioning: torch.Tensor | None = (
            conditioning.to(diffmp.utils.DEVICE) if conditioning is not None else None
        )

    def __getitem__(self, idx):
        if self.conditioning is not None:
            return {
                "regular": self.regular[idx],
                "conditioning": self.conditioning[idx],
            }
        return {"regular": self.regular[idx], "conditioning": torch.Tensor()}

    def __len__(self):
        return len(self.regular)
