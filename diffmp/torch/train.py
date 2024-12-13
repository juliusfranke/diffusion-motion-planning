from torch.utils.data import random_split
import diffmp
from . import Model


def train(model: Model):
    dataset = diffmp.utils.load_data(model.config.dataset)
    dataset.tensors[0].to(diffmp.utils.DEVICE)

    test_abs = (len(dataset)*model.config.validation_split)
    train_subset, val_subset = random_split(dataset, [test_abs, len(dataset)-test_abs])

    # for epoch in 
