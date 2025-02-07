"""Training and Evaluation functions."""

import numpy as np
from torch import DoubleTensor, tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train(
    data: DataLoader,
    model: Module,
    loss_func: _Loss,
    optimizer: Optimizer,
) -> float:
    """Training loop."""
    model.train()
    batch_loss = 0.0
    for _, batch in enumerate(data):
        left, right, labels = batch
        labels = np.array(labels).transpose()
        labels = tensor(labels)
        output = model(left, right)
        output = output.type(DoubleTensor)
        loss = loss_func(output, labels)
        loss_val = loss.detach().item()
        batch_loss += loss_val
        loss.backward()

        optimizer.step()

    return batch_loss


def evaluate(data: DataLoader, model: Module, loss_func: _Loss) -> float:
    """Evaluation Loop."""
    model.eval()
    batch_loss = 0.0
    for _, batch in enumerate(data):
        left, right, labels = batch
        labels = np.array(labels).transpose()
        labels = tensor(labels)
        output = model(left, right)

        loss = loss_func(output, labels)
        loss_val = loss.detach().item()
        batch_loss += loss_val

    return batch_loss
