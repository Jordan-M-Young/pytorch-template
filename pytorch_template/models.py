"""Model classes and helper functions."""

from torch import Tensor, save
from torch.nn import Dropout, Linear, Module, ReLU, Sequential, Sigmoid


class CustomModel(Module):
    """Trivial CustomModel Template."""

    def __init__(self):
        """Initialize CustomModel."""
        super(CustomModel, self).__init__()

        self.fc1 = Linear(2, 5)
        self.re1 = ReLU()
        self.d = Dropout(0.2)
        self.fc2 = Linear(5, 1)
        self.sigmoid = Sigmoid()

        self.net = Sequential(self.fc1, self.re1, self.d, self.fc2, self.sigmoid)

    def forward(self, x) -> Tensor:
        """Forward pass function."""
        output = self.net(x)
        return output


def save_model(model: Module, path: str):
    """Save Model To File."""
    save(model, path)
