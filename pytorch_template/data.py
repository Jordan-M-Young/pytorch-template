"""Dataset Class, helper functions."""

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Custom Dataset Class."""

    def __init__(self, features, targets):
        """Initialize Custom Dataset."""
        self.features = (features,)
        self.targets = targets

    def __getitem__(self, index):
        """Get item from dataset."""
        sample = self.features[index]
        target = self.targets[index]

        return sample, target

    def __len__(self) -> int:
        """Get length of dataset."""
        return len(self.features)
