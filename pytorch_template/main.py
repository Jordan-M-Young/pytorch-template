"""Main Training Script."""

from pytorch_template.models import CustomModel
from pytorch_template.train import train, evaluate
from pytorch_template.data import CustomDataset
from pytorch_template.utils import log_epoch
from pytorch_template.config import get_config
import numpy as np
from torch.optim import Adam
from torch.nn import BCELoss
from torch.utils.data import random_split, DataLoader


def main() -> None:
    """Main Training Function."""

    # config
    config = get_config()
    BATCH_SIZE = config["batch_size"]
    TEST_FRACTION = config["test_size"]
    EPOCHS = config["epochs"]

    # load features and targets, change according to your needs.
    features = np.random.rand(10, 2)
    targets = np.array([0, 1, 0, 1, 0, 1, 1, 0, 0, 0])

    # load training and test datasets.
    dataset = CustomDataset(features=features, targets=targets)
    size = len(dataset)
    train_size = int((1 - TEST_FRACTION) * size)
    test_size = size - train_size
    train_dataset, test_dataset = random_split(dataset)

    # initialize dataloaders.
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # initialize model, optimizer, and loss function.
    model = CustomModel()
    optimizer = Adam(params=model.parameters(), lr=0.0001)
    loss_fn = BCELoss()

    # main training loop.
    for epoch in range(EPOCHS):
        train_loss = train(train_dataloader, model, loss_fn, optimizer)
        evaluate_loss = evaluate(test_dataloader, model, loss_fn)
        log_epoch(epoch, train_loss, evaluate_loss)


if __name__ == "__main__":
    main()
