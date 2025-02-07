"""Main Training Script."""
from pytorch_template.models import CustomModel
from pytorch_template.data import CustomDataset
import numpy as np
from torch.utils.data import random_split
def main() -> None:
    """Main Training Function."""
    TEST_FRACTION = 0.2


    features = np.random.rand(10,2)
    targets = np.array([0,1,0,1,0,1,1,0,0,0])

    dataset = CustomDataset(features=features, targets=targets)

    size = len(dataset)
    train_size = int((1 - TEST_FRACTION) * size)
    test_size = size - train_size
    train_dataset, test_dataset = random_split(dataset)

    model = CustomModel()

    print("Training Code Goes Here")


if __name__ == "__main__":
    main()
