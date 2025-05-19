"""
The program loads the neural network and tests it against the CIFAR-10 test set.
"""

# For testing the model
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

# Shared constants and the model
from network_utils import Program_Constants as constants
from network_utils import Hyperparameters as hyperparams
from network_utils import Utility_Functions as uf
from network_utils import Cifar10_NN


def test(dataloader: torch.utils.data.DataLoader, model: Cifar10_NN, loss_fn) -> float:
    """Tests the data using a dataset provided, producing an accuracy score.

    Args:
        dataloader (torch.utils.data.DataLoader): A dataloader managing the dataset for testing.
        model (Cifar10_NN): The neural network learning model to be tested.
        loss_fn (loss function): The criterion.

    Returns:
        accuracy (float): The accuracy of the model, as a percentage 
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(constants.DEVICE), y.to(constants.DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    accuracy = 100 * correct / size
    uf.print_accuracy(accuracy, test_loss)
    return accuracy


if __name__ == "__main__":

    # Getting the model before installing the data means that we can avoid downloading the data before discovering there is a problem
    try:
        model = uf.load_model(constants.MODEL_SAVE_PATH)
    except FileNotFoundError:
        print(f"No file exists at {constants.MODEL_SAVE_PATH}")
        quit(1)

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(constants.CIFAR10_TRAIN_MEANS,
                         constants.CIFAR10_TRAIN_STDVS)
        ])
    )

    test_dataloader = DataLoader(test_data, batch_size=hyperparams.BATCH_SIZE)

    test(test_dataloader, model, hyperparams.LOSS_FN)
