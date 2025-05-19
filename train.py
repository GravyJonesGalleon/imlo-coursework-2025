"""
The program trains the neural network and saves the trained model to a file.
"""

# For training the Neural Network
import torch
from torch import nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

# Shared constants, the model, and utility functions
from network_utils import Program_Constants as constants
from network_utils import Hyperparameters as hyperparams
from network_utils import Utility_Functions as uf
from network_utils import Cifar10_NN

# For visualising training progress
from datetime import datetime
from matplotlib import pyplot as plt


def train(dataloader: torch.utils.data.DataLoader, model: Cifar10_NN, loss_fn, optimiser: torch.optim.Optimizer) -> None:
    """Run a training loop over the entire dataset provided by the dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): A dataloader managing the dataset for training.
        model (Cifar10_NN): The neural network learning model to be trained.
        loss_fn (loss function): The criterion for the optimiser to minimise.
        optimiser (torch.optim.Optimizer): The optimisation function to use.
    """
    size = len(dataloader.dataset)
    numbatches = size // hyperparams.BATCH_SIZE
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(constants.DEVICE), y.to(constants.DEVICE)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        if batch % (numbatches // 10) == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(
                f"Loss: {loss:7f} [{current:>5d}/{size:>5d},{(100*current/size):3.0f}%]")


def validate(dataloader: torch.utils.data.DataLoader, model: Cifar10_NN, loss_fn) -> float:
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
    all_training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=v2.Compose([
            v2.ToImage(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(30),
            v2.RandomAdjustSharpness(0.5, p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(constants.CIFAR10_TRAIN_MEANS,
                         constants.CIFAR10_TRAIN_STDVS)
        ])
    )
    training_data, validation_data = random_split(
        all_training_data, hyperparams.VALIDATION_RATIO)

    # Create dataloaders from the datasets
    train_dataloader = DataLoader(
        training_data, batch_size=hyperparams.BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(
        validation_data, batch_size=hyperparams.BATCH_SIZE, shuffle=True)

    # Instantiate the model
    model = Cifar10_NN().to(constants.DEVICE)

    # Instantiate the loss functions and optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(
        model.parameters(), lr=hyperparams.LEARNING_RATE, weight_decay=hyperparams.WEIGHT_DECAY)

    # Accuracies are saved as histories for progess viewing purposes
    # Start time is logged to see the time the training has taken
    accuracies = [0]
    start_time = datetime.now()
    # Perform the training
    for t in range(hyperparams.NUM_EPOCHS):
        uf.print_training_progress(t, hyperparams.NUM_EPOCHS)
        epoch_start_time = datetime.now()

        train(train_dataloader, model, loss_fn, optimiser)
        accuracy = validate(validation_dataloader, model, loss_fn)
        accuracies.append(accuracy)

        epoch_end_time = datetime.now()
        uf.print_epoch_summary(start_time,
                               epoch_start_time,
                               epoch_end_time,
                               accuracies[-2],
                               accuracies[-1])

    end_time = datetime.now()
    print(f"Done in {end_time - start_time}!")

    uf.save_model_weights(model, constants.MODEL_SAVE_PATH, accuracies[-1])

    # Plot results
    plt.scatter(range(1, hyperparams.NUM_EPOCHS+1), accuracies[1:])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy / %")
    plt.ylim(0, 100)
    plt.show()
