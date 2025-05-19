"""
The network_utils file contains universal constants, hyperparameters for training, and the neural network class definition.
"""

import torch
from torch import nn
from datetime import datetime


class Program_Constants():
    """
    A class containing constants relevant to the operation of the programs.
    These constants do not need to be altered except for personal preferences, such as changing the model save file path.
    """

    # We want to use the CIFAR-10 dataset to train the data.
    # To normalise this data, we need to know the means and stdevs of the dataset.
    # These have been generated using the program find_mean_and_std.py
    CIFAR10_TRAIN_MEANS = (0.4914008203125,
                           0.482158984375,
                           0.4465309375)
    """The mean value for each colour channel (Red, Green, Blue) in the CIFAR-10 train dataset"""
    CIFAR10_TRAIN_STDVS = (0.2470276019799247,
                           0.24348346588930533,
                           0.2615877093211206)
    """The standard deviation of each colour channel (Red, Green, Blue) in the CIFAR-10 train dataset"""

    DEVICE = "cpu"
    """We have to use the CPU for this assessment."""

    MODEL_SAVE_PATH = "model.pth"
    """The file where data should be saved. If just a filename is given, saves in the CWD."""


class Hyperparameters():
    """
    A class containing constants relevant to the training of the neural network.
    These constants can be altered before runtime to affect various factors of learning.
    """

    BATCH_SIZE = 64
    """The size of each batch provided by the dataloader."""

    _VALIDATION_FRACTION = 0.1
    """The fraction in range [0, 1] of the training dataset to be set aside for validation."""

    VALIDATION_RATIO = (1 - _VALIDATION_FRACTION, _VALIDATION_FRACTION)
    """This value does not need to be adjusted, it is calculated once from `VALIDATION_FRACTION`."""

    LEARNING_RATE = 1e-3
    """Adjust this to change the step size (eta) of the optimiser. 
    Higher values will normally converge quicker, but may skip over the minimum."""

    WEIGHT_DECAY = 0.0005
    """Weight decay is used for regularisation. It is the constant (lambda) by which the
    Regularisation Function, R is multiplied. For Adam, R is the L2 (Euclidean) norm."""

    NUM_EPOCHS = 15
    """The number of learning epochs."""

    LOSS_FN = nn.CrossEntropyLoss()
    """We need to use some variant of Cross Entropy Loss because we are training to identify more than two classes."""


class Cifar10_NN(nn.Module):
    """
    The class for my neural network model. Inherits from the `torch.nn` module.

    Defines a layer stack with a convolution section and a network section.
    Also defines a forward function. This is not to be called directly, but instead is 
    called automatically by PyTorch when the model trains.
    """

    def __init__(self):
        """
        Calls the `nn.Module` constructor and then sets up the layers.
        """
        super().__init__()
        self.layer_stack = nn.Sequential(
            # To make this convolutional, we are going to apply convolutional layers
            # These layers will learn features

            # The number of input channels is 3 because it is a 3-colour image
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # The number of input channels here is the number of output channels from above
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # By using maxpool, we shrink the image by a facter of kernel size (2)
            nn.MaxPool2d(2, 2),
            # We want to renormalise the data
            nn.BatchNorm2d(64),
            # Convolving has occurred >:).

            # This didn't work the first billion times, so let's just make it happen more :]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # We should be down to 4x4 here
            nn.BatchNorm2d(256),

            # We first need to flatten the tensor
            nn.Flatten(),
            # The size of this input layer is fixed based on the properties of our convolution layers
            # We have shrunk the image multiple times, (formely by having no padding when we convolved), and now by using maxpool
            # We can determine size after a convolution layer using the formula:
            # (input_width - filter_size + 2*padding) / stride + 1
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        """WARNING: This function should **not** be called directly!"""
        logits = self.layer_stack(x)
        return logits


class Utility_Functions():
    """Functions required as utilities for the training and the testing of the model.
    """

    def save_model_weights(model: Cifar10_NN, path: str, accuracy=0) -> None:
        """Save the weights of the model to a specified file.

        Args:
            model (Cifar10_NN): The model whose weights should be saved.
            path (str): The path of the file to save the model weights to.
            accuracy (int, optional): The accuracy of the model, used to determine if 
                        the model should be saved to a separate file. Defaults to 0.
        """

        if (accuracy > 80):
            # For models with noteable accuracy, we can save them to a separate file.
            torch.save(model.state_dict(), f"{accuracy:.0f}pc-{path}.")
            print(f"Saved PyTorch Model State to {accuracy:.0f}pc-{path}.")
        torch.save(model.state_dict(), path)
        print(f"Saved PyTorch Model State to {path}.")

    def load_model(path: str) -> None:
        """Creates an instance of the `Cifar10_NN` model and loads saved weights into the model
        from a specified file.

        Args:
            path (str): The path of the file to retrieve the model weights from.

        Returns:
            Cifar10_NN: The model with weights loaded.
        """

        model = Cifar10_NN()
        model.load_state_dict(torch.load(path, weights_only=True))
        return model

    def print_accuracy(percentage: float, loss: float) -> None:
        """Print the accuracy (percentage and loss) neatly."""

        print(
            f"Test Error:\nAccuracy: {(percentage):>0.1f}%, Avg loss:{loss:>8f}")
        print(
            f"[\033[92m{"#"*int(percentage)}\033[91m{"="*(100 - int(percentage))}\033[00m]")

    def print_training_progress(curr: int, max: int) -> None:
        """Print the training progress neatly."""

        made = "#" * round(86*((curr+1)/max))
        togo = "-" * round(86*(1-((curr+1)/max)))
        print(f"\nEpoch {curr+1:>2d} / {max:>2d} [{made}{togo}]")

    def print_epoch_summary(train_start: datetime,
                            start: datetime,
                            end: datetime,
                            previous_accuracy: float,
                            current_accuracy: float) -> None:
        """Print a neat summary of an epoch."""

        print(
            f"Epoch completed in {(end - start)}. Elapsed: {(end - train_start)}")
        improvement = (current_accuracy - previous_accuracy)
        colour = "\033[92m" if improvement > 0 else "\033[91m\a"
        print(f"Improvement of {colour}{improvement:0.1f}%\033[00m")
