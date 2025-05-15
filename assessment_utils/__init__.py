# This is an initialisation file used to setup constants used in both training and testing.
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


#################################
#       Hyperparameters
#################################

batch_size = 54
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
epochs = 15


#################################
#            Datasets
#################################

# We want to use the CIFAR-10 dataset to train the data.
# To normalise this data, we need to know the means and stdevs of the dataset.
# This can be found using the program find_mean_and_std.py

cifar_train_means = (0.4914008203125,
                     0.482158984375,
                     0.4465309375)
cifar_train_stdvs = (0.2470276019799247,
                     0.24348346588930533,
                     0.2615877093211206)

training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(30),
        v2.RandomAdjustSharpness(0.5, p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(cifar_train_means, cifar_train_stdvs)
    ])
)

train_dataloader = DataLoader(
    training_data, batch_size=batch_size, shuffle=True)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(cifar_train_means, cifar_train_stdvs)
    ])
)

test_dataloader = DataLoader(test_data, batch_size=batch_size)


# These are the classes in the CIFAR-10 dataset, in order
cifar10_classes = ("plane", "car", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck")


#################################
#           Constants
#################################


# We want to use the CPU for this assessment
device = torch.accelerator.current_accelerator(
).type if torch.accelerator.is_available() else "cpu"
# device = "cpu"
print(f"Using {device} device")

# The file where data should be saved
# Saves in the CWD
model_save_path = "model.pth"

# The pictures in the CIFAR-10 datasets are of size 3 * 32 * 32 (3 colours, 32 height, 32 width)
input_features = 3*32*32


#################################
#             Model
#################################


class Cifar_NN(nn.Module):
    def __init__(self):
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
        logits = self.layer_stack(x)
        return logits


#################################
#           Methods
#################################


def train(dataloader: torch.utils.data.DataLoader, model, loss_fn, optimiser) -> None:
    size = len(dataloader.dataset)
    numbatches = size // batch_size
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

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


def test(dataloader: torch.utils.data.DataLoader, model, loss_fn) -> float:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error:\nAccuracy: {(100*correct):>0.1f}%, Avg loss:{test_loss:>8f}")
    print(
        f"[\033[92m{"#"*int(100*correct)}\033[91m{"="*(100 - int(100*correct))}\033[00m]")
    return correct


def save_model_weights(model: torch.nn, path: str, accuracy=0) -> None:
    if (accuracy > 80):
        path = f"{accuracy:.0f}pc-{path}"
    torch.save(model.state_dict(), path)
    print(f"Saved PyTorch Model State to {path}")


def load_model(path: str) -> None:
    # Getting the model before installing the data means that we can avoid downloading the data before discovering there is a problem
    model = Cifar_NN(input_features)
    model.load_state_dict(torch.load(
        path, weights_only=True))

    return model
