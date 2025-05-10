# This is an initialisation file used to setup constants used in both training and testing.
import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


#################################
#            Datasets
#################################


batch_size = 64

# We want to use the CIFAR-10 dataset to train the data.
training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=batch_size)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

test_dataloader = DataLoader(test_data, batch_size=batch_size)


#################################
#           Constants
#################################

# We want to use the CPU for this assessment
device = "cpu"

# The file where data should be saved
# Saves in the CWD
model_save_path = "model.pth"

# The pictures in the CIFAR-10 datasets are of size 3 * 32 * 32 (3 colours, 32 height, 32 width)
input_features = 3*32*32

#################################
#         Training Params
#################################
loss_fn = nn.CrossEntropyLoss()
lr = 1e-2
epochs = 5

#################################
#             Model
#################################


class Cifar_NN(nn.Module):
    def __init__(self,
                 input_features):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


#################################
#           Methods
#################################


def train(dataloader: torch.utils.data.DataLoader, model, loss_fn, optimiser) -> None:
    size = len(dataloader.dataset)
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

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Loss: {loss:7f} [{current:>5d}/{size:>5d}]")


def test(dataloader: torch.utils.data.DataLoader, model, loss_fn) -> None:
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
        f"Test Error:\nAccuracy: {(100*correct):>0.1f}%, Avg loss:{test_loss:>8f} \n")


def save_model_weights(model: torch.nn, path: str) -> None:
    torch.save(model.state_dict(), path)
    print(f"Saved PyTorch Model State to {path}")


def load_model(path: str) -> None:
    # Getting the model before installing the data means that we can avoid downloading the data before discovering there is a problem
    model = Cifar_NN(input_features)
    model.load_state_dict(torch.load(
        path, weights_only=True))

    return model
