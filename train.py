import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# We want to use the CIFAR-10 dataset to train the data.
training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

# What is batch size? TODO: Look tha shit up.
batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)


class cifar_NN(nn.Module):
    def __init__(self,
                 input_features: int,
                 hidden_layer_features: list[int],
                 output_features: int,
                 activation_function: function):
        super().__init__()

        # We want to make sure that the layers entered are valid before we create them
        if (input_features < 1):
            raise Exception(
                "The neural network needs at least one input feature")
        if (len(hidden_layer_features) < 1):
            raise Exception(
                "The neural network needs at least one hidden layer")

        # We want to keep track of all our layers :)
        self.layers = []

        self.layers[0] = nn.Linear(input_features, hidden_layer_features[0])
        self.add_module("input_layer", self.layers[0])
        for i in range(1, len(hidden_layer_features)):
            self.layers.append(
                nn.Linear(hidden_layer_features[i - 1], hidden_layer_features[i]))
            self.add_module(f"hidden_layer{i}", self.layers[i])

        # Set the output layer
        self.out = nn.Linear(hidden_layer_features[-1], output_features)

        self.activation = activation_function

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.activation(self.layers[i](x))
        return self.out(x)


# Instantiate the model
classifier = cifar_NN(input_features=4,
                      hidden_layer_features=[16, 8],
                      output_features=3,
                      activation_function=nn.ReLU
                      )
