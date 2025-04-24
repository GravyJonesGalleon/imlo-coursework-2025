import torch
import torch.nn as nn
import torch.nn.functional as F


class classifier(nn.Module):
    def __init__(self,
                 input_features,
                 hidden_layers,
                 output_features,
                 activation_function):
        super(My_NN, self).__init__()

        # We want at least one hidden layer
        if len(hidden_layers) < 1:
            raise Exception("My_NN needs at least one hidden layer")

        # Initialise the layers
        self.layers = []
        # When we add a layer, we specify its size and the size of the layer after
        self.layers.append(nn.Linear(input_features, hidden_layers[0]))
        self.add_module("input_layer", self.layers[0])

        # Now we add the hidden layers
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            self.add_module(f"hidden_layer_{i}", self.layers[i])

        # The process for adding the output layer is different
        # self.out always refers to the output layer
        self.out = nn.Linear(hidden_layers[-1], output_features)

        # Set the activation function
        # We will set the same activation function throughout
        self.activation_function = activation_function

    # We need to describe how the data will though through the network
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.activation(self.layers[i](x))
        return self.out(x)
