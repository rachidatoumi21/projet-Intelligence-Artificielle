import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

class FashionMLP(nn.Module):
    def __init__(
            self,
            input_size,
            h1_size,
            h2_size,
            output_size,
            activation1,
            activation2,
            output_activation=None):
        super(FashionMLP, self).__init__()
        
        # TODO
        # Créer un réseau de neurones MLP selon les paramètres d'entrée. Utilisez les couches de torch.nn
        self.flatten = nn.Flatten()  # couche pour adapter l'entrée 28x28 en vecteur

        self.layers = []
        self.layers.append(nn.Linear(input_size, h1_size))
        self.layers.append(activation1)

        self.use_h2 = h2_size is not None and h2_size > 0

        if self.use_h2:
            self.layers.append(nn.Linear(h1_size, h2_size))
            self.layers.append(activation2)
            self.layers.append(nn.Linear(h2_size, output_size))
        else:
            self.layers.append(nn.Linear(h1_size, output_size))

        if output_activation is not None:
            self.layers.append(output_activation)

        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        # TODO
        # Implémentez le forward pass de votre réseau de neurones
        x = self.flatten(x)  # [batch, 1, 28, 28] -> [batch, 784]
        logits = self.network(x)
        return logits