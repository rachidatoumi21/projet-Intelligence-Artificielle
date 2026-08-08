import torch
import torch.nn as nn
from MLP import FashionMLP

model = FashionMLP(
    input_size=28*28,
    h1_size=64,
    h2_size=64,
    output_size=10,
    activation1=nn.Sigmoid(),
    activation2=nn.Sigmoid()
)

x = torch.randn(8, 1, 28, 28)
out = model(x)

print(out.shape)