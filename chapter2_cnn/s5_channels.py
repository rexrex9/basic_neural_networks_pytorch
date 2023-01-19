
import torch
from torch import nn

conv2d = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3)

x = torch.rand(size=(128, 3, 12, 12))

print(conv2d(x).shape)

#batch_size