
import torch
from torch import nn

conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,2), padding=(1,2),stride=(2,1,1,1))

x = torch.rand(size=(1,1,8, 8))
y = conv2d(x)

print(y.shape)
