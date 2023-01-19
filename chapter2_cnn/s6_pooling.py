from torch import nn
import torch

p = nn.MaxPool2d(kernel_size=2,padding=0,stride=1)

x = torch.FloatTensor([[[[1,2,3],[4,5,6],[7,8,9]]]])

print(x.shape)

y=p(x)
print(y)
#nn.MaxPool2d(kernel_size=(2,3),padding=(1,2),stride=(2,3))


