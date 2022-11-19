import torch
from torch import nn

lstm = nn.LSTM( input_size = 12, hidden_size = 6)

input = torch.randn(5, 24, 12)
outputs, hn = lstm(input)

print(outputs.size())
h,c = hn
print(h.shape)
print(c.shape)