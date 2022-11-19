import torch
from torch import nn

gru = nn.GRU( input_size = 12, hidden_size = 6)

input = torch.randn(5, 24, 12)
outputs, hn = gru(input)

print(outputs.size())
print(hn.size())