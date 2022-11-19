import torch
from torch import nn


#rnn = nn.RNN( input_size = 12, hidden_size = 6, batch_first = True)
rnn = nn.RNN( input_size = 12, hidden_size = 6)


input = torch.randn(5, 24, 12)
outputs, hn = rnn(input)

print(outputs.size())
print(hn.size())