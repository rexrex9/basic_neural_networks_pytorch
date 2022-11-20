import torch
from torch import nn

BATCH_SIZE = 24
IN_DIM =12
OUT_DIM = 6
SEQ_LENS = 7

rnn = nn.RNN( input_size = IN_DIM, hidden_size = OUT_DIM, batch_first = True)
#rnn = nn.RNN( input_size = 12, hidden_size = 6)


input = torch.randn(BATCH_SIZE, SEQ_LENS, IN_DIM)
outputs, hn = rnn(input)
hn = torch.squeeze(hn)
print(outputs.size())
print(hn.size())