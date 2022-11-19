from torch import nn
import torch

class GRU_Cell(nn.Module):

    def __init__(self, in_dim, hidden_dim ):
        '''
        :param in_dim: 输入向量的维度
        :param hidden_dim: 输出的隐藏层维度
        '''
        super( GRU_Cell, self ).__init__()
        self.rx_linear = nn.Linear(in_dim,hidden_dim)
        self.rh_linear = nn.Linear(hidden_dim,hidden_dim)
        self.zx_linear = nn.Linear(in_dim, hidden_dim)
        self.zh_linear = nn.Linear(hidden_dim, hidden_dim)
        self.hx_linear = nn.Linear(in_dim, hidden_dim)
        self.hh_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self,x,h_1):
        '''
        :param x:  输入的序列中第t个物品向量 [ batch_size, in_dim ]
        :param h_1:  上一个GRU单元输出的隐藏向量 [ batch_size, hidden_dim ]]
        :return: h 当前层输出的隐藏向量 [ batch_size, hidden_dim ]
        '''

        r = torch.sigmoid(self.rx_linear(x)+self.rh_linear(h_1))
        z = torch.sigmoid(self.zx_linear(x)+self.zh_linear(h_1))
        h_ = torch.tanh(self.hx_linear(x)+self.hh_linear(r*h_1))
        h = z*h_1+(1-z)*h_
        return h


class GRU_Naive( nn.Module ):

    def __init__( self, in_dim, hidden_dim ):
        super( GRU_Naive, self ).__init__( )
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.rnn_cell = GRU_Cell( in_dim, hidden_dim )

    def forward( self, x ):
        '''
        :param x: 输入的序列向量, 维度为 [ batch_size, seq_lens, dim ]
        :return: outs: 所有RNN_Cell出的隐藏向量[ batch_size, seq_lens, dim ]
                 h: 最后一个RNN_Cell输出的隐藏向量[ batch_size, dim ]
        '''
        outs = []
        h = None
        for seq_x in  x :
            if h==None:
                #初始化第一层的输入h
                h = torch.randn( x.shape[1], self.hidden_dim )
            h = self.rnn_cell(seq_x, h )
            outs.append( torch.unsqueeze( h, dim=1 ) )
        outs = torch.cat( outs, dim=1 )
        return outs, h



if __name__ == '__main__':
    x = torch.randn(24, 12)
    h = torch.randn(24,6)
    rc = GRU_Cell(12,6)
    h = rc(x,h)
    print(h.shape)

    rnn = GRU_Naive(12,6)
    x = torch.randn(7,24,12)
    outs,h = rnn(x)
    print(outs.shape)
    print(h.shape)