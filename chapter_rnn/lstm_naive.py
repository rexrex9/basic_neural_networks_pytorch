from torch import nn
import torch

class LSTM_Cell(nn.Module):

    def __init__(self, in_dim, hidden_dim ):
        '''
        :param in_dim: 输入向量的维度
        :param hidden_dim: 输出的隐藏层维度
        '''
        super( LSTM_Cell, self ).__init__()
        self.ix_linear = nn.Linear(in_dim,hidden_dim)
        self.ih_linear = nn.Linear(hidden_dim,hidden_dim)
        self.fx_linear = nn.Linear(in_dim, hidden_dim)
        self.fh_linear = nn.Linear(hidden_dim, hidden_dim)
        self.ox_linear = nn.Linear(in_dim, hidden_dim)
        self.oh_linear = nn.Linear(hidden_dim, hidden_dim)
        self.cx_linear = nn.Linear(in_dim, hidden_dim)
        self.ch_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self,x,h_1,c_1):
        '''
        :param x:  输入的序列中第t个物品向量 [ batch_size, in_dim ]
        :param h_1:  上一个lstm单元输出的隐藏向量 [ batch_size, hidden_dim ]]
        :param c_1: 上一个lstm单元输出的c向量 [ batch_size, hidden_dim ]]
        :return: h 当前层输出的隐藏向量 [ batch_size, hidden_dim ]
        '''
        i = torch.sigmoid(self.ix_linear(x) + self.ih_linear(h_1))
        f = torch.sigmoid(self.fx_linear(x) + self.fh_linear(h_1))
        o = torch.sigmoid(self.ox_linear(x) + self.oh_linear(h_1))
        c_ = torch.tanh(self.cx_linear(x) + self.ch_linear(h_1))
        c = f*c_1+i*c_
        h = o*torch.tanh(c)
        return h,c


class LSTM_Naive( nn.Module ):

    def __init__( self, in_dim, hidden_dim ):
        super( LSTM_Naive, self ).__init__( )
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.rnn_cell = LSTM_Cell( in_dim, hidden_dim )

    def forward( self, x ):
        '''
        :param x: 输入的序列向量, 维度为 [ batch_size, seq_lens, dim ]
        :return: outs: 所有RNN_Cell出的隐藏向量[ batch_size, seq_lens, dim ]
                 h: 最后一个RNN_Cell输出的隐藏向量[ batch_size, dim ]
        '''
        outs = []
        h,c = None,None
        for seq_x in  x :
            if h==None: h = torch.randn( x.shape[1], self.hidden_dim )
            if c==None: c= torch.randn( x.shape[1], self.hidden_dim )
            h,c = self.rnn_cell(seq_x, h,c )
            outs.append( torch.unsqueeze( h, dim=1 ) )
        outs = torch.cat( outs, dim=1 )
        return outs, h



if __name__ == '__main__':
    x = torch.randn(24, 12)
    h = torch.randn(24,6)
    c = torch.randn(24,6)
    rc = LSTM_Cell(12,6)
    h,c = rc(x,h,c)
    print(h.shape)

    rnn =LSTM_Naive(12,6)
    x = torch.randn(7,24,12)
    outs,h = rnn(x)
    print(outs.shape)
    print(h.shape)