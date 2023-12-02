import torch
from torch import nn
from torch.nn import Parameter

class RNN_Cell(nn.Module):

    def __init__(self, in_dim, hidden_dim ):
        '''
        :param in_dim: 输入向量的维度
        :param hidden_dim: 输出的隐藏层维度
        '''
        super( RNN_Cell, self ).__init__()
        self.Wx = Parameter( torch.randn( in_dim,hidden_dim) )
        self.Wh =  Parameter(torch.randn( hidden_dim, hidden_dim ) )
        self.b = Parameter( torch.randn( 1, hidden_dim ) )

    def forward(self,x,h_1):
        '''
        :param x:  输入的序列中第t个物品向量 [ batch_size, in_dim ]
        :param h_1:  上一个rnn单元输出的隐藏向量 [ batch_size, hidden_dim ]]
        :return: h 当前层输出的隐藏向量 [ batch_size, hidden_dim ]
        '''
        #[ batch_size, hidden_dim ]
        h = torch.tanh( torch.matmul( x, self.Wx )+torch.matmul( h_1, self.Wh )+self.b )
        return h


class RNN_Naive( nn.Module ):

    def __init__( self, in_dim, hidden_dim ):
        super( RNN_Naive, self ).__init__( )
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.rnn_cell = RNN_Cell( in_dim, hidden_dim )

    def forward( self, x ):
        '''
        :param x: 输入的序列向量, 维度为 [ seq_lens, batch_size, dim ]
        :return: outs: 所有RNN_Cell出的隐藏向量[ seq_lens, batch_size, dim ]
                 h: 最后一个RNN_Cell输出的隐藏向量[ batch_size, dim ]
        '''
        outs = []
        h = None
        for seq_x in x :
            if h==None:
                #初始化第一层的输入h
                h = torch.randn(x.shape[1], self.hidden_dim)
            h = self.rnn_cell(seq_x, h )
            outs.append( torch.unsqueeze( h, dim=0 ) )
        outs = torch.cat( outs )
        return outs, h



if __name__ == '__main__':
    BATCH_SIZE = 24
    IN_DIM = 12
    OUT_DIM = 6
    SEQ_LENS = 7

    x = torch.randn(BATCH_SIZE, IN_DIM)
    h = torch.randn(BATCH_SIZE,OUT_DIM)
    rc = RNN_Cell(IN_DIM,OUT_DIM)
    h = rc(x,h)
    print(h.shape)

    rnn = RNN_Naive(IN_DIM,OUT_DIM)
    x = torch.randn(SEQ_LENS,BATCH_SIZE,IN_DIM)
    outs,h = rnn(x)
    #print(outs,h)
    print(outs.shape)
    print(h.shape)
