import torch
from torch import nn

def corr(x,k):
    h,w = k.shape
    y = torch.zeros((x.shape[0]-h+1,x.shape[1]-w+1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i,j]=(x[i:i+h,j:j+w]*k).sum()
    return y

class Conv(nn.Module):
    def __init__(self,kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self,x):
        return corr(x,self.weight) + self.bias



if __name__ == '__main__':
    x = torch.FloatTensor([[1,2,3],[4,5,6],[7,8,9]])
    k = torch.tensor([[0,1],[2,3]])
    y=corr(x,k)
    print(y)


    conv = Conv([2,3])
    y=conv(x)
    print(y)

    nn.Conv2d()