import torch
from torch import nn
from torch.utils.data import DataLoader
from chapter1 import data_generator

def getDatas():
    Ws=[2,3];b=5;n=128
    datas = data_generator.generateData(Ws,b,n)
    return datas

class Linear_Reg( nn.Module ):
    def __init__( self, n_features ):
        super(Linear_Reg, self).__init__()
        self.linear = nn.Linear(n_features,1,bias=True)

    def forward( self, x ):
        y = self.linear(x)
        y = torch.squeeze( y )
        return y

def train( epochs = 30, batchSize = 24, lr = 0.05 ):
    datas = getDatas()
    net = Linear_Reg(len(datas[0])-1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD( net.parameters(), lr=lr)
    for e in range(epochs):
        all_lose = 0
        for datas in DataLoader(datas, batch_size = batchSize, shuffle = True):
            optimizer.zero_grad()
            X = datas[:,:-1]
            y = datas[:,-1]
            y_hat = net( X )
            loss = criterion(y_hat, y)
            all_lose += loss
            loss.backward()
            optimizer.step()
        print('epoch {},loss={:.4f}'.format(e,loss))
    for p in net.parameters():
        print(p)

if __name__ == '__main__':
    train()