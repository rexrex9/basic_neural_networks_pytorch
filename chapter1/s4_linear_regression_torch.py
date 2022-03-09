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
        #self.W = nn.Parameter(torch.rand(n_features,1))
        #self.b = nn.Parameter(torch.rand(1))

    def forward( self, x ):
        y = self.linear(x)
        #y=x.matmul(self.W)+self.b
        y = torch.squeeze( y )
        return y

def train( epochs = 20, batchSize = 24, lr = 0.05):
    datas = getDatas() #得到数据
    net = Linear_Reg(len(datas[0])-1) # 初始化线性回归模型
    criterion = torch.nn.MSELoss() #平方差损失函数
    optimizer = torch.optim.SGD( net.parameters(), lr=lr) #随机梯度下降
    for e in range(epochs):
        for datas in DataLoader(datas, batch_size = batchSize, shuffle = True):
            optimizer.zero_grad() #梯度归0
            X = datas[:,:-1] # 获取X
            y = datas[:,-1] # 获取y
            y_pred = net( X ) # 得到预测值y
            loss = criterion(y_pred, y) #将预测的y与真实的y带入损失函数计算损失值
            loss.backward() # 后向传播
            optimizer.step() #更新所有参数
        print('epoch {},loss={:.4f}'.format(e,loss))
    for p in net.parameters():
        print(p)

if __name__ == '__main__':
    train()