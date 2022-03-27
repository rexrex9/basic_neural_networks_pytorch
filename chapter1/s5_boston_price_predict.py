import torch
from torch import nn
from torch.utils.data import DataLoader
from chapter1 import data_download_and_load as ddal
from utils import pltUtils as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class Linear_Reg( nn.Module ):
    def __init__( self, n_features ):
        super(Linear_Reg, self).__init__()
        self.linear = nn.Linear(n_features,1,bias=True)

    def forward( self, x ):
        y = self.linear(x)
        y = torch.squeeze( y )
        return y

@torch.no_grad()
def eva(dates,net,min,max):
    net.eval()
    X = torch.Tensor(dates[:,:-1])
    y = dates[:,-1]
    y_pred = net(X)
    y_pred = anti_minMaxScaler(y_pred,min,max)
    y = anti_minMaxScaler(y,min,max)
    criterion = torch.nn.MSELoss()  # 平方差损失函数
    loss = criterion(y_pred,torch.Tensor(y))
    print(loss**0.5)
    plt.drawLines([y,y_pred],['true','pred'])

def preprocess(df):
    ss = MinMaxScaler()
    df = ss.fit_transform(df)
    df = pd.DataFrame(df)
    return df,ss.data_min_[-1],ss.data_max_[-1]

def anti_minMaxScaler(d,min,max):
    '''
        (x-min)/(max-min)
    '''
    return d*(max-min)+min

def train( epochs = 20, batchSize = 16, lr = 0.01):

    df,min,max = preprocess(ddal.loadBoston())
    train_df, test_df = ddal.split_train_test_from_df(df,test_ratio=0.2)

    net = Linear_Reg(train_df.shape[1]-1) # 初始化线性回归模型
    criterion = torch.nn.MSELoss() #平方差损失函数
    optimizer = torch.optim.SGD( net.parameters(), lr=lr) #随机梯度下降
    net.train()
    for e in range(epochs):
        for datas in DataLoader(train_df.values, batch_size = batchSize, shuffle = True):
            optimizer.zero_grad() #梯度归0
            X = datas[:,:-1] # 获取X
            y = datas[:,-1] # 获取y
            y_pred = net( X ) # 得到预测值y
            loss = criterion(y_pred, y) #将预测的y与真实的y带入损失函数计算损失值
            loss.backward() # 后向传播
            optimizer.step() #更新所有参数
        print('epoch {},loss={:.4f}'.format(e,loss))
    eva(test_df.values,net,min,max)

if __name__ == '__main__':
    train()