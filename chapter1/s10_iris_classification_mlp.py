import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from chapter1 import data_download_and_load as ddal
from utils import pltUtils as plt
from utils import evaUtils as eu

class Softmax_Reg_MLP( nn.Module ):
    def __init__( self, n_features,n_classes ):
        super(Softmax_Reg_MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(n_features,n_features*2),
            nn.Sigmoid(),
            nn.Linear(n_features*2, n_features),
            nn.Sigmoid(),
            nn.Linear(n_features, n_classes),
            nn.Softmax(),
        )

    def forward( self, x ):
        y = self.mlp(x)
        return y

@torch.no_grad()
def eva(dates,net):
    net.eval()
    X = torch.Tensor(dates[:,:-1])
    y = dates[:,-1]
    y_pred = net(X)
    pred_classes = np.argmax(y_pred,axis=1)
    acc = eu.accuracy4classification(y,pred_classes)
    #print(acc)
    #plt.drawScatter([y,pred_classes],['true','pred'])
    return acc

def train( epochs = 100, batchSize = 8, lr = 0.01):

    df = ddal.loadIris()
    n_classes = len(df['target'].unique())
    n_featrues = df.shape[1]-1
    train_df, test_df = ddal.split_train_test_from_df(df,test_ratio=0.4)

    net = Softmax_Reg_MLP(n_featrues,n_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW( net.parameters(), lr=lr) #随机梯度下降
    net.train()
    for e in range(epochs):
        for datas in DataLoader(train_df.values, batch_size = batchSize, shuffle = True):
            optimizer.zero_grad() #梯度归0
            X = datas[:,:-1] # 获取X
            y = torch.LongTensor(datas[:,-1].detach().numpy()) # 获取y
            y_pred = net( X ) # 得到预测值y
            loss = criterion(y_pred, y) #将预测的y与真实的y带入损失函数计算损失值
            loss.backward() # 后向传播
            optimizer.step() #更新所有参数
        #print('epoch {},loss={:.4f}'.format(e,loss))
    return eva(test_df.values,net)


if __name__ == '__main__':
    from tqdm import tqdm
    a = []
    for i in tqdm(range(10)):
        acc = train()
        a.append(acc)
    print(np.average(np.array(a)))
