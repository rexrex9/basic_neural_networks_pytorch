import torch
from torch import nn
from chapter1 import data_download_and_load as ddad
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class Linear_Reg( nn.Module ):
    def __init__( self, n_features ):
        super(Linear_Reg, self).__init__()
        self.linear = nn.Linear(n_features,1)

    def forward( self, x ):
        y = self.linear(x)
        y = torch.squeeze( y )
        return y

def preprocess(df):
    ss = MinMaxScaler()
    df = ss.fit_transform(df)
    df = pd.DataFrame(df)
    return df

def train( epochs = 20, batchSize = 32, lr = 0.005 ):
    #读取数据
    train_df, test_df = ddad.split_train_test_from_df(preprocess(ddad.loadBoston()))
    len_features = train_df.shape[1]-1
    net = Linear_Reg( len_features)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW( net.parameters(), lr=lr,weight_decay=0.05)

    for e in range(epochs):
        all_lose = 0
        for datas in DataLoader(train_df.values, batch_size = batchSize, shuffle = True):
            optimizer.zero_grad()
            X = datas[:,:-1]
            y = datas[:,-1]
            y_hat = net( X )
            loss = criterion(y_hat, y)
            all_lose += loss
            loss.backward()
            optimizer.step()
        print(y_hat)
        print(y)
        print('epoch {},avg_loss={:.4f}'.format(e,all_lose/(train_df.shape[0]//batchSize)))

        # #评估模型
        # if e % eva_per_epochs == 0:
        #     p, r, acc = doEva(net, x_train, y_train)
        #     print('train:p:{:.4f}, r:{:.4f}, acc:{:.4f}'.format(p, r, acc))
        #     p, r, acc = doEva(net, x_test, y_test)
        #     print('test:p:{:.4f}, r:{:.4f}, acc:{:.4f}'.format(p,r, acc))

if __name__ == '__main__':
    train()