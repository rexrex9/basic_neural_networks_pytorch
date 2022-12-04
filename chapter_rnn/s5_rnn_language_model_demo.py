import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from torch.utils.data import DataLoader
from data_set import filepaths as fp
#import jieba
from tqdm import tqdm
import numpy as np
from torch import nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Tokenizer():

    def __init__(self,path):
        with open(path,'r',encoding='utf-8') as f:
            docs = f.read()
            docs = docs.replace('\n', ',')
        #all = list(jieba.cut(docs))
        all = [c for c in docs]

        self.tokens = list(set(all))
        self.token_num = len(self.tokens)
        self.token2id = {k:i for i,k in enumerate(self.tokens)}
        self.id2token = {self.token2id[k]:k for k in self.token2id}
        self.tokenids = [self.token2id[t] for t in all]

    def sentences2tokens(self,s):
        #s = list(jieba.cut(s))
        tokens = [self.token2id[w] for w in s]
        return tokens

    def tokens2sentences(self,ts):
        s = ''.join([self.id2token[t] for t in ts])
        return s

    def token2char(self,t):
        return self.id2token[t]


def generalTrainData(tokenids,sens_len=12):
    max_len = len(tokenids)
    seqs = []
    for i in tqdm(range(max_len-sens_len-1)):
        seqs.append([tokenids[i:i+sens_len],tokenids[i+1:i+sens_len+1]])
    seqs = np.array(seqs)
    np.random.shuffle(seqs)
    return seqs

class RNNLM( nn.Module ):

    def __init__( self, n_items, hidden_size=512, in_dim = 512):
        super( RNNLM, self ).__init__()
        # 随机初始化所有tokens
        self.hidden_size = hidden_size
        self.token_embs = nn.Embedding( n_items, in_dim, max_norm = 1 )
        self.rnn = nn.LSTM( in_dim, hidden_size, batch_first = True )
        self.dense = self.dense_layer( hidden_size, n_items )

    def grad_clipping(self, theta=1):
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in self.parameters()))
        if norm > theta:
            for param in self.parameters():
                param.grad[:] *= theta / norm

    # 全连接层
    def dense_layer(self,in_features,out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Tanh())

    def forward(self, x):
        # [batch_size, len_seqs, dim]
        token_embs = self.token_embs(x)
        # [batch_size, len_seqs, hidden_size]
        outs,_ = self.rnn(token_embs)
        # [batch_size*len_seqs, out_size]
        outs = outs.reshape(-1,self.hidden_size )
        #[batch_size*len_seqs, out_size]
        out = self.dense(outs)
        return out

    def predict(self,x):
        token_embs = self.token_embs(x)
        _, (h,c) = self.rnn(token_embs)
        h = torch.squeeze(h)
        out = self.dense(h)
        return out

def predict(s,tokenizer,net):
    tokenids = tokenizer.sentences2tokens(s)
    tokenids = torch.LongTensor(tokenids)
    logits = net.predict(tokenids)
    out = np.argmax(logits.detach().numpy(),axis=0)
    s = tokenizer.token2char(out)
    return s

def generate(s,lens,tokenizer,net):
    for i in range(lens):
        s+=predict(s,tokenizer,net)
    print(s)


def train( epochs = 50, sens_len=3,batchSize = 512, lr = 0.01, rnn_hidden_size = 256, in_dim = 256 ):
    tokenizer = Tokenizer(fp.JAY_LYRICS)
    seqs = generalTrainData(tokenizer.tokenids,sens_len)
    net = RNNLM( tokenizer.token_num, rnn_hidden_size, in_dim).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW( net.parameters(), lr=lr, weight_decay=5e-3)
    #开始训练
    for e in range(epochs):
        loss=None
        for seq in tqdm(DataLoader(seqs, batch_size = batchSize, shuffle = True)):
            x = torch.LongTensor(seq[:,0,:].detach().numpy()).to(device)
            y = torch.LongTensor(seq[:,1,:].detach().numpy()).to(device)
            optimizer.zero_grad()
            logits = net( x )
            y = y.reshape(-1)
            loss = criterion(logits, y)
            loss.backward()
            #net.grad_clipping()
            optimizer.step()
        generate('想要有',30,tokenizer,net)
        print('epoch {},avg_loss={:.4f}'.format(e, loss))

if __name__ == '__main__':
    train()

