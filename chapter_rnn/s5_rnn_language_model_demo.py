import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from torch.utils.data import DataLoader
from data_set import filepaths as fp
from tqdm import tqdm
import numpy as np
from torch import nn
import torch
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def getSongsi():
    with open(fp.SONG_CI_JSON,'r',encoding='utf-8') as f:
        ds = json.load(f)
    ls = []
    [ls.extend(d['paragraphs']) for d in ds]
    copurs = ''.join(ls)
    return copurs

class Tokenizer():

    def __init__(self,corpus):
        all = [c for c in corpus]

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
        self.linear = nn.Linear( hidden_size, n_items )

    def forward(self, x):
        # [batch_size, len_seqs, dim]
        token_embs = self.token_embs(x)
        # [batch_size, len_seqs, hidden_size]
        outs,_ = self.rnn(token_embs)
        # [batch_size*len_seqs, out_size]
        outs = outs.reshape(-1,self.hidden_size )
        #[batch_size*len_seqs, out_size]
        out = self.linear(outs)
        return out

    def predict(self,x):
        token_embs = self.token_embs(x)
        _, (h,c) = self.rnn(token_embs)
        h = torch.squeeze(h)
        out = self.linear(h)
        return out

def predict(s,tokenizer,net):
    tokenids = tokenizer.sentences2tokens(s)
    tokenids = torch.LongTensor(tokenids).to(device)
    logits = net.predict(tokenids)
    out = np.argmax(logits.detach().cpu().numpy(),axis=0)
    s = tokenizer.token2char(out)
    return s

def generate(s,lens,tokenizer,net):
    cs = s[:]
    for i in range(lens):
        c = predict(cs,tokenizer,net)
        s+=c
        cs+=c
        cs=cs[1:]
    print(s)

def train( epochs = 50, sens_len=6,batchSize = 512, lr = 0.002, rnn_hidden_size = 64, in_dim = 64, per_epochs=10):
    tokenizer = Tokenizer(getSongsi())
    seqs = generalTrainData(tokenizer.tokenids,sens_len)
    net = RNNLM( tokenizer.token_num, rnn_hidden_size, in_dim).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW( net.parameters(), lr=lr)
    #开始训练
    for e in tqdm(range(epochs)):
        loss=None
        for seq in DataLoader(seqs, batch_size = batchSize, shuffle = True):
            x = torch.LongTensor(seq[:,0,:].detach().numpy()).to(device)
            y = torch.LongTensor(seq[:,1,:].detach().numpy()).to(device)
            optimizer.zero_grad()
            logits = net( x )
            y = y.reshape(-1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        if e%per_epochs==0:
            generate('翠禽枝上消魂',50,tokenizer,net)
            print('epoch {},avg_loss={:.4f}'.format(e, loss))

if __name__ == '__main__':
    train()

