from chapter1 import data_generator
import random
import numpy as np

def getDatas():
    Ws=[2,3];b=5;n=128
    datas = data_generator.generateData(Ws,b,n)
    return datas

def initWsAndB(datas):
    w_number = len(datas[0])-1
    Ws = np.random.random(w_number)
    b = np.random.random()
    return Ws,b

def train( datas, learning_rate=0.1, epochs=5 ):
    w_number = len(datas[0])-1
    Ws = [random.random() for _ in range(w_number)]
    b = random.random()
    for e in range( epochs ):
        for d in datas:
            x = d[:-1]
            y_pred = sum(x*Ws)+b
            dis = y_pred-d[-1]
            Ws -= dis*learning_rate*x
            b -= dis*learning_rate

def BGD( datas, learning_rate=0.01, epochs=20 ):
    Ws, b = initWsAndB(datas)
    datas = np.array(datas)
    for e in range( epochs ):
        x = datas[:,:-1]
        y_true = datas[:,-1]
        y_pred = np.dot(x,Ws)+b
        dis = y_pred-y_true
        dis = dis.reshape((-1,1))
        Ws -= (dis*learning_rate*x).sum(axis=0)
        b -= (dis*learning_rate).sum(axis=0)
        # 因为w在同一轮epoch时不会改变，todo组织语言

    print(Ws)
    print(b)


def naive_SGD( datas, learning_rate=0.01, epochs=100 ):
    Ws, b = initWsAndB(datas)
    all_datas = np.array(datas)
    for e in range(epochs):
        np.random.shuffle(all_datas)
        datas = all_datas[:int(0.2*len(all_datas))]
        x = datas[:, :-1]
        y_true = datas[:, -1]
        y_pred = np.dot(x, Ws) + b
        dis = y_pred - y_true
        dis = dis.reshape((-1, 1))
        Ws -= (dis * learning_rate * x).sum(axis=0)
        b -= (dis * learning_rate).sum(axis=0)
    print(Ws)
    print(b)

def dataIter(datas,batch_size=24):
    begin = 0
    end = batch_size
    while begin<len(datas):
        yield datas[begin:end]
        begin+=batch_size
        end+=batch_size

def mini_batch_GD( datas, learning_rate=0.01, epochs=20 ):
    Ws, b = initWsAndB(datas)
    all_datas = np.array(datas)
    for e in range( epochs ):
        for datas in dataIter(all_datas,batch_size=24):
            x = datas[:,:-1]
            y_pred = np.dot(x,Ws)+b
            y_true = datas[:,-1]
            dis = y_pred-y_true
            dis = dis.reshape((-1,1))
            Ws -= (dis*learning_rate*x).sum(axis=0)
            b -= (dis*learning_rate).sum(axis=0)
    print(Ws)
    print(b)

def mini_batch_SGD( datas, learning_rate=0.01, epochs=20 ):
    Ws, b = initWsAndB(datas)
    all_datas = np.array(datas)
    np.random.shuffle(all_datas)
    for e in range( epochs ):
        for datas in dataIter(all_datas,batch_size=24):
            x = datas[:,:-1]
            y_pred = np.dot(x,Ws)+b
            y_true = datas[:,-1]
            dis = y_pred-y_true
            dis = dis.reshape((-1,1))
            Ws -= (dis*learning_rate*x).sum(axis=0)
            b -= (dis*learning_rate).sum(axis=0)
    print(Ws)
    print(b)


if __name__ == '__main__':
    mini_batch_SGD(getDatas())