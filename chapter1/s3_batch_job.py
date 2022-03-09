from chapter1 import data_generator
import random
import numpy as np

def getDates():
    Ws=[2,3];b=5;n=128
    Dates = data_generator.generateData(Ws,b,n)
    return Dates

def train( Dates, learning_rate=0.1, epochs=5 ):
    w_number = len(Dates[0])-1
    Ws = [random.random() for _ in range(w_number)]
    b = random.random()
    for e in range( epochs ):
        for d in Dates:
            x = d[:-1]
            y_pred = sum(x*Ws)+b
            dis = y_pred-d[-1]
            Ws -= dis*learning_rate*x
            b -= dis*learning_rate
    print(Ws)
    print(b)

def BGD( Dates, learning_rate=0.01, epochs=20 ):
    w_number = len(Dates[0])-1
    Ws = np.random.random(w_number)
    b = np.random.random()
    Dates = np.array(Dates)
    for e in range( epochs ):
        x = Dates[:,:-1]
        y_true = Dates[:,-1]
        y_pred = np.dot(x,Ws)+b
        dis = y_pred-y_true
        dis = dis.reshape((-1,1))
        Ws -= (dis*learning_rate*x).sum(axis=0)
        b -= (dis*learning_rate).sum(axis=0)
    print(Ws)
    print(b)

def SGD( Dates, learning_rate=0.01, epochs=100 ):
    w_number = len(Dates[0])-1
    Ws = np.random.random(w_number)
    b = np.random.random()
    all_Dates = np.array(Dates)
    for e in range( epochs ):
        np.random.shuffle(all_Dates)
        Dates = all_Dates[:int(0.2*len(all_Dates))]
        x = Dates[:,:-1]
        y_true = Dates[:,-1]
        y_pred = np.dot(x,Ws)+b
        dis = y_pred-y_true
        dis = dis.reshape((-1,1))
        Ws -= (dis*learning_rate*x).sum(axis=0)
        b -= (dis*learning_rate).sum(axis=0)
    print(Ws)
    print(b)

def dataIter(dates,batch_size=24):
    begin = 0
    end = batch_size
    while begin<len(dates):
        yield dates[begin:end]
        begin+=batch_size
        end+=batch_size


def MBGD( Dates, learning_rate=0.01, epochs=10 ):
    w_number = len(Dates[0])-1
    Ws = np.random.random(w_number)
    b = np.random.random()
    all_Dates = np.array(Dates)
    for e in range( epochs ):
        for Dates in dataIter(all_Dates,batch_size=24):
            x = Dates[:,:-1]
            y_true = Dates[:,-1]
            y_pred = np.dot(x,Ws)+b
            dis = y_pred-y_true
            dis = dis.reshape((-1,1))
            Ws -= (dis*learning_rate*x).sum(axis=0)
            b -= (dis*learning_rate).sum(axis=0)
    print(Ws)
    print(b)

def MSGD( Dates, learning_rate=0.02, epochs=20 ):
    w_number = len(Dates[0])-1
    Ws = np.random.random(w_number)
    b = np.random.random()
    all_Dates = np.array(Dates)
    for e in range( epochs ):
        np.random.shuffle(all_Dates)
        for Dates in dataIter(all_Dates,batch_size=24):
            x = Dates[:,:-1]
            y_true = Dates[:,-1]
            y_pred = np.dot(x,Ws)+b
            dis = y_pred-y_true
            dis = dis.reshape((-1,1))
            Ws -= (dis*learning_rate*x).sum(axis=0)
            b -= (dis*learning_rate).sum(axis=0)
    print(Ws)
    print(b)


if __name__ == '__main__':
    dates=getDates()

    MSGD(dates)
