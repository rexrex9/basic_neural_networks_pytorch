from chapter1 import data_generator
import random

def getDatas():
    Ws=[2,3];b=2;n=128
    datas = data_generator.generateData(Ws,b,n)
    return datas

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
    print(Ws)
    print(b)


if __name__ == '__main__':
    datas=getDatas()
    #print(datas)
    train(datas)