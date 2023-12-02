import matplotlib.pyplot as plt

def drawLines(ds,names):
    '''
    :param ds: [数据列表1，数据列表2]
    :param names:[name1,name2]
    :return:
    '''
    plt.figure()
    x = range(len(ds[0]))
    for d,name in zip(ds,names):
        plt.plot(x,d,label=name)
        plt.xlabel('X', fontsize=14)
        plt.ylabel('Y', fontsize=14)
        plt.legend(fontsize=16, loc='upper left')
        plt.grid(c='gray')
    plt.show()

def drawScatter(ds,names):
    fig, ax = plt.subplots()
    x = range(len(ds[0]))
    for d,name in zip(ds,names):
        ax.scatter(x,d,alpha=0.6,label=name)
        ax.legend(fontsize=16, loc='upper left')
        ax.grid(c='gray')
    plt.show()