import pandas as pd
from sklearn import datasets
from data_set import filepaths as fp
from os import path as osp
import numpy as np
from utils import osUtils as ou
import random

BOSTON_CSV = osp.join(fp.BOSTON_DIR,'boston.csv')
IRIS_CSV = osp.join(fp.IRIS_DIR,'iris.csv')
ML100K_TSV = osp.join(fp.ML100K_DIR,'rating_index.tsv')

def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

def downloadBoston():
    boston = datasets.load_boston()
    df = sklearn_to_df(boston)
    df.to_csv(BOSTON_CSV)

def loadBoston():
    return pd.read_csv(BOSTON_CSV,index_col=0,dtype=np.float32)

def downloadIris():
    iris = datasets.load_iris()
    df = sklearn_to_df(iris)
    df.to_csv(IRIS_CSV)

def split_train_test_from_df(df,test_ratio=0.2):
    test_df = df.sample(frac=test_ratio)
    train_df = df[~df.index.isin(test_df.index)]
    return train_df,test_df



def readRecData( path = ML100K_TSV,test_ratio = 0.2 ):
    user_set,item_set=set(),set()
    triples=[]
    for u, i, r in ou.readTriple(path):
        user_set.add(int(u))
        item_set.add(int(i))
        triples.append((int(u),int(i),int(r)))

    test_set=random.sample(triples,int(len(triples)*test_ratio))
    train_set=list(set(triples)-set(test_set))

    #返回用户集合列表，物品集合列表，与用户，物品，评分三元组列表
    return list(user_set),list(item_set),train_set,test_set












if __name__ == '__main__':
    #downloadBoston()
    downloadIris()
    # df = loadBoston()
    # df.plot()
    # # ss = StandardScaler()
    # # df = ss.fit_transform(df)
    # # print(df)
    # #print(minMaxScalarToTarget(df))
    # import matplotlib.pyplot as plt
    # plt.show()