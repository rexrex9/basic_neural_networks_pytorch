import pandas as pd
from sklearn import datasets #机器学习库
from data_set import filepaths as fp
from os import path as osp
import numpy as np
from Utils import osUtils as ou
import random

'''
MedInc: 区域内的平均收入中位数。
HouseAge: 区域内房屋的平均年龄。
AveRooms: 每户的平均房间数。
AveBedrms: 每户的平均卧室数。
Population: 区域人口。
AveOccup: 每户的平均居住人数。
Latitude: 区域的纬度。
Longitude: 区域的经度。
目标变量（即需要预测的变量）是：

MedHouseVal: 区域内房屋价值的中位数（以十万美元为单位）。

'''

BOSTON_CSV = osp.join(fp.BOSTON_DIR,'boston1.csv')
IRIS_CSV = osp.join(fp.IRIS_DIR,'iris.csv')
ML100K_TSV = osp.join(fp.ML100K_DIR,'rating_index.tsv')

def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

def downloadBoston():
    boston = datasets.fetch_california_housing()
    df = sklearn_to_df(boston)
    print(df)
    df.to_csv(BOSTON_CSV)

def loadBoston():
    return pd.read_csv(BOSTON_CSV,index_col=0,dtype=np.float32)

def downloadIris():
    iris = datasets.load_iris()
    df = sklearn_to_df(iris)
    df.to_csv(IRIS_CSV)

def loadIris():
    return pd.read_csv(IRIS_CSV,index_col=0,dtype=np.float32)

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
    downloadBoston()

    #downloadIris()
    # df = loadBoston()
