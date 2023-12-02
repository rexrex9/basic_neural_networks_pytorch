from os.path import join as osp
import os
from datasets import Dataset

DATAS_DIR = osp(os.path.split(os.path.realpath(__file__))[0],'datas')
DATAS_JSON = osp(DATAS_DIR,'data.json')

def load():
    dataset = Dataset.from_json(DATAS_JSON)
    return dataset

if __name__ == '__main__':
    print(load())

