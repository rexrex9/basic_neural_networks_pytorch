from os.path import join as osp
import os
import json
import pandas as pd
from datasets import Dataset

DATAS_DIR = osp(os.path.split(os.path.realpath(__file__))[0],'datas')
DATAS_JSON = osp(DATAS_DIR,'data.json')

def load():
    with open(DATAS_JSON,'r',encoding='utf-8') as f:
        datas=json.load(f)
    df = pd.DataFrame(datas)
    dataset = Dataset.from_pandas(df)
    return dataset

