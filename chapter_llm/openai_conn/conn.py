import os
from os.path import join as osp
KEY_TXT = osp(os.path.split(os.path.realpath(__file__))[0],'key.txt') #在当前目录创建个key.txt,填上自己的openai key
with open(KEY_TXT,'r',encoding='utf-8') as f:
    OPENAI_KEY = f.read().strip()

