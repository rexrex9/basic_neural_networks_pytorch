import os

ROOT = os.path.split(os.path.realpath(__file__))[0]


#http://lib.stat.cmu.edu/datasets/boston
BOSTON_DIR = os.path.join( ROOT,'boston' )
IRIS_DIR= os.path.join( ROOT,'iris' )
ML100K_DIR = os.path.join( ROOT, 'ml-100k')

SEQS = os.path.join(ML100K_DIR, 'seqs.npy')
SEQS_NEG = os.path.join(ML100K_DIR, 'seqsWithNeg.npy')