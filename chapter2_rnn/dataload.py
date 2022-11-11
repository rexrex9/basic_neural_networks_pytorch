import numpy as np
from data_set import filepaths as fp

def getTrainAndTestSeqs(inPath, test_ratio=0.1):
    seqs = np.load(inPath)

    allItems = set()
    for seq in seqs:
        allItems|=set(seq[:-1])

    np.random.shuffle(seqs)
    split_number = int(len(seqs)*test_ratio)
    test = seqs[:split_number]
    train = seqs[split_number:]
    return train, test, allItems


def getTrainAndTestSeqsWithNeg(seqInPath, test_ratio=0.1):
    seqs = np.load(seqInPath)

    allItems = set()
    for seq in seqs:
        allItems |= set(seq[0])

    np.random.shuffle(seqs)
    split_number = int(len(seqs) * test_ratio)
    test = seqs[:split_number]
    train = seqs[split_number:]
    return train, test, allItems

if __name__ == '__main__':


    #train, test, allItems = getTrainAndTestSeqsWithNeg(fp.Ml_latest_small.SEQS_NEG)
    #print(train)


    train,test,allItems = getTrainAndTestSeqs(fp.SEQS)
    print(train)