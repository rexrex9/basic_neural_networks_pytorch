import numpy as np


def accuracy4classification(true_classes,pred_classes):
    return sum(np.array(true_classes)==np.array(pred_classes))/len(true_classes)

