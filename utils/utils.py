from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt

def equal_error_rate(gts, preds, plot=False):
    fpr, tpr, threshold = roc_curve(gts, preds)
    fnr = 1 - tpr

    eer = fpr[np.nanargmin(np.absolute(fnr-fpr))]
    eer_threshold = threshold[np.nanargmin(np.absolute(fnr-fpr))]

    if plot:
        plt.plot(threshold, fnr, label='FRR')
        plt.plot(threshold, fpr, label='FAR')
        plt.legend()
        plt.show()

    return eer, eer_threshold