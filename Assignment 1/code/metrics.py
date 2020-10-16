# Dependencies
import numpy as np 


# This script defines metrics which will be displayed during model training process
# Available metrics
#   RMSE
#   CrossEntropy
#   Accuracy


def metric_dict():
    metrics = {
        'RMSE': RMSE,
        'Crossentropy': CrossEntropy,
        'Accuracy': Accuracy
    }
    return metrics


def RMSE(y_true, y_pred):
    """
    Root mean squared error
    """
    return np.sum((y_true - y_pred)**2)/len(y_true)


def CrossEntropy(y_true, y_pred):
    """
    For classification tasks
    Expects each row of y_true and y_pred to be array of probabilities
    """
    return np.sum([-np.sum(np.multiply(y_true[i], np.log(y_pred[i]))) for i in range(len(y_true))])/len(y_true)


def Accuracy(y_true, y_pred):
    """
    Ratio of correct answers in y_pred with respect to y_true
    """
    return np.mean([np.argmax(y_true[i]) == np.argmax(y_pred[i]) for i in range(len(y_true))])
