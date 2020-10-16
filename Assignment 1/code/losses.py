
# Dependencies
import numpy as np 


# This script defines some loss functions
# Available loss functions
#   loss_dict : Dictionary with all losses
#   RMSE
#   CrossEntropy


def loss_dict():
    losses = {
        'RMSE': RMSE,
        'CrossEntropy': CrossEntropy
    }
    return losses


class RMSE():

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.N = len(y_true)
 
    def get_value(self):
        return np.sum((self.y_true - self.y_pred)**2)/self.N

    def grad(self):
        return -(self.y_true - self.y_pred)


class CrossEntropy():
    
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred.reshape(y_true.shape)
        self.N = len(y_true)

    def get_value(self):
        return -np.sum(self.y_true * np.log(self.y_pred))/self.N

    def grad(self):
        return -(self.y_true/(self.y_pred + 1e-9))

    