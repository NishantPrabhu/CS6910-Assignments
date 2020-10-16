
# Dependencies
import numpy as np 


# This script defines various activation functions
# Available activations
#   Linear (NOT WORKING)
#   Sigmoid
#   ReLU (NOT WORKING)
#   Tanh
#   Softmax


class Linear():
    """
    Linear activation, returns whatever activation it gets
    """
    def __init__(self):
        pass

    def get_value(self, x):
        return x

    def grad(self, s):
        return np.identity(s.shape[0])


class Sigmoid():
    """
    Standard sigmoid activation
    beta is a weight hyperparameter
    """
    def __init__(self, c=1.0, beta=1.0):
        self.beta = beta
        self.c = c
    
    def get_value(self, z):
        y = np.exp(-self.beta * z)
        return (self.c / (1.0 + y))
    
    def grad(self, s):
        return np.identity(s.shape[0])*(1 - s)*s


class Tanh():
    """
    Tanh activation function
    """
    def __init__(self, c=1.0):
        self.c = c

    def get_value(self, z):
        return self.c * np.tanh(z)

    def grad(self, s):
        return np.identity(s.shape[0])*(1-s**2)


class ReLU():
    """
    Rectified linear unit activation
    alpha parameter can be set to non-zero value for LeakyReLU
    """
    def __init__(self):
        return

    def get_value(self, z):
        return (z * (z > 0))
    
    def grad(self, s):
        return np.identity(s.shape[0])*(s > 0).astype('int')


class Softmax():
    """
    Softmax activation with temperature = 1
    Use in output layers for classification problems
    """
    def __init__(self):
        pass
    
    def get_value(self, z):
        return np.exp(z)/np.sum(np.exp(z))
    
    def grad(self, s):
        return (np.identity(s.shape[0]) - s).T * s