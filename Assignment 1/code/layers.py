
# Dependencies
import numpy as np 


# This script defines layer objects to be added to the network object
# Available layers:
#   Input
#   Dense


class Input():
    """
    This layers accepts the data as input
    Need to figure the dimension of data that has to be passed to this
    """
    def __init__(self, units, label=None):
        self.units = units 
        self.label = label
        self.activations = None
        self.bias = None

    def name(self):
        return self.label

    def size(self):
        return self.units


class Dense():
    """
    Standard fully connected layer
    """
    def __init__(self, units, activation, label=None):
        self.units = units
        self.label = label
        self.a_func = activation
        self.activations = None 
        self.gradients = None 
        self.bias = None
        self.weights = None
        self.sgd_update = None          # Used in SGD optimizer
        self.gss = None                 # Used in AdaGrad 
        self.q_vals = None              # For Adam, expectation of gradients (q)
        self.r_vals = None              # For Adam, expectation of gradient squared (r)
        self.w_history = None           # To be used for AdaDelta
        self.g_history = None           # To be used for AdaDelta
        self.x_count = None             # Counter to keep track of number of input vectors passed
        
    def name(self):
        return self.label

    def size(self):
        return self.units

    def set_weights(self, weights):
        try:
            assert weights.shape == self.weights.shape
        except:
            raise ValueError('Shape {} of weights does not match expected shape {}'.format(
                                    weights.shape, 
                                    self.weights.shape
                                ))
        self.weights = weights


        
    



