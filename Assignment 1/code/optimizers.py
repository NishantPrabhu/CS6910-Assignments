# Dependencies 
import numpy as np 


# This script defines optimizers that will be used in training
# Available optimizers
#   SGD
#   AdaGrad
#   AdaDelta
#   Adam


class SGD():
    def __init__(self, lr, momentum):
        self.lr = lr
        self.alpha = momentum 
        self.name = 'SGD'

    def get_update(self, layer, w_grad):
        return (-self.lr * w_grad) + (self.alpha * layer.w_last_update)


class AdaGrad():
    def __init__(self, lr, epsilon=1e-2):
        self.lr = lr
        self.epsilon = epsilon
        self.name = 'AdaGrad'

    def get_update(self, layer, w_grad):
        return (-self.lr * w_grad) / (self.epsilon + np.sqrt(layer.gss))


class AdaDelta():
    def __init__(self, rho=0.9, L=50, epsilon=0.1):
        self.rho = rho
        self.L = L
        self.epsilon = epsilon

    def get_update(self, layer, w_grad):
        nmr = np.power((self.rho/self.L)*np.sum(np.power(layer.w_history[-(self.L+1):-1], 2))+(1-self.rho)*(layer.w_history[-1])**2, 0.5)
        dmr = np.power((self.rho/self.L)*np.sum(np.power(layer.g_history[-(self.L+1):-1], 2))+(1-self.rho)*(layer.g_history[-1])**2, 0.5)
        return (nmr/(dmr+self.epsilon)) * w_grad


class Adam():
    def __init__(self, lr, rho_1=0.9, rho_2=0.999, epsilon=1e-2):
        self.lr = lr
        self.rho_1 = rho_1
        self.rho_2 = rho_2
        self.epsilon = epsilon
        self.name = 'Adam'

    def get_update(self, layer, w_grad):
        layer.q_vals = (self.rho_1)*(layer.q_vals) + (1-self.rho_1) * w_grad
        layer.r_vals = (self.rho_2)*(layer.r_vals) + (1-self.rho_2) * (w_grad**2)
        q_hat = layer.q_vals/(1 - self.rho_1**layer.x_count)
        r_hat = layer.r_vals/(1 - self.rho_2**layer.x_count)

        return (-self.lr * q_hat)/(self.epsilon + r_hat**0.5)
