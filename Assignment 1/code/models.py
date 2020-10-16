
# Dependencies
import numpy as np 

from losses import loss_dict
from metrics import metric_dict
losses_dict = loss_dict()
metrics_dict = metric_dict()


# This script defines the model object and all associated functions


class Network():

    def __init__(self):
        self.layers = None
        self.train_loss_history = list()
        self.train_metric_history = list()
        self.val_loss_history = list()
        self.val_metric_history = list()

    # SETUP FUNCTIONS ====================================================================================================

    def set_layers(self, layer_list):
        """
        Every element in the layer list is a layer object from the 
        layers script. When the list of layers is provided, weights 
        gradients associated with each layer are initialized here.  
        """
        for i in range(1, len(layer_list)):
            layer_list[i].weights = np.random.normal(loc=0, scale=1.0, size=layer_list[i-1].units*layer_list[i].units).reshape((layer_list[i-1].units, layer_list[i].units))
            layer_list[i].gradients = np.zeros((layer_list[i].units, 1))
            layer_list[i].activations = np.zeros((layer_list[i].units, 1))
            layer_list[i].bias = np.ones((layer_list[i].units, 1))
            layer_list[i].w_last_update = np.zeros(layer_list[i].weights.shape)
            layer_list[i].gss = np.zeros(layer_list[i].weights.shape)
            layer_list[i].w_history = [np.zeros(layer_list[i].weights.shape)]
            layer_list[i].g_history = [np.zeros(layer_list[i].weights.shape)]
            layer_list[i].q_vals = np.zeros(layer_list[i].weights.shape)
            layer_list[i].r_vals = np.zeros(layer_list[i].weights.shape)
        
        self.layers = layer_list

    def compile(self, optimizer, loss='RMSE', metric='RMSE'):
        """
        Sets the loss functions and metrics
        """
        self.loss = losses_dict[loss]
        self.metric_name = metric
        self.metrics = metrics_dict[self.metric_name]
        self.optim = optimizer

    # TRAINING FUNCTIONS ==================================================================================================

    def forward(self, x):
        """
        Forward pass function to obtain activations at output layer.
        """
        self.layers[0].activations = x.reshape((-1, 1))
        for i in range(1, len(self.layers)):
            self.layers[i].activations = self.layers[i].a_func.get_value(
                np.dot(self.layers[i].weights.T, self.layers[i-1].activations) + self.layers[i].bias
            )
    
    def backpropagate(self, y, x_count):
        
        # For output layer
        y = y.reshape(self.layers[-1].activations.shape)

        self.layers[-1].x_count = x_count + 1
        self.layers[-1].gradients = np.dot(self.layers[-1].a_func.grad(self.layers[-1].activations), self.loss(y, self.layers[-1].activations).grad())
        w_grad = np.dot(self.layers[-2].activations, self.layers[-1].gradients.T)
        self.layers[-1].weights += self.optim.get_update(self.layers[-1], w_grad)     
        self.layers[-1].bias += -self.optim.lr * self.layers[-1].gradients                             
        
        if self.optim.name == 'AdaDelta':    
            self.layers[-1].g_history.append(w_grad) 
            self.layers[-1].w_history.append(self.optim.get_update(self.layers[-1], w_grad))      
        elif self.optim.name == 'SGD':                                
            self.layers[-1].w_last_update = self.optim.get_update(self.layers[-1], w_grad)   
        elif self.optim.name == 'AdaGrad':
            self.layers[-1].gss += w_grad**2                                           
       
        # For other layers
        for i in np.arange(len(self.layers)-2, 0, -1):
            self.layers[i].x_count = x_count + 1
            self.layers[i].gradients = np.dot(self.layers[i].a_func.grad(self.layers[i].activations), np.dot(self.layers[i+1].weights, self.layers[i+1].gradients))
            w_grad = np.dot(self.layers[i-1].activations, self.layers[i].gradients.T)
            self.layers[i].weights += self.optim.get_update(self.layers[i], w_grad)
            self.layers[i].bias += -self.optim.lr * self.layers[i].gradients 
            
            if self.optim.name == 'AdaDelta':
                self.layers[i].g_history.append(w_grad)
                self.layers[i].w_history.append(self.optim.get_update(self.layers[i], w_grad))
            elif self.optim.name == 'SGD':
                self.layers[i].w_last_update = self.optim.get_update(self.layers[i], w_grad)
            elif self.optim.name == 'AdaGrad':
                self.layers[i].gss += w_grad**2
            
    def predict(self, X_test):
        """
        Performs forward pass on model with provided testing data.
        """
        preds = []
        for i in range(len(X_test)):
            self.forward(X_test[i])
            preds.append(self.layers[-1].activations)
        outputs = np.array(preds)
        return outputs.reshape(outputs.shape[:2])

    def train(self, X, y, epochs=20, val_split=0.0, val_sets=None, log_frequency=1, track_epochs=[]):
        """
        This function performs forward passes and backpropagation
        for specified number of epochs with specified learning rate.
        """
        # This section will keep track of layer outputs
        self.layer_outs = {}

        if val_split > 0:
            print('\nTraining on {} samples, validating on {} samples.\n'.format(
                    len(X) - int(val_split*len(X)),
                    int(val_split*len(X))
                ))
        elif val_sets is not None:
            print('\nTraining on {} samples, validating on {} samples.\n'.format(
                    len(X),
                    len(val_sets[0])
                ))
        else:
            print('\nTraining on {} samples.\n'.format(
                len(X)
            ))

        for epoch in range(epochs):

            if epoch+1 in track_epochs:
                layer_output_dict = {}
                for i in range(1, len(self.layers)):
                    layer_output_dict.update({self.layers[i].label: list()})
            
            if val_split > 0:
    
                val_index = np.random.choice([i for i in range(len(X))], size=int(val_split * len(X)), replace=False)
                X_val, y_val = X[val_index, :], y[val_index, :]
                X_train = X[[i for i in range(len(X)) if i not in val_index], :]
                y_train = y[[i for i in range(len(y)) if i not in val_index], :]
            
                for i in range(len(X_train)):
                    self.forward(X_train[i])
                    self.backpropagate(y_train[i], i)
                    
                if epoch+1 in track_epochs:
                    x1, x2 = np.arange(0, 1, 0.01), np.arange(0, 1, 0.01)
                    x1, x2 = np.meshgrid(x1, x2)
                    data = np.c_[x1.ravel(), x2.ravel()]
                    for i in range(len(data)):
                        self.forward(data[i])  
                        for j in range(1, len(self.layers)):
                            layer_output_dict[self.layers[j].label].append(self.layers[j].activations.reshape((1, -1)).tolist()) 
                    self.layer_outs.update({epoch+1: layer_output_dict})
                
                y_train_pred = self.predict(X_train)
                y_val_pred = self.predict(X_val)
            
                if epoch % log_frequency == 0:
                    print("Epoch {}/{} \t Train Loss : {:.6f} - Train {} : {:.6f} \t Val Loss : {:.6f} - Val {} : {:.6f}".format(
                        epoch,
                        epochs,
                        self.loss(y_train, y_train_pred).get_value(),
                        self.metric_name,
                        self.metrics(y_train, y_train_pred),
                        self.loss(y_val, y_val_pred).get_value(),
                        self.metric_name,
                        self.metrics(y_val, y_val_pred)
                    ))

                self.train_loss_history.append(self.loss(y_train, y_train_pred).get_value())
                self.train_metric_history.append(self.metrics(y_train, y_train_pred))
                self.val_loss_history.append(self.loss(y_val, y_val_pred).get_value())
                self.val_metric_history.append(self.metrics(y_val, y_val_pred))

            elif val_sets is not None:
                X_val, y_val = val_sets[0], val_sets[1]
                X_train, y_train = X, y

                for i in range(len(X_train)):
                    self.forward(X_train[i])
                    self.backpropagate(y_train[i], i)
                    
                if epoch+1 in track_epochs:
                    x1, x2 = np.arange(0, 1, 0.01), np.arange(0, 1, 0.01)
                    x1, x2 = np.meshgrid(x1, x2)
                    data = np.c_[x1.ravel(), x2.ravel()]
                    for i in range(len(data)):
                        self.forward(data[i])  
                        for j in range(1, len(self.layers)):
                            layer_output_dict[self.layers[j].label].append(self.layers[j].activations.reshape((1, -1)).tolist()) 
                    self.layer_outs.update({epoch+1: layer_output_dict})
                
                y_train_pred = self.predict(X_train)
                y_val_pred = self.predict(X_val)
            
                if epoch % log_frequency == 0:
                    print("Epoch {}/{} \t Train Loss : {:.6f} - Train {} : {:.6f} \t Val Loss : {:.6f} - Val {} : {:.6f}".format(
                        epoch,
                        epochs,
                        self.loss(y_train, y_train_pred).get_value(),
                        self.metric_name,
                        self.metrics(y_train, y_train_pred),
                        self.loss(y_val, y_val_pred).get_value(),
                        self.metric_name,
                        self.metrics(y_val, y_val_pred)
                    ))

                self.train_loss_history.append(self.loss(y_train, y_train_pred).get_value())
                self.train_metric_history.append(self.metrics(y_train, y_train_pred))
                self.val_loss_history.append(self.loss(y_val, y_val_pred).get_value())
                self.val_metric_history.append(self.metrics(y_val, y_val_pred))

            else:
                for i in range(len(X)):
                    self.forward(X[i])
                    self.backpropagate(y[i], i)
                
                if epoch+1 in track_epochs: 
                    x1, x2 = np.arange(0, 1, 0.01), np.arange(0, 1, 0.01)
                    x1, x2 = np.meshgrid(x1, x2)
                    data = np.c_[x1.ravel(), x2.ravel()]
                    for i in range(len(data)):
                        self.forward(data[i])  
                        for j in range(1, len(self.layers)):
                            layer_output_dict[self.layers[j].label].append(self.layers[j].activations.reshape((1, -1)).tolist()) 
                    self.layer_outs.update({epoch+1: layer_output_dict})
                
                y_train_pred = self.predict(X)
               
                if epoch % log_frequency == 0:
                    print("Epoch {}/{} \t Train Loss : {:.6f} - Train {} : {:.6f}".format(
                        epoch,
                        epochs,
                        self.loss(y, y_train_pred).get_value(),
                        self.metric_name,
                        self.metrics(y, y_train_pred),
                    ))

                self.train_loss_history.append(self.loss(y, y_train_pred).get_value())
                self.train_metric_history.append(self.metrics(y, y_train_pred))
    