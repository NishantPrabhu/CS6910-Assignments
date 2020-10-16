import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt

from models import Network
from layers import Dense, Input 
from activations import Sigmoid, ReLU, Tanh, Softmax, Linear
from optimizers import SGD, AdaGrad, Adam
from plotting import SpatialPlot, ParityPlot, CallbackPlot, LayerOutputPlot


# Data preparation
train_reg = pd.read_csv('/home/nishant/Desktop/Semester 6/CS6910/Assignments/Assignment 1/data/function_approx/train.csv')
val_reg = pd.read_csv('/home/nishant/Desktop/Semester 6/CS6910/Assignments/Assignment 1/data/function_approx/val.csv')

X_train = train_reg[['x1', 'x2']].values.reshape((-1, 2))
y_train = train_reg['y'].values.reshape((-1, 1))
X_val = val_reg[['x1', 'x2']].values.reshape((-1, 2))
y_val = val_reg['y'].values.reshape((-1, 1))


# Define layers in correct sequence
layers = [
    Input(2, label='Input'),
    Dense(8, activation=Sigmoid(), label='Hidden_1'),
    Dense(6, activation=Sigmoid(), label='Hidden_2'),
    Dense(1, activation=Linear(), label='Output')
]


# Define model object and set layers
model = Network()
model.set_layers(layers)
model.compile(loss='RMSE', optimizer=SGD(lr=2e-06, momentum=0.9), metric='RMSE')

# Model training
model.train(X_train, y_train, epochs=30000, log_frequency=1000, val_split=0.0, val_sets=[X_val, y_val], track_epochs=[30000])
y_pred = model.predict(X_train)

# Plot regression results
SpatialPlot(X_train, y_train, y_pred)
ParityPlot(y_train, y_pred)

# Plot trend of loss function
CallbackPlot(model, 'Loss')
CallbackPlot(model, 'Metric')

# Output surface plot
save_path = '/home/nishant/Desktop/Semester 6/CS6910/Assignments/Assignment 1/data/function_approx/plots/'
LayerOutputPlot(model, X_train, track_epochs=[30000], save_path=save_path)
