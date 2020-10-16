import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt

from models import Network
from layers import Dense, Input 
from losses import RMSE 
from activations import Sigmoid, ReLU, Tanh, Softmax, Linear
from optimizers import SGD, AdaGrad, Adam
from utils import OneHotEncoder, MinMaxScaler
from plotting import SpatialPlot, ParityPlot, CallbackPlot, DecisionRegionPlot, ConfusionMatrix, LayerOutputPlot


# Data preparation
train_clf = pd.read_csv("/home/nishant/Desktop/Semester 6/CS6910/Assignments/Assignment 1/data/2d_nonlinear/2d_nonlinear_data.csv")

X_train = train_clf[['x1', 'x2']].values.reshape((-1, 2))
y_true = train_clf['label'].values.reshape((-1, 1))
y_train = OneHotEncoder().transform(y_true)

# Scale data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)


# Define layers in correct sequence
layers = [
    Input(2, label='Input'),
    Dense(5, activation=Sigmoid(), label='Hidden_1'),
    Dense(5, activation=Sigmoid(), label='Hidden_2'),
    Dense(3, activation=Softmax(), label='Output')
]


# Define model object and set layers
model = Network()
model.set_layers(layers)
model.compile(loss='CrossEntropy', optimizer=SGD(lr=0.01, momentum=0.9), metric='Accuracy')


# Model training
track_epochs = [1, 2, 10, 50, 200]
model.train(X_train, y_train, epochs=200, log_frequency=10, val_split=0.3, track_epochs=track_epochs)
y_probs = model.predict(X_train)
y_pred = np.array([np.argmax(i) for i in y_probs])

# Plot regression results
SpatialPlot(X_train, y_true, y_pred)

# Plot trend of loss function
CallbackPlot(model, 'Loss')
CallbackPlot(model, 'Metric')

# Plot decision regions
DecisionRegionPlot(model, X_train, y_true, y_pred)

# Confusion matrix
ConfusionMatrix(y_true, y_pred)

# Layer output plots
# LayerOutputPlot(model, track_epochs=track_epochs, save_path='.././data/2d_nonlinear/plots/')

