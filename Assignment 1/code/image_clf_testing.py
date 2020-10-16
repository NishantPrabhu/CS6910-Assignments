import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
np.random.seed(1)

from models import Network
from layers import Dense, Input 
from losses import RMSE 
from activations import Sigmoid, ReLU, Tanh, Softmax, Linear
from optimizers import SGD, AdaGrad, Adam
from utils import OneHotEncoder, ImFeatureExtractor, MinMaxScaler
from plotting import SpatialPlot, ParityPlot, CallbackPlot, ConfusionMatrix


# Data preparation
load_path = '/home/nishant/Desktop/Semester 6/CS6910/Assignments/Assignment 1/data/image_data/train/'
save_path = '/home/nishant/Desktop/Semester 6/CS6910/Assignments/Assignment 1/data/image_data/FeatureExtraction_2D/'
save_dummy = '/home/nishant/Desktop/Semester 6/CS6910/Assignments/Assignment 1/data_team33/image_data/train/'
X_train, y_true, label_map = ImFeatureExtractor().load_features(save_dummy)
y_true = np.array(list(y_true)).reshape((-1, 1))
y_train = OneHotEncoder().transform(y_true)

# Scale data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Define layers in correct sequence
input_dim = X_train.shape[1]
layers = [
    Input(input_dim, label='Input'),
    Dense(64, activation=Sigmoid(), label='Hidden_1'),
    Dense(64, activation=Sigmoid(), label='Hidden_2'),
    Dense(len(label_map), activation=Softmax(), label='Output')
]


# Define model object and set layers
model = Network()
model.set_layers(layers)
model.compile(loss='CrossEntropy', optimizer=SGD(lr=0.001, momentum=0.9), metric='Accuracy')


# Model training
model.train(X_train, y_train, epochs=500, log_frequency=10, val_split=0.3)
y_probs = model.predict(X_train)
y_pred = np.array([np.argmax(i) for i in y_probs])

# Plot regression results
SpatialPlot(X_train, y_true, y_pred)

# Plot trend of loss function
CallbackPlot(model, 'Loss')
CallbackPlot(model, 'Metric')

# Confusion Matrix
ConfusionMatrix(y_true, y_pred)
