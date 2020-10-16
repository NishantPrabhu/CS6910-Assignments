import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import NuSVC
from plotting import DecisionRegionPlot

# Data preparation
train_clf = pd.read_csv("/home/nishant/Desktop/Semester 6/CS6910/Assignments/Assignment 1/data/2d_nonlinear/2d_nonlinear_data.csv")

X_train = train_clf[['x1', 'x2']].values.reshape((-1, 2))
y_train = train_clf['label'].values

# Scale data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Support vector classifier
clf = NuSVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
y_pred = y_pred

# Plot decision regions
DecisionRegionPlot(clf, X_train, y_train, y_pred)
