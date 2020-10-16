# Dependencies
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from warnings import filterwarnings
filterwarnings(action='ignore')


# This script defines plotting functions used to 
# visualize various results and intermediate conditions
# Available functions:
#   SpatialPlot : Plots the target with data points in space
#   ParityPlot : Plots target versus predictions


def SpatialPlot(X, y_true, y_pred):
    """
    [WARNING] Function assumes data is two dimensional. For any other
    dimensional data, plot will not be representative of complete situation.
    Blue dots represent actual data, red dots represent predictions.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y_true, zdir='z', s=20, c='blue', alpha=0.5, depthshade=True)    # True values
    ax.scatter(X[:, 0], X[:, 1], y_pred, zdir='z', s=20, c='red', alpha=0.5, depthshade=True)     # Predicted values
    plt.legend(['True', 'Predicted'])
    plt.show()


def ParityPlot(y_true, y_pred):
    """
    Plots actual values versus predicted values
    Helps determine goodness of fit for regression problems
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.scatter(y_pred, y_true, s=30, c='red', alpha=0.6)
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x, color='black')
    plt.xlabel('Predictions')
    plt.ylabel('Target')
    plt.title('Parity plot')
    plt.grid()
    plt.show()


def CallbackPlot(model, callback='Loss'):
    """
    Plots loss history and metric history
    """
    plt.figure(figsize=(8, 8))

    if callback == 'Loss':
        plt.plot(model.train_loss_history, color='blue', alpha=1.0)
        plt.plot(model.val_loss_history, color='orange', alpha=1.0)
    elif callback == 'Metric':
        plt.plot(model.train_metric_history, color='blue', alpha=1.0)
        plt.plot(model.val_metric_history, color='orange', alpha=1.0)
    
    plt.grid()
    plt.legend(['Train', 'Validation'])
    plt.title('{} history'.format(callback))
    plt.ylabel('{}'.format(callback))
    plt.xlabel('Epoch')
    plt.show()


def DecisionRegionPlot(model, X, y_true, y_pred, colors=['red', 'yellow', 'blue'], cmap=plt.cm.RdYlBu):
    """
    [WARNING]: Function assumes data is 2 dimensional
    """
    n_classes = len(np.unique(y_true))
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.05, X[:, 1].max() + 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z_preds = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array([np.argmax(i) for i in Z_preds])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 8))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.6)

    for i, color in zip(range(n_classes), colors):
        idx = np.where(y_true == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=i, edgecolor='black', s=20)

    plt.title('Decision region plot')
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.show()


def ConfusionMatrix(y_true, y_pred):
    """
    Calculates values of confusion matrix and renders on seaborn heatmap
    """
    y_true = y_true.reshape((1, -1))
    y_pred = y_pred.reshape((1, -1))
    data = np.vstack((y_true, y_pred)).T
    df = pd.DataFrame(data, columns=['a', 'p'])
    confusion_matrix = pd.crosstab(df['a'], df['p'], rownames=['Actual'], colnames=['Predicted'])
    plt.figure(figsize=(8, 8))
    sns.heatmap(confusion_matrix, annot=True, square=True, cmap='BuGn', fmt='g', linewidth=1, linecolor='black')
    plt.title('Confusion Matrix')
    plt.show()


def LayerOutputPlot(model, track_epochs, save_path):
    """
    Plots outputs of hidden layers of the model as surfaces
    """
    x1, x2 = np.meshgrid(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01))
    for epoch in track_epochs:
        for label in model.layer_outs[epoch].keys():
            for node in range(np.array(model.layer_outs[epoch][label]).shape[2]):
                y_grid = np.array(model.layer_outs[epoch][label])[:, :, node].reshape(x1.shape)

                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(x1, x2, y_grid, rstride=1, cstride=1, cmap='jet') 
                plt.title('Epoch {} - Layer {} - Node {}'.format(epoch, label, node))
                plt.savefig(save_path + 'epoch' + str(epoch) + '_' + label + '_node' + str(node) + '.png')





