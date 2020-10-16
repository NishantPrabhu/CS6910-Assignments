# Dependencies 
import numpy as np
import os
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.transforms as transforms
import torchvision.models as models

# Some utility functions
# Available functions
#   OneHotEncoder : One-hot encodes categorical outputs
#   MinMaxScaler : Scales data by min-max rule
#   ImageFeatureExtractor : Scripts given by course TAs


class OneHotEncoder():
    """
    Converts categorical data into one hot encoded vectors.
    Returns a numpy array.
    Can perform an inverse transform to get back labels from one-hot arrays.
    """
    def __init__(self):
        pass

    def transform(self, data):
        num_classes = len(np.unique(data))
        retval = []
        for label in data:
            retval.append([1.0 if i == label else 0.0 for i in range(num_classes)])
        return np.array(retval).astype(np.float32)

    def inverse_transform(self, data):
        return np.array([np.argmax(i) for i in data])


class MinMaxScaler():
    """
    Returns scaled matrix such that all values along columns lie between 0 and 1
    """
    def __init__(self):
        self.maxs = None
        self.mins = None
        self.dupl = None
    
    def fit(self, data):
        self.dupl = data.astype(np.float64)
        self.maxs = data.max(axis=0)
        self.mins = data.min(axis=0)
        
    def transform(self, data):
        for col in range(self.dupl.shape[1]):
            self.dupl[:, col] = (data[:, col] - self.mins[col])/(self.maxs[col] - self.mins[col])
        return self.dupl
                             
    def inverse_transform(self, data):
        for col in range(data.shape[1]):
            self.dupl[:, col] = self.mins[col] + data[:, col]*(self.maxs[col] - self.mins[col])
        return self.dupl

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


class ImFeatureExtractor():
    """
    Based on functions provided for extraction of features
    from image data.
    """
    def __init__(self):
        pass

    def generate_features(self, load_path, save_path):
        net = models.vgg16_bn(pretrained=True)
        os.mkdir(save_path)
        get_class_names = os.listdir(load_path)

        for i in get_class_names:
            new_save_path = save_path + i
            class_path = load_path + i
            img = np.load(class_path)
            arr = []

            for j in img:   
                j = torch.tensor(j)
                j = j.view([-1, 3, 32, 32])
                j = F.interpolate(j, (224, 224))
                z = net.features(j)
                m = F.avg_pool2d(z, (7, 7), 1, 0)
                m = m.view([-1]).detach()
                m = np.asarray(m)
                arr.append(m)

            arr = np.asarray(arr)
            np.save(new_save_path, arr)

    def load_features(self, feature_path):
        path = feature_path
        self.class_names = os.listdir(path)
        self.label_map = {}
        val = 0
        data_points = []
        data_points_class = []
       
        for i in self.class_names:
            load_name = os.path.join(path, i)
            extracted_features = np.load(load_name)
            
            for j in extracted_features:
                data_points.append(j)
                data_points_class.append(val)
            
            self.label_map.update({val: i})
            val += 1

        temp = list(zip(data_points,data_points_class))
        shuffle(temp)

        data_points, data_points_class = zip(*temp)
        data_points = np.asanyarray(data_points)
        return data_points, data_points_class, self.label_map

    def get_features(self, load_path, save_path):
        """
        Combines the two functions above this.
        """
        print("\n[INFO] Creating features ...\n")
        self.generate_features(load_path, save_path)
        print("\n[INFO] Loading features ...\n")
        features, labels, label_map = self.load_features(save_path)
        return features, labels, label_map


