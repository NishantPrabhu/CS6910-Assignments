import os
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle


# To set the load path for the extracted 2d features
# Change it to your team number
path = './Team_1'
class_names = os.listdir(path)

print(class_names)

# val is to assign different labels to each class
val = 0
# To store all the data_points
data_points = []
# To store all the corresponding class values
data_points_class = []

# for each class
for i in class_names:
    
    # Load the corresponding class_file
    load_name = os.path.join(path,i)
    extracted_features = np.load(load_name)
    
    # for each data_point in the class
    for j in extracted_features:
        
        # store it in the array
        data_points.append(j)
        # store the corresponding the class value
        data_points_class.append(val)
    
    # The print is to show the corresponding class names and labels assigned to the class labels      
    # Note the corresponding labels and class assigned 
    print(val,i)
    val += 1


# the stored data points are in a sequence which can affect the model performance
# To randomlly shuffle the examples as to maintain the corresponding class label

temp = list(zip(data_points,data_points_class))
shuffle(temp)


data_points,data_points_class = zip(*temp)
data_points = np.asanyarray(data_points)

# The final data_points in one array
print(data_points.shape)